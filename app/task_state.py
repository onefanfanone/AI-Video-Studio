from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TASK_STAGES = (
    "preflight",
    "voice",
    "alignment",
    "captions",
    "visual_plan",
    "asset_search",
    "asset_semantic_review",
    "ai_fallback",
    "asset_review",
    "asset_download",
    "license_audit",
    "storyboard",
    "clips",
    "preview",
    "draft",
    "validation",
)

REUSE_TASK_STAGES = (
    "preflight",
    "voice",
    "alignment",
    "captions",
    "visual_plan",
    "asset_reuse_match",
    "asset_reuse_review",
    "asset_search",
    "asset_semantic_review",
    "ai_fallback",
    "asset_review",
    "asset_download",
    "license_audit",
    "storyboard",
    "clips",
    "preview",
    "draft",
    "validation",
)


def _stage_record() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "input_hash": None,
        "started_at": None,
        "finished_at": None,
        "artifacts": [],
        "error": None,
    }


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


class TaskState:
    """A small durable task record for one local build."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        task_id: str,
        run_id: str,
        project_id: str,
        input_hash: str,
        options: dict[str, Any] | None = None,
        stage_order: Iterable[str] | None = None,
    ) -> "TaskState":
        now = _now()
        ordered_stages = tuple(stage_order or TASK_STAGES)
        if not ordered_stages or len(set(ordered_stages)) != len(ordered_stages):
            raise ValueError("任务阶段顺序为空或包含重复阶段。")
        task = cls(
            path,
            {
                "schema_version": 1,
                "task_id": task_id,
                "run_id": run_id,
                "project_id": project_id,
                "input_hash": input_hash,
                "options": options or {},
                "stage_order": list(ordered_stages),
                "status": "running",
                "current_stage": "preflight",
                "created_at": now,
                "updated_at": now,
                "error": None,
                "stages": {name: _stage_record() for name in ordered_stages},
            },
        )
        task.save()
        return task

    @classmethod
    def load(cls, path: Path) -> "TaskState":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(f"不支持的 task.json 版本：{path}")
        # Earlier tasks did not persist stage_order. They remain 16-stage tasks;
        # derived projects explicitly opt into the 18-stage order.
        order = data.get("stage_order")
        if not isinstance(order, list) or not order:
            order = list(TASK_STAGES)
            data["stage_order"] = order
        for name in order:
            data.setdefault("stages", {}).setdefault(str(name), _stage_record())
        return cls(path, data)

    @property
    def stage_order(self) -> tuple[str, ...]:
        order = self.data.get("stage_order")
        return tuple(str(name) for name in order) if isinstance(order, list) else TASK_STAGES

    def save(self) -> None:
        self.data["updated_at"] = _now()
        _atomic_json(self.path, self.data)

    def begin(self, stage: str, input_hash: str) -> None:
        record = self.data["stages"][stage]
        record["status"] = "running"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["input_hash"] = input_hash
        record["started_at"] = _now()
        record["finished_at"] = None
        record["error"] = None
        self.data["status"] = "running"
        self.data["current_stage"] = stage
        self.data["error"] = None
        self.save()

    def succeed(
        self, stage: str, input_hash: str, artifacts: Iterable[Path | str] = ()
    ) -> None:
        record = self.data["stages"][stage]
        record["status"] = "succeeded"
        record["input_hash"] = input_hash
        record["finished_at"] = _now()
        record["artifacts"] = [str(item) for item in artifacts]
        record["error"] = None
        self.save()

    def fail(self, stage: str, error: str) -> None:
        if stage in self.data["stages"]:
            record = self.data["stages"][stage]
            record["status"] = "failed"
            record["finished_at"] = _now()
            record["error"] = error
        self.data["status"] = "failed"
        self.data["current_stage"] = stage
        self.data["error"] = error
        self.save()

    def complete(self) -> None:
        self.data["status"] = "succeeded"
        self.data["current_stage"] = None
        self.data["error"] = None
        self.data["finished_at"] = _now()
        self.save()

    def wait_for_review(
        self, stage: str, input_hash: str, artifacts: Iterable[Path | str] = ()
    ) -> None:
        record = self.data["stages"][stage]
        record["status"] = "waiting_for_review"
        record["input_hash"] = input_hash
        record["finished_at"] = None
        record["artifacts"] = [str(item) for item in artifacts]
        record["error"] = None
        self.data["status"] = "waiting_for_review"
        self.data["current_stage"] = stage
        self.data["error"] = None
        self.save()

    def can_reuse(self, stage: str, input_hash: str) -> bool:
        record = self.data["stages"].get(stage, {})
        if record.get("status") != "succeeded" or record.get("input_hash") != input_hash:
            return False
        for artifact_text in record.get("artifacts", []):
            artifact = Path(artifact_text)
            if not artifact.exists():
                return False
            if artifact.is_file() and artifact.stat().st_size == 0:
                return False
        return True

    def invalidate_after(self, stage: str) -> None:
        """Mark downstream stages pending when an upstream artifact is repaired."""
        try:
            start = self.stage_order.index(stage) + 1
        except ValueError as exc:
            raise ValueError(f"未知任务阶段：{stage}") from exc
        for name in self.stage_order[start:]:
            record = self.data["stages"][name]
            record.update(
                {
                    "status": "pending",
                    "input_hash": None,
                    "started_at": None,
                    "finished_at": None,
                    "artifacts": [],
                    "error": None,
                }
            )
        self.save()


def find_task_dir(
    outputs_root: Path,
    token: str,
    *,
    project_id: str | None = None,
    draft_root: Path | None = None,
    input_hash: str | None = None,
) -> Path:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for task_path in outputs_root.glob("*/task.json"):
        try:
            data = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if project_id and data.get("project_id") != project_id:
            continue
        if input_hash and data.get("input_hash") != input_hash:
            continue
        stored_draft_root = data.get("options", {}).get("draft_root")
        if (
            token == "latest"
            and draft_root is not None
            and (
                not stored_draft_root
                or os.path.normcase(str(Path(stored_draft_root).resolve()))
                != os.path.normcase(str(draft_root.resolve()))
            )
        ):
            continue
        if token != "latest" and token not in {
            str(data.get("task_id")),
            str(data.get("run_id")),
            task_path.parent.name,
        }:
            continue
        if token == "latest" and data.get("status") == "succeeded":
            continue
        candidates.append((task_path.stat().st_mtime, task_path.parent, data))
    if not candidates:
        suffix = f"（项目 {project_id}）" if project_id else ""
        raise FileNotFoundError(f"找不到可续跑任务：{token}{suffix}")
    return max(candidates, key=lambda item: item[0])[1]
