from __future__ import annotations

from pathlib import Path
from typing import Any

from .deepseek_planner import (
    DeepSeekPlannerError,
    audit_scene_plan,
    create_scene_plan as create_deepseek_plan,
)
from .openai_visuals import OpenAIVisualError, create_scene_plan as create_openai_plan


class VisualPlannerError(RuntimeError):
    pass


def create_visual_plan(
    script: str,
    shots: list[dict[str, Any]],
    planner: dict[str, Any],
    env: dict[str, str],
    diagnostics_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = str(planner.get("provider", "openai")).lower()
    secret_ref = str(
        planner.get("secret_ref")
        or ("DEEPSEEK_API_KEY" if provider in {"deepseek", "chat_completions"} else "OPENAI_API_KEY")
    )
    try:
        if provider in {"deepseek", "chat_completions"}:
            return create_deepseek_plan(
                script,
                shots,
                planner,
                env.get(secret_ref, ""),
                diagnostics_path=diagnostics_path,
            )
        if provider == "openai":
            return create_openai_plan(script, shots, planner, env.get(secret_ref, ""))
    except (DeepSeekPlannerError, OpenAIVisualError) as exc:
        raise VisualPlannerError(str(exc)) from exc
    raise VisualPlannerError(
        f"不支持的场景规划 provider：{provider}。可用值为 deepseek 或 openai。"
    )


def create_visual_plan_with_audit(
    script: str,
    shots: list[dict[str, Any]],
    planner: dict[str, Any],
    env: dict[str, str],
    diagnostics_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, safe = create_visual_plan(
        script, shots, planner, env, diagnostics_path=diagnostics_path
    )
    audit_config = dict(planner.get("semantic_audit", {}))
    if not audit_config.get("enabled", True):
        plan["semantic_audit_status"] = "disabled"
        return plan, safe, {"status": "disabled", "attempts": []}
    audit_provider = str(audit_config.get("provider", "deepseek")).lower()
    if audit_provider not in {"deepseek", "chat_completions"}:
        raise VisualPlannerError(
            f"当前只支持 DeepSeek 计划语义审校，收到：{audit_provider}"
        )
    failure_policy = str(audit_config.get("failure_policy", "manual_review")).lower()
    try:
        plan, audit = audit_scene_plan(
            script,
            plan,
            planner,
            env.get(
                str(audit_config.get("secret_ref") or planner.get("secret_ref") or "DEEPSEEK_API_KEY"),
                "",
            ),
        )
    except DeepSeekPlannerError as exc:
        if failure_policy != "manual_review":
            raise VisualPlannerError(str(exc)) from exc
        plan["semantic_audit_status"] = "unavailable"
        audit = {
            "status": "unavailable",
            "attempts": [],
            "message": str(exc),
        }
        print(f"      计划语义审校不可用，将强制进入人工审核：{exc}")
    return plan, safe, audit
