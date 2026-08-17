from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .comfyui_client import (
    ComfyUIError,
    generate_image as generate_comfyui_image,
    workflow_fingerprint,
)
from .asset_sources import (
    AssetSourceError,
    RIGHTS_LINKS,
    download_bytes,
    normalize_met_object,
    normalize_smithsonian_row,
    normalize_wikimedia_page,
    search_assets,
    verify_image_bytes,
    _request_json,
)
from .openai_visuals import OpenAIVisualError, generate_image as generate_openai_image


class VisualSupplyError(RuntimeError):
    pass


def resolve_ai_image_limit(
    config: dict[str, Any], provider: str, shot_count: int
) -> int:
    raw_limit = config.get("max_images_per_run", 4)
    if isinstance(raw_limit, str):
        dynamic_limit = raw_limit.lower()
        if dynamic_limit not in {"all_shots", "all_candidates"}:
            raise VisualSupplyError(
                "max_images_per_run 必须是正整数、all_shots 或 all_candidates。"
            )
        if provider != "comfyui_local":
            raise VisualSupplyError(
                "all_shots 动态额度只允许用于本机 comfyui_local；"
                "外部付费生图必须配置固定整数上限。"
            )
        headroom = int(config.get("regeneration_headroom", 4))
        if headroom < 0:
            raise VisualSupplyError("regeneration_headroom 不能小于 0。")
        multiplier = (
            int(config.get("candidates_per_shot", 1))
            if dynamic_limit == "all_candidates"
            else 1
        )
        if multiplier < 1 or multiplier > 8:
            raise VisualSupplyError("candidates_per_shot 必须在 1–8 之间。")
        return max(1, int(shot_count) * multiplier + headroom)
    maximum = int(raw_limit)
    if maximum < 1:
        raise VisualSupplyError("max_images_per_run 必须大于 0。")
    return maximum


def select_ai_candidate_targets(
    candidates: dict[str, Any], candidate_policy: str
) -> list[dict[str, Any]]:
    policy = str(candidate_policy or "gaps").strip().lower()
    shots = list(candidates.get("shots", []))
    if policy == "gaps":
        return [shot for shot in shots if not shot.get("recommended_asset_id")]
    if policy == "all_shots":
        return shots
    raise VisualSupplyError("candidate_policy 必须是 gaps 或 all_shots。")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_search_results(
    scene_plan: dict[str, Any], search_config: dict[str, Any], env: dict[str, str]
) -> dict[str, Any]:
    try:
        return search_assets(scene_plan, search_config, env)
    except AssetSourceError as exc:
        raise VisualSupplyError(str(exc)) from exc


def build_ai_only_candidates(
    scene_plan: dict[str, Any], search_config: dict[str, Any]
) -> dict[str, Any]:
    """Create the normal candidate envelope without making any museum request."""
    provider_status = {
        str(provider): {
            "status": "skipped",
            "message": "项目选择纯 AI 画面，未发起馆藏网络搜索。",
            "count": 0,
            "successful_queries": 0,
            "failed_queries": 0,
        }
        for provider in search_config.get(
            "providers", ["met", "smithsonian", "wikimedia", "openverse"]
        )
    }
    shots: list[dict[str, Any]] = []
    for intent in scene_plan.get("shots", []):
        shots.append(
            {
                "shot_id": intent["shot_id"],
                "intent_id": intent["intent_id"],
                "start": intent.get("start", 0.0),
                "end": intent.get("end", 0.0),
                "narration": intent.get("narration", ""),
                "query": "",
                "successful_providers": [],
                "museum_source_succeeded": False,
                "recommended_asset_id": None,
                "candidates": [],
                "time_context": intent.get("time_context", {}),
                "must_include": intent.get("must_include", []),
                "avoid": intent.get("avoid", []),
                "semantic_review_status": "not_applicable",
            }
        )
    return {
        "schema_version": 1,
        "searched_at": None,
        "visual_strategy": "ai_only",
        "search_skipped": True,
        "search_skip_reason": "ai_only",
        "at_least_one_source_succeeded": False,
        "provider_status": provider_status,
        "semantic_review_status": "not_applicable",
        "shots": shots,
    }


def add_ai_fallbacks(
    candidates: dict[str, Any],
    scene_plan: dict[str, Any],
    config: dict[str, Any],
    credentials: dict[str, str] | str,
    cache_root: Path,
    provenance_root: Path,
    project_dir: Path | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    if not config.get("enabled", True):
        return candidates, []
    ai_only = str(candidates.get("visual_strategy", "")) == "ai_only"
    provider = str(config.get("provider", "openai")).lower()
    if provider not in {"openai", "comfyui_local"}:
        raise VisualSupplyError(
            f"不支持的 AI 生图 provider：{provider}。可用值为 comfyui_local 或 openai。"
        )
    candidate_policy = str(config.get("candidate_policy", "gaps")).strip().lower()
    # ``all_shots`` is an explicit instruction to create a local candidate for
    # every shot, independent of whether museum search found anything. It is
    # therefore safe to continue through a museum outage: no paid call is
    # inferred from a failed search. Gap-based fallback and external providers
    # remain fail-closed so that an outage can never be mistaken for "no result".
    local_all_shots = provider == "comfyui_local" and candidate_policy == "all_shots"
    if (
        not ai_only
        and not candidates.get("at_least_one_source_succeeded")
        and not local_all_shots
    ):
        raise VisualSupplyError(
            "所有馆藏源均未成功完成搜索。为避免把服务故障误判为缺图，本次不会触发 AI 生图。"
        )
    targets = select_ai_candidate_targets(candidates, candidate_policy)
    gaps = [shot for shot in candidates.get("shots", []) if not shot.get("recommended_asset_id")]
    unavailable = [
        shot for shot in gaps if not shot.get("museum_source_succeeded", True)
    ]
    if unavailable and not ai_only and not local_all_shots:
        shot_numbers = ", ".join(str(item.get("shot_id")) for item in unavailable)
        raise VisualSupplyError(
            f"镜头 {shot_numbers} 没有任何馆藏 provider 成功完成搜索；"
            "这属于服务故障而非零结果，本次不会触发 AI 生图。"
        )
    if unavailable and local_all_shots:
        outage_shots = [int(item["shot_id"]) for item in unavailable]
        candidates["museum_outage_ai_override"] = {
            "policy": "local_comfyui_all_shots",
            "shot_ids": outage_shots,
            "reason": (
                "馆藏搜索服务故障；按用户明确选择的本机 ComfyUI 全镜头候选策略继续，"
                "未把故障解释为馆藏零结果。"
            ),
        }
        shot_numbers = ", ".join(str(item) for item in outage_shots)
        print(
            f"      [提示] 镜头 {shot_numbers} 的馆藏搜索服务异常；"
            "按“本机 ComfyUI＋所有镜头”策略继续生成 AI 候选，故障状态将保留供审核。"
        )
    env = (
        credentials
        if isinstance(credentials, dict)
        else {"OPENAI_API_KEY": str(credentials)}
    )
    project_dir = (project_dir or Path.cwd()).resolve()
    maximum = resolve_ai_image_limit(config, provider, len(candidates.get("shots", [])))
    candidates_per_shot = int(config.get("candidates_per_shot", 1))
    if candidates_per_shot < 1 or candidates_per_shot > 8:
        raise VisualSupplyError("candidates_per_shot 必须在 1–8 之间。")
    total_jobs = len(targets) * candidates_per_shot
    call_kind = "付费调用" if provider == "openai" else "本机生图任务"
    if total_jobs > maximum:
        raise VisualSupplyError(
            f"预计需要生成 {total_jobs} 张 AI 图，超过单次上限 {maximum}；"
            f"已在任何{call_kind}发生前中止。请先优化检索词或人工补图。"
        )
    intents = {int(item["shot_id"]): item for item in scene_plan.get("shots", [])}
    generated: list[Path] = []
    provenance_root.mkdir(parents=True, exist_ok=True)
    ai_cache = cache_root / "ai"
    ai_cache.mkdir(parents=True, exist_ok=True)
    job_index = 0
    for shot in targets:
        intent = intents[int(shot["shot_id"])]
        fixed = (
            " Vertical 9:16 composition, cinematic photorealistic historical reconstruction. "
            "Period-accurate materials and clothing. Leave a calm, uncluttered subtitle safe area "
            "across the lower 22 percent. No text, no lettering, no watermark, "
            "no logos, no brands, and no unnecessary likeness of a famous person."
        )
        dynamic_constraints = (
            " Must visibly include: " + "; ".join(intent.get("must_include", [])) + "."
            " Avoid: " + "; ".join(intent.get("avoid", [])) + "."
        )
        base_prompt = str(intent.get("ai_prompt") or "") + dynamic_constraints + fixed
        if provider == "comfyui_local":
            try:
                workflow_sha = workflow_fingerprint(config, project_dir)
            except ComfyUIError as exc:
                raise VisualSupplyError(str(exc)) from exc
            generation_fingerprint = {
                "provider": provider,
                "workflow_sha256": workflow_sha,
                "width": int(config.get("width", 768)),
                "height": int(config.get("height", 1344)),
            }
        else:
            generation_fingerprint = {
                "provider": provider,
                "model": config.get("model", "gpt-image-2"),
                "size": config.get("size", "1024x1536"),
                "quality": config.get("quality", "medium"),
                "format": config.get("output_format", "jpeg"),
            }
        for variation_index in range(1, candidates_per_shot + 1):
            job_index += 1
            variation_instruction = ""
            if variation_index > 1:
                variation_instruction = (
                    f" Create composition variation {variation_index} of {candidates_per_shot}: "
                    "keep the same historical facts and required objects, but use a distinctly "
                    "different camera distance, angle, subject placement, and foreground layering."
                )
            prompt = base_prompt + variation_instruction
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "prompt": prompt,
                        "generation": generation_fingerprint,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            image_path = ai_cache / f"{cache_key}.jpg"
            metadata_path = ai_cache / f"{cache_key}.json"
            cached = image_path.is_file() and metadata_path.is_file()
            if cached:
                try:
                    verify_image_bytes(
                        image_path.read_bytes(),
                        "image/jpeg",
                        min_long_edge=1,
                        min_short_edge=1,
                    )
                except (AssetSourceError, OSError):
                    cached = False
            if cached:
                print(
                    f"      AI 画面 {job_index}/{total_jobs}：镜头 {shot['shot_id']} "
                    f"候选 {variation_index}/{candidates_per_shot} 复用缓存"
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                print(
                    f"      AI 画面 {job_index}/{total_jobs}：镜头 {shot['shot_id']} "
                    f"候选 {variation_index}/{candidates_per_shot} 正在通过 {provider} 生成..."
                )
                try:
                    if provider == "comfyui_local":
                        seed = int(cache_key[:16], 16)
                        data, metadata = generate_comfyui_image(
                            prompt, config, project_dir, seed=seed
                        )
                    else:
                        data, metadata = generate_openai_image(
                            prompt,
                            config,
                            env.get(str(config.get("secret_ref") or "OPENAI_API_KEY"), ""),
                        )
                except (ComfyUIError, OpenAIVisualError) as exc:
                    raise VisualSupplyError(str(exc)) from exc
                try:
                    verify_image_bytes(
                        data,
                        "image/jpeg",
                        min_long_edge=1,
                        min_short_edge=1,
                    )
                except AssetSourceError as exc:
                    raise VisualSupplyError(f"AI 生图返回了损坏图片：{exc}") from exc
                image_path.write_bytes(data)
                _atomic_json(metadata_path, metadata)
            try:
                width, height, _ = verify_image_bytes(
                    image_path.read_bytes(),
                    "image/jpeg",
                    min_long_edge=1,
                    min_short_edge=1,
                )
            except AssetSourceError as exc:
                raise VisualSupplyError(
                    f"AI 图片缓存无法完整解码：{image_path.name}；{exc}"
                ) from exc
            print(
                f"      AI 画面 {job_index}/{total_jobs}：镜头 {shot['shot_id']} "
                f"候选 {variation_index}/{candidates_per_shot} 完成（{width}x{height}）"
            )
            is_comfyui = provider == "comfyui_local"
            asset_id = f"{'comfyui' if is_comfyui else 'openai'}-{cache_key[:20]}"
            generation = {
                **metadata,
                "variation_index": variation_index,
                "variations_per_shot": candidates_per_shot,
            }
            candidate = {
            "asset_id": asset_id,
            "provider": provider,
            "source_id": metadata.get("request_id") or cache_key,
            "title": (
                f"AI historical reconstruction for shot {shot['shot_id']} "
                f"· variation {variation_index}"
            ),
            "creator": "Local ComfyUI workflow" if is_comfyui else "OpenAI image model",
            "institution": "Local ComfyUI" if is_comfyui else "OpenAI",
            "source_page": (
                str(config.get("server_url", "http://127.0.0.1:8000")).rstrip("/") + "/"
                if is_comfyui
                else "https://developers.openai.com/api/docs/guides/image-generation"
            ),
            "download_url": "",
            "thumbnail_url": "",
            "rights_code": "provider_terms",
            "rights_url": (
                str(config.get("model_license_url") or "http://127.0.0.1:8000/")
                if is_comfyui
                else "https://openai.com/policies/service-terms/"
            ),
            "width": width,
            "height": height,
            "mime": "image/jpeg",
            "selectable": True,
            "requires_reverification": False,
            "ai_generated": True,
            "semantic_status": "ai_unreviewed",
            "semantic_requires_override": False,
            "semantic_score": None,
            "semantic_review": {
                "verdict": "not_applicable",
                "conflicts": [],
                "reason": "AI 候选未经过图像视觉语义审校，需由人工查看画面。",
            },
            "score": 70,
            "score_detail": {"ai_fallback": 70},
            "local_preview": str(image_path),
            "generation": generation,
            "raw_metadata": {"generation": generation},
            }
            shot_candidates = shot.setdefault("candidates", [])
            existing_ids = {str(item.get("asset_id")) for item in shot_candidates}
            if asset_id not in existing_ids:
                ai_count = sum(1 for item in shot_candidates if item.get("ai_generated"))
                shot_candidates.insert(ai_count, candidate)
            semantic_unavailable = (
                shot.get("semantic_review_status") == "unavailable"
                or candidates.get("semantic_review_status")
                in {"unavailable", "plan_unavailable"}
            )
            if (
                not ai_only
                and not semantic_unavailable
                and not shot.get("recommended_asset_id")
                and variation_index == 1
            ):
                shot["recommended_asset_id"] = asset_id
            provenance = provenance_root / f"{asset_id}-generation.json"
            _atomic_json(provenance, generation)
            generated.extend([image_path, provenance])
            if checkpoint_path is not None:
                _atomic_json(checkpoint_path, candidates)
    candidates["ai_generated_count"] = sum(
        1
        for shot in candidates.get("shots", [])
        for item in shot.get("candidates", [])
        if item.get("ai_generated")
    )
    candidates["ai_candidate_policy"] = str(config.get("candidate_policy", "gaps"))
    candidates["ai_candidates_per_shot"] = candidates_per_shot
    candidates["ai_generation_limit"] = maximum
    return candidates, generated


def apply_review_request(
    candidates: dict[str, Any],
    scene_plan: dict[str, Any],
    request: dict[str, Any],
    visuals_config: dict[str, Any],
    env: dict[str, str],
    cache_root: Path,
    provenance_root: Path,
    project_dir: Path | None = None,
) -> dict[str, Any]:
    """Handle an explicit per-shot re-search or re-generation from the review UI."""
    shot_id = int(request["shot_id"])
    action = str(request["action"])
    note = str(request.get("note") or "").strip()
    target = next(
        (item for item in candidates.get("shots", []) if int(item["shot_id"]) == shot_id),
        None,
    )
    intent = next(
        (item for item in scene_plan.get("shots", []) if int(item["shot_id"]) == shot_id),
        None,
    )
    if not target or not intent:
        raise VisualSupplyError(f"审核请求引用了不存在的镜头 {shot_id}。")
    if action == "search":
        if str(candidates.get("visual_strategy", "")) == "ai_only":
            raise VisualSupplyError(
                "当前项目选择纯 AI 画面，不会发起馆藏网络重搜；请使用“再生此镜头”。"
            )
        revised_intent = json.loads(json.dumps(intent, ensure_ascii=False))
        if note:
            revised_intent["search_terms_en"] = [note, *revised_intent.get("search_terms_en", [])]
            revised_intent["search_terms_zh"] = [note, *revised_intent.get("search_terms_zh", [])]
        else:
            revised_intent["search_terms_en"] = [
                *revised_intent.get("search_terms_en", []),
                "museum collection alternate view",
            ]
        refreshed = build_search_results(
            {"shots": [revised_intent]}, visuals_config.get("search", {}), env
        )["shots"][0]
        merged = {
            str(item["asset_id"]): item
            for item in [*refreshed.get("candidates", []), *target.get("candidates", [])]
        }
        target["candidates"] = sorted(
            merged.values(), key=lambda item: (-int(item.get("score", 0)), str(item["asset_id"]))
        )[: int(visuals_config.get("search", {}).get("candidates_per_shot", 6))]
        target["query"] = refreshed.get("query", target.get("query"))
        target["museum_source_succeeded"] = refreshed.get("museum_source_succeeded", False)
        target["successful_providers"] = refreshed.get("successful_providers", [])
    elif action == "regenerate":
        ai_config = visuals_config.get("ai_fallback", {})
        existing_count = sum(
            1
            for shot in candidates.get("shots", [])
            for item in shot.get("candidates", [])
            if item.get("ai_generated")
        )
        provider = str(ai_config.get("provider", "openai")).lower()
        maximum = resolve_ai_image_limit(
            ai_config, provider, len(candidates.get("shots", []))
        )
        if existing_count >= maximum:
            call_kind = "付费调用" if provider == "openai" else "本机生图任务"
            raise VisualSupplyError(
                f"本次任务已经生成 {existing_count} 张 AI 图，达到上限 {maximum}；"
                f"未发生新的{call_kind}。"
            )
        revised_intent = json.loads(json.dumps(intent, ensure_ascii=False))
        variation = note or f"Create a distinctly different composition variation {existing_count + 1}."
        revised_intent["ai_prompt"] = str(revised_intent.get("ai_prompt", "")) + " " + variation
        mini = {
            "at_least_one_source_succeeded": True,
            "shots": [
                {
                    "shot_id": shot_id,
                    "intent_id": intent["intent_id"],
                    "narration": intent.get("narration", ""),
                    "museum_source_succeeded": True,
                    "recommended_asset_id": None,
                    "candidates": [],
                }
            ],
        }
        mini, _ = add_ai_fallbacks(
            mini,
            {"shots": [revised_intent]},
            {
                **ai_config,
                "candidates_per_shot": 1,
                "max_images_per_run": 1,
            },
            env,
            cache_root,
            provenance_root,
            project_dir,
        )
        generated = mini["shots"][0]["candidates"][0]
        target.setdefault("candidates", []).insert(0, generated)
        if str(candidates.get("visual_strategy", "")) != "ai_only":
            target["recommended_asset_id"] = generated["asset_id"]
    else:
        raise VisualSupplyError(f"未知审核动作：{action}")
    history_path = provenance_root / "asset-review-requests.json"
    history = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(request)
    _atomic_json(history_path, history)
    return candidates


def _refetch_candidate(candidate: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    provider = str(candidate.get("provider"))
    if provider in {"openai", "comfyui_local"}:
        return candidate
    if provider == "met":
        item = _request_json(
            "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
            + str(candidate["source_id"])
        )
        refreshed = normalize_met_object(item)
    elif provider == "wikimedia":
        import urllib.parse

        params = urllib.parse.urlencode(
            {
                "action": "query", "format": "json", "formatversion": 2,
                "pageids": str(candidate["source_id"]), "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 480,
            }
        )
        payload = _request_json(f"https://commons.wikimedia.org/w/api.php?{params}")
        pages = payload.get("query", {}).get("pages", [])
        refreshed = normalize_wikimedia_page(pages[0]) if pages else None
    elif provider == "smithsonian":
        key = env.get("SMITHSONIAN_API_KEY", "")
        if not key:
            raise VisualSupplyError("下载复核 Smithsonian 授权时缺少 SMITHSONIAN_API_KEY。")
        import urllib.parse

        object_id = str(candidate["source_id"]).split(":", 1)[0]
        params = urllib.parse.urlencode({"api_key": key})
        payload = _request_json(
            f"https://api.si.edu/openaccess/api/v1.0/content/{urllib.parse.quote(object_id)}?{params}"
        )
        rows = normalize_smithsonian_row(payload.get("response", {}))
        refreshed = next(
            (item for item in rows if item["download_url"] == candidate["download_url"]),
            rows[0] if rows else None,
        )
    else:
        refreshed = None
    if not refreshed or refreshed.get("rights_code") != candidate.get("rights_code"):
        raise VisualSupplyError(
            f"素材 {candidate.get('asset_id')} 的授权已变化、来源消失或无法由原 provider 复核。"
        )
    return refreshed


def download_selected_assets(
    selection: dict[str, Any],
    search_config: dict[str, Any],
    env: dict[str, str],
    cache_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    asset_cache = cache_root / "assets"
    asset_cache.mkdir(parents=True, exist_ok=True)
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    selections = selection.get("selections", [])

    def make_entry(
        selected: dict[str, Any],
        original: dict[str, Any],
        local_path: Path,
        provenance_path: Path,
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        generation = original.get("generation") or {}
        return {
            "shot_id": int(selected["shot_id"]),
            "intent_id": selected["intent_id"],
            "narration": selected.get("narration", ""),
            "asset_id": original["asset_id"],
            "provider": original["provider"],
            "title": original["title"],
            "creator": original["creator"],
            "institution": original["institution"],
            "collection_id": original.get("source_id", ""),
            "source_page": original["source_page"],
            "download_url": original.get("download_url", ""),
            "rights_code": original["rights_code"],
            "rights_url": original["rights_url"],
            "retrieved_at": verification.get("verified_at"),
            "reviewed_at": selection.get("reviewed_at"),
            "sha256": verification["sha256"],
            "width": int(verification["width"]),
            "height": int(verification["height"]),
            "mime": verification["mime"],
            "local_path": str(local_path),
            "provenance_ref": str(provenance_path.relative_to(run_dir)),
            "ai_generated": bool(original.get("ai_generated")),
            "score": original.get("score", 0),
            "semantic_status": selected.get(
                "semantic_status", original.get("semantic_status", "unknown")
            ),
            "semantic_score": selected.get(
                "semantic_score", original.get("semantic_score")
            ),
            "semantic_reason": selected.get(
                "semantic_reason",
                (original.get("semantic_review") or {}).get("reason", ""),
            ),
            "semantic_conflicts": selected.get(
                "semantic_conflicts",
                (original.get("semantic_review") or {}).get("conflicts", []),
            ),
            "semantic_override": bool(selected.get("semantic_override")),
            "generation": original.get("generation"),
            "ai_model": generation.get("model", ""),
            "ai_request_id": generation.get("request_id", ""),
            "ai_prompt": generation.get("prompt", ""),
            "ai_size": generation.get("size", ""),
            "ai_quality": generation.get("quality", ""),
            "ai_generated_at": generation.get("generated_at", ""),
        }

    def existing_entry(
        selected: dict[str, Any], original: dict[str, Any]
    ) -> dict[str, Any] | None:
        provenance_path = provenance_dir / f"{original['asset_id']}.json"
        if not provenance_path.is_file():
            return None
        try:
            snapshot = json.loads(provenance_path.read_text(encoding="utf-8"))
            verification = snapshot["download_verification"]
            if snapshot.get("review_candidate", {}).get("asset_id") != original["asset_id"]:
                return None
            digest = str(verification["sha256"])
            matches = list(
                assets_dir.glob(f"shot-{int(selected['shot_id']):03d}-{digest[:12]}.*")
            )
            if len(matches) != 1:
                return None
            data = matches[0].read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                return None
            width, height, _ = verify_image_bytes(
                data,
                str(verification.get("mime") or ""),
                min_long_edge=int(search_config.get("min_long_edge", 1024)),
                min_short_edge=int(search_config.get("min_short_edge", 640)),
            )
            if width != int(verification["width"]) or height != int(verification["height"]):
                return None
            return make_entry(
                selected, original, matches[0], provenance_path, verification
            )
        except (
            AssetSourceError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    partial_path = run_dir / "working" / "assets_manifest.partial.json"
    for item_index, selected in enumerate(selections, 1):
        original = selected["candidate"]
        reused = existing_entry(selected, original)
        if reused:
            entries.append(reused)
            print(
                f"      素材 {item_index}/{len(selections)}：镜头 {selected['shot_id']} "
                f"复用已下载文件（{original['provider']}）"
            )
            continue
        print(
            f"      素材 {item_index}/{len(selections)}：镜头 {selected['shot_id']} "
            f"复核并下载 {original['provider']} · {str(original.get('title', 'Untitled'))[:60]}"
        )
        try:
            candidate = _refetch_candidate(original, env)
        except (AssetSourceError, VisualSupplyError) as exc:
            raise VisualSupplyError(str(exc)) from exc
        if original.get("ai_generated"):
            data = Path(original["local_preview"]).read_bytes()
            mime = "image/jpeg"
            candidate = original
        else:
            data, mime = download_bytes(str(candidate["download_url"]))
        try:
            width, height, extension = verify_image_bytes(
                data,
                mime,
                min_long_edge=int(search_config.get("min_long_edge", 1024)),
                min_short_edge=int(search_config.get("min_short_edge", 640)),
            )
        except AssetSourceError as exc:
            raise VisualSupplyError(str(exc)) from exc
        digest = hashlib.sha256(data).hexdigest()
        cache_path = asset_cache / digest
        if not cache_path.is_file():
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, cache_path)
        local_path = assets_dir / f"shot-{int(selected['shot_id']):03d}-{digest[:12]}.{extension}"
        shutil.copy2(cache_path, local_path)
        provenance_path = provenance_dir / f"{original['asset_id']}.json"
        snapshot = {
            "review_candidate": original,
            "download_verification": {
                "verified_at": datetime.now().astimezone().isoformat(),
                "mime": mime,
                "width": width,
                "height": height,
                "sha256": digest,
            },
        }
        _atomic_json(provenance_path, snapshot)
        verification = snapshot["download_verification"]
        entries.append(
            make_entry(selected, original, local_path, provenance_path, verification)
        )
        _atomic_json(
            partial_path,
            {
                "schema_version": 1,
                "completed": len(entries),
                "total": len(selections),
                "assets": entries,
            },
        )
        print(
            f"      素材 {item_index}/{len(selections)}：完成（{width}x{height}，"
            f"{len(data) / 1024 / 1024:.1f} MB）"
        )
    manifest = {
        "schema_version": 1,
        "human_reviewed": bool(selection.get("reviewed")),
        "created_at": datetime.now().astimezone().isoformat(),
        "assets": entries,
    }
    _atomic_json(run_dir / "assets_manifest.json", manifest)
    return manifest


def validate_asset_manifest(
    manifest: dict[str, Any], search_config: dict[str, Any]
) -> None:
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise VisualSupplyError("素材清单为空，不能复用 asset_download 阶段。")
    for item in assets:
        path = Path(str(item.get("local_path") or ""))
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise VisualSupplyError(f"素材文件无法读取：{path}") from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != str(item.get("sha256") or ""):
            raise VisualSupplyError(f"素材文件哈希已变化：{path.name}")
        try:
            width, height, _ = verify_image_bytes(
                data,
                str(item.get("mime") or ""),
                min_long_edge=int(search_config.get("min_long_edge", 1024)),
                min_short_edge=int(search_config.get("min_short_edge", 640)),
            )
        except AssetSourceError as exc:
            raise VisualSupplyError(f"素材文件无法完整解码：{path.name}；{exc}") from exc
        if width != int(item.get("width") or 0) or height != int(item.get("height") or 0):
            raise VisualSupplyError(f"素材文件尺寸与清单不一致：{path.name}")


def write_license_outputs(manifest: dict[str, Any], run_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    assets = manifest.get("assets", [])
    if not manifest.get("human_reviewed") or not assets:
        raise VisualSupplyError("授权审计要求所有镜头已人工审核并有本地素材。")
    for item in assets:
        allowed = item.get("rights_code") in {"cc0-1.0", "pdm-1.0"}
        if item.get("ai_generated"):
            allowed = item.get("rights_code") == "provider_terms"
        if not allowed or not item.get("source_page") or not item.get("sha256"):
            raise VisualSupplyError(f"素材 {item.get('asset_id')} 未通过关闭式授权审计。")
    csv_path = run_dir / "licenses.csv"
    fields = [
        "shot_id", "asset_id", "title", "creator", "institution", "collection_id",
        "source_page", "download_url", "rights_code", "rights_url", "retrieved_at",
        "reviewed_at", "narration", "sha256", "ai_generated",
        "semantic_status", "semantic_score", "semantic_reason", "semantic_override",
        "ai_model", "ai_request_id", "ai_prompt", "ai_size", "ai_quality", "ai_generated_at",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(assets)
    credits = run_dir / "CREDITS.md"
    credit_lines = ["# 素材来源与授权", ""]
    for item in assets:
        credit_lines.extend(
            [
                f"## 镜头 {item['shot_id']} · {item['title']}", "",
                f"- 作者：{item['creator']}", f"- 机构：{item['institution']}",
                f"- 权利标识：{item['rights_code']}",
                f"- [来源页]({item['source_page']})", f"- SHA-256：`{item['sha256']}`", "",
            ]
        )
    credits.write_text("\n".join(credit_lines), encoding="utf-8")
    disclosure = run_dir / "AI_DISCLOSURE.md"
    ai_assets = [item for item in assets if item.get("ai_generated")]
    disclosure.write_text(
        "# AI 画面说明\n\n"
        + ("本视频包含部分 AI 历史重构画面。\n\n" if ai_assets else "本次成片未使用 AI 生成画面。\n\n")
        + "AI 画面仅作为视觉重构，不构成史料或事实来源。\n",
        encoding="utf-8",
    )
    audit = {
        "asset_rights_ready": True,
        "human_reviewed": True,
        "ai_disclosure_required": bool(ai_assets),
        "provider_counts": dict(Counter(str(item["provider"]) for item in assets)),
        "semantic_status_counts": dict(
            Counter(str(item.get("semantic_status", "unknown")) for item in assets)
        ),
        "semantic_override_count": sum(
            1 for item in assets if item.get("semantic_override")
        ),
    }
    _atomic_json(run_dir / "license_audit.json", audit)
    return csv_path, credits, disclosure, audit


def build_sourced_storyboard(
    title: str,
    duration: float,
    canvas: dict[str, Any],
    shots: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    assets = {int(item["shot_id"]): item for item in manifest.get("assets", [])}
    motion_cycle = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
    storyboard_shots: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, 1):
        asset = assets.get(index)
        if not asset:
            raise VisualSupplyError(f"镜头 {index} 缺少已审核并下载的素材。")
        storyboard_shots.append(
            {
                "id": index,
                **shot,
                "source": asset["local_path"],
                "source_name": Path(asset["local_path"]).name,
                "kind": "image",
                "source_start": 0.0,
                "source_width": asset["width"],
                "source_height": asset["height"],
                "fit": "crop" if asset["height"] >= asset["width"] else "blur",
                "focal_x": 0.5,
                "focal_y": 0.5,
                "motion": motion_cycle[index % len(motion_cycle)],
                "rendered_clip": f"working/clips/shot-{index:03d}.mp4",
                "intent_id": asset["intent_id"],
                "asset_id": asset["asset_id"],
                "visual_origin": "ai" if asset["ai_generated"] else "collection",
                "provenance_ref": asset["provenance_ref"],
                "rights_code": asset["rights_code"],
                "ai_generated": asset["ai_generated"],
                "reviewed": True,
                "asset_score": asset["score"],
                "semantic_status": asset.get("semantic_status", "unknown"),
                "semantic_score": asset.get("semantic_score"),
                "semantic_reason": asset.get("semantic_reason", ""),
                "semantic_override": bool(asset.get("semantic_override")),
            }
        )
    return {
        "schema_version": 2,
        "visual_mode": "sourced",
        "title": title,
        "canvas": canvas,
        "audio_duration": duration,
        "visual_duration": float(shots[-1]["end"]),
        "seed": None,
        "draft_path": None,
        "shots": storyboard_shots,
    }
