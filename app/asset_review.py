from __future__ import annotations

import html
import json
import os
import secrets
import threading
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .asset_sources import download_bytes, perceptual_hash


class ReviewError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def prepare_thumbnails(run_dir: Path, candidates: dict[str, Any]) -> dict[str, Path]:
    root = run_dir / "working" / "asset_thumbnails"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for shot in candidates.get("shots", []):
        for candidate in shot.get("candidates", []):
            asset_id = str(candidate["asset_id"])
            suffix = ".jpg"
            path = root / f"{asset_id}{suffix}"
            if path.is_file() and path.stat().st_size:
                paths[asset_id] = path
                continue
            local_preview = candidate.get("local_preview")
            if local_preview and Path(local_preview).is_file():
                path.write_bytes(Path(local_preview).read_bytes())
                paths[asset_id] = path
                continue
            try:
                data, _ = download_bytes(str(candidate["thumbnail_url"]), limit_bytes=8 * 1024 * 1024)
                path.write_bytes(data)
                paths[asset_id] = path
            except Exception as exc:
                candidate["thumbnail_error"] = str(exc)
    return paths


def prepare_and_deduplicate(
    run_dir: Path, candidates: dict[str, Any], recommendation_threshold: int
) -> dict[str, Any]:
    """Cache review thumbnails, collapse visual duplicates and stabilize recommendations."""
    paths = prepare_thumbnails(run_dir, candidates)
    previous_hash: str | None = None
    for shot in candidates.get("shots", []):
        seen: set[str] = set()
        for candidate in shot.get("candidates", []):
            path = paths.get(str(candidate["asset_id"]))
            if not path:
                continue
            try:
                value = perceptual_hash(path)
            except Exception:
                continue
            candidate["perceptual_hash"] = value
            if value in seen:
                candidate["selectable"] = False
                candidate["rejection_reason"] = "与本镜头更高分候选的缩略图重复"
            else:
                seen.add(value)
        eligible = [
            item
            for item in shot.get("candidates", [])
            if item.get("selectable")
            and int(item.get("score", 0)) >= recommendation_threshold
            and item.get("perceptual_hash") != previous_hash
        ]
        recommendation = eligible[0]["asset_id"] if eligible else None
        shot["recommended_asset_id"] = recommendation
        previous_hash = eligible[0].get("perceptual_hash") if eligible else None
    return candidates


def validate_selection(
    candidates: dict[str, Any],
    selected_ids: dict[int, str],
    semantic_overrides: set[int] | None = None,
) -> dict[str, Any]:
    semantic_overrides = semantic_overrides or set()
    selected: list[dict[str, Any]] = []
    previous: str | None = None
    previous_hash: str | None = None
    for shot in candidates.get("shots", []):
        shot_id = int(shot["shot_id"])
        asset_id = selected_ids.get(shot_id)
        if not asset_id:
            raise ReviewError(f"镜头 {shot_id} 尚未选择素材。")
        lookup = {str(item["asset_id"]): item for item in shot.get("candidates", [])}
        candidate = lookup.get(asset_id)
        if not candidate:
            raise ReviewError(f"镜头 {shot_id} 提交了不在候选清单中的素材。")
        if not candidate.get("selectable"):
            raise ReviewError(f"镜头 {shot_id} 选择的素材尚未通过授权核验。")
        semantic_rejected = (
            candidate.get("semantic_status") == "rejected"
            or bool(candidate.get("semantic_requires_override"))
        )
        if semantic_rejected and shot_id not in semantic_overrides:
            raise ReviewError(
                f"镜头 {shot_id} 选择了语义低相关候选；必须勾选风险确认后才能覆盖。"
            )
        duplicate_override = bool(candidate.get("duplicate_override"))
        if asset_id == previous and not duplicate_override:
            raise ReviewError("相邻镜头不能选择同一素材。")
        candidate_hash = str(candidate.get("perceptual_hash") or "")
        if (
            candidate_hash
            and previous_hash
            and candidate_hash == previous_hash
            and not duplicate_override
        ):
            raise ReviewError("相邻镜头不能选择感知哈希相同的重复素材。")
        previous = asset_id
        previous_hash = candidate_hash or None
        selected.append(
            {
                "shot_id": shot_id,
                "intent_id": shot["intent_id"],
                "narration": shot.get("narration", ""),
                "asset_id": asset_id,
                "semantic_status": candidate.get("semantic_status", "unknown"),
                "semantic_score": candidate.get("semantic_score"),
                "semantic_reason": (candidate.get("semantic_review") or {}).get(
                    "reason", ""
                ),
                "semantic_conflicts": (candidate.get("semantic_review") or {}).get(
                    "conflicts", []
                ),
                "semantic_override": semantic_rejected,
                "duplicate_override": duplicate_override,
                "candidate": candidate,
            }
        )
    return {
        "schema_version": 1,
        "reviewed": True,
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "selections": selected,
    }


def _page(candidates: dict[str, Any], csrf: str, message: str = "") -> str:
    sections: list[str] = []
    ai_only = str(candidates.get("visual_strategy", "")) == "ai_only"
    ai_candidate_count = int(candidates.get("ai_candidates_per_shot", 4))
    semantic_status = str(candidates.get("semantic_review_status", "unknown"))
    warning = ""
    if semantic_status in {"unavailable", "partial", "plan_unavailable"}:
        warning = (
            '<div class="warning">DeepSeek 语义复核未完整完成；本页不会自动勾选素材。'
            "请逐镜头查看图片后人工选择。</div>"
        )
    for shot in candidates.get("shots", []):
        shot_id = int(shot["shot_id"])
        cards: list[str] = []
        rejected_cards: list[str] = []
        recommended = shot.get("recommended_asset_id")
        time_context = shot.get("time_context") or {}
        start_year = time_context.get("start_year")
        end_year = time_context.get("end_year")
        years = "未知年份"
        if start_year is not None or end_year is not None:
            years = f"{start_year if start_year is not None else '?'} 至 {end_year if end_year is not None else '?'}"
        for candidate in shot.get("candidates", []):
            asset_id = str(candidate["asset_id"])
            hard_blocked = not candidate.get("selectable")
            semantic_rejected = candidate.get("semantic_status") == "rejected" or bool(
                candidate.get("semantic_requires_override")
            )
            disabled = "disabled" if hard_blocked or semantic_rejected else ""
            checked = (
                "checked"
                if asset_id == recommended and not hard_blocked and not semantic_rejected
                else ""
            )
            badge = "AI 历史重构" if candidate.get("ai_generated") else html.escape(str(candidate.get("provider", "")))
            rights = html.escape(str(candidate.get("rights_code") or "未核验"))
            semantic = candidate.get("semantic_review") or {}
            compact_metadata = (candidate.get("semantic_metadata") or {}).get(
                "metadata", {}
            )
            fact_parts = []
            for key in (
                "culture",
                "period",
                "objectdate",
                "objectname",
                "classification",
                "medium",
            ):
                value = compact_metadata.get(key)
                if value not in (None, "", []):
                    fact_parts.append(f"{key}: {value}")
            semantic_score = candidate.get("semantic_score")
            semantic_text = (
                "AI 图待人工看图"
                if candidate.get("ai_generated")
                else (
                    "语义复核不可用"
                    if candidate.get("semantic_status") == "unavailable"
                    else f"语义 {semantic_score if semantic_score is not None else '-'}"
                )
            )
            conflicts = "；".join(str(item) for item in semantic.get("conflicts", []))
            card = (
                f"""
                <label class="card {'recommended' if checked else ''} {'blocked' if hard_blocked else ''} {'low' if semantic_rejected else ''}">
                  <input type="radio" name="shot_{shot_id}" value="{html.escape(asset_id)}" {disabled} {checked} {'data-semantic-low='+str(shot_id) if semantic_rejected and not hard_blocked else ''}>
                  <img src="/thumb/{urllib.parse.quote(asset_id)}" alt="candidate">
                  <div class="meta"><b>{html.escape(str(candidate.get('title', 'Untitled')))}</b>
                  <span>{badge} · {rights} · {candidate.get('width', 0)}×{candidate.get('height', 0)}</span>
                  <span>{html.escape(str(candidate.get('institution', '')))} · {html.escape('；'.join(fact_parts))}</span>
                  <span>总分 {candidate.get('score', 0)} · {semantic_text} · {html.escape(str(candidate.get('creator', 'Unknown')))}</span>
                  <span>{html.escape(str(semantic.get('reason', '')))}</span>
                  <span class="conflict">{html.escape(conflicts)}</span>
                  <a href="{html.escape(str(candidate.get('source_page', '#')))}" target="_blank">查看来源页</a></div>
                </label>"""
            )
            if semantic_rejected and not hard_blocked:
                rejected_cards.append(card)
            else:
                cards.append(card)
        low_section = ""
        if rejected_cards:
            low_section = (
                f'<details class="low-box"><summary>显示 {len(rejected_cards)} 个低相关候选（默认不可选）</summary>'
                f'<label class="override"><input type="checkbox" name="semantic_override_{shot_id}" value="1" '
                f'data-enable-low="{shot_id}"> 我已查看冲突理由，仍要承担误选风险</label>'
                f'<div class="grid">{"".join(rejected_cards)}</div></details>'
            )
        sections.append(
            f"""<section><h2>镜头 {shot_id} <small>{float(shot.get('start', 0)):.1f}s</small></h2>
            <p>{html.escape(str(shot.get('narration', '')))}</p>
            <div class="intent"><b>时代：</b>{html.escape(str(time_context.get('label', '未知')))} · {html.escape(str(time_context.get('region', '未知')))} · {html.escape(years)}<br>
            <b>必须出现：</b>{html.escape('；'.join(str(item) for item in shot.get('must_include', [])) or '无')}<br>
            <b>避免出现：</b>{html.escape('；'.join(str(item) for item in shot.get('avoid', [])) or '无')}</div>
            <div class="grid">{''.join(cards)}</div>{low_section}
            <div class="retry"><input name="retry_note_{shot_id}" placeholder="可选：补充新的搜索词或构图要求">
            {'' if ai_only else f'<button class="secondary" name="review_action" value="search:{shot_id}" type="submit">退回重搜</button>'}
            <button class="secondary cost" name="review_action" value="regenerate:{shot_id}" type="submit">再生此镜头（本机 ComfyUI）</button></div></section>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Video 素材审核</title>
    <style>
    :root{{--bg:#101113;--panel:#191b1f;--text:#f5f5f2;--muted:#a6a7aa;--gold:#ffd54a}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
    main{{max-width:1240px;margin:auto;padding:28px}}header{{position:sticky;top:0;z-index:3;background:#101113ee;padding:10px 0 18px;backdrop-filter:blur(14px)}}
    h1{{margin:0 0 5px}}.hint,small,.meta span{{color:var(--muted)}}section{{margin:22px 0 34px}}section>p{{font-size:17px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.card{{display:block;background:var(--panel);border:2px solid transparent;border-radius:12px;overflow:hidden;cursor:pointer}}
    .card:has(input:checked){{border-color:var(--gold)}}.card.blocked{{opacity:.42;cursor:not-allowed}}.card.low{{border-color:#724f35}}.card img{{display:block;width:100%;aspect-ratio:9/12;object-fit:cover;background:#27292e}}
    .card input{{position:absolute;margin:12px;accent-color:var(--gold);width:20px;height:20px}}.meta{{display:flex;flex-direction:column;gap:5px;padding:11px}}a{{color:#8ec5ff}}
    button{{border:0;border-radius:10px;background:var(--gold);color:#171717;font-weight:750;padding:13px 22px;font-size:16px;cursor:pointer}}.error{{color:#ff8f8f}}.warning{{background:#493b1e;border:1px solid #a88332;padding:11px 14px;border-radius:9px;margin-top:10px}}.intent{{background:#16181c;padding:10px 13px;border-left:3px solid #5d6570;margin:9px 0 13px}}.conflict{{color:#ff9b8f}}.low-box{{margin-top:13px;background:#17191d;border:1px solid #5a4233;padding:10px;border-radius:10px}}.low-box summary{{cursor:pointer;color:#ffd6a0}}.override{{display:block;margin:10px 0;color:#ffd6a0}}
    .retry{{display:flex;gap:9px;margin-top:12px;flex-wrap:wrap}}.retry input{{flex:1;min-width:260px;background:#202329;border:1px solid #3b3e45;color:white;border-radius:9px;padding:10px}}
    button.secondary{{background:#30343a;color:#eee;padding:9px 13px;font-size:14px}}button.cost{{color:#ffd6a0}}
    </style></head><body><main><header><h1>素材最终审核</h1><div class="hint">{f'纯 AI 模式：每个镜头必须从 {ai_candidate_count} 张本机候选中明确选择，程序不会自动勾选。' if ai_only else '每个镜头必须明确选择。带灰色的 Openverse 线索或授权不完整素材不可提交。'}</div>
    {warning}<div class="error">{html.escape(message)}</div></header><form method="post" action="/submit"><input type="hidden" name="csrf" value="{csrf}">{''.join(sections)}
    <button type="submit">确认选择并继续构建</button></form></main>
    <script>document.querySelectorAll('[data-enable-low]').forEach(function(box){{box.addEventListener('change',function(){{document.querySelectorAll('[data-semantic-low="'+box.dataset.enableLow+'"]') .forEach(function(r){{r.disabled=!box.checked}})}})}});</script></body></html>"""


def run_review_server(
    run_dir: Path,
    candidates: dict[str, Any],
    *,
    open_browser: bool = True,
) -> str:
    thumbnails = prepare_thumbnails(run_dir, candidates)
    csrf = secrets.token_urlsafe(24)
    result = {"status": "waiting"}
    server_holder: dict[str, ThreadingHTTPServer] = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, _page(candidates, csrf).encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path.startswith("/thumb/"):
                asset_id = urllib.parse.unquote(self.path.removeprefix("/thumb/"))
                path = thumbnails.get(asset_id)
                if path and path.is_file():
                    self._send(200, path.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"", "text/plain")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/submit":
                self._send(404, b"not found", "text/plain")
                return
            length = min(int(self.headers.get("Content-Length", "0")), 128_000)
            values = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
            if values.get("csrf", [""])[0] != csrf:
                self._send(403, b"forbidden", "text/plain")
                return
            review_action = values.get("review_action", [""])[0]
            if review_action:
                try:
                    action, shot_text = review_action.split(":", 1)
                    shot_id = int(shot_text)
                except (ValueError, TypeError):
                    self._send(400, b"invalid action", "text/plain")
                    return
                valid_shots = {int(item["shot_id"]) for item in candidates.get("shots", [])}
                if action not in {"search", "regenerate"} or shot_id not in valid_shots:
                    self._send(400, b"invalid action", "text/plain")
                    return
                _atomic_json(
                    run_dir / "asset_review_request.json",
                    {
                        "action": action,
                        "shot_id": shot_id,
                        "note": values.get(f"retry_note_{shot_id}", [""])[0].strip(),
                        "requested_at": datetime.now().astimezone().isoformat(),
                    },
                )
                result["status"] = "retry"
                self._send(
                    200,
                    "<meta charset=utf-8><h2>请求已收到，候选更新后审核页会重新打开。</h2>".encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                threading.Thread(target=server_holder["server"].shutdown, daemon=True).start()
                return
            selected = {
                int(key.removeprefix("shot_")): items[0]
                for key, items in values.items()
                if key.startswith("shot_") and items
            }
            semantic_overrides = {
                int(key.removeprefix("semantic_override_"))
                for key, items in values.items()
                if key.startswith("semantic_override_") and items and items[0] == "1"
            }
            try:
                payload = validate_selection(
                    candidates, selected, semantic_overrides=semantic_overrides
                )
            except ReviewError as exc:
                self._send(400, _page(candidates, csrf, str(exc)).encode("utf-8"), "text/html; charset=utf-8")
                return
            _atomic_json(run_dir / "asset_selection.json", payload)
            result["status"] = "submitted"
            body = "<meta charset=utf-8><h2>审核已提交，可以关闭此页面。</h2>".encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            threading.Thread(target=server_holder["server"].shutdown, daemon=True).start()

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_holder["server"] = server
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[素材审核] 仅本机可访问：{url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return str(result["status"])
