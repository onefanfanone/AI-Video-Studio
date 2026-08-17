from __future__ import annotations

import html
import json
import os
import re
import secrets
import shutil
import threading
import urllib.parse
import webbrowser
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

from .script_workbench import (
    ALLOWED_DURATIONS,
    ScriptWorkbenchError,
    _latest_video_task,
    create_draft,
    create_revision_draft,
    discard_draft,
    latest_editing_draft,
    load_draft,
    lock_draft,
    run_ai_revision,
    save_draft,
    save_manual_version,
    script_stats,
    update_from_form,
    voice_preview_files,
    configure_workbench_paths,
)
from .asset_reuse import list_reusable_tasks
from .studio_environment import (
    APP_VERSION,
    check_release_once,
    inspect_environment,
    list_edge_voices,
)
from .studio_profiles import (
    PROFILE_KINDS,
    ProfileError,
    ProfileStore,
    export_profile_bundle,
    import_profile_bundle,
    import_workflow_profile,
    validate_profile,
)
from .studio_providers import (
    ProviderTestError,
    audition_voice_profile,
    profile_fingerprint,
    ValidationStore,
    test_comfyui_profile,
    test_external_image_profile,
    test_llm_profile,
    test_subtitle_profile,
)
from .studio_settings import (
    CODE_ROOT,
    SecretStore,
    SettingsStore,
    StudioSettingsError,
    discover_jianying_draft_root,
    ensure_workspace,
    get_studio_paths,
    import_legacy_env,
)


WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_REQUEST_BYTES = 2_500_000
MAX_FORM_FIELDS = 180


class StudioConsoleError(RuntimeError):
    """A recoverable Studio console error safe to render locally."""


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _icon(name: str) -> str:
    paths = {
        "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10M9 20v-6h6v6"/>',
        "plus": '<path d="M12 5v14M5 12h14"/>',
        "task": '<path d="M7 5h13M7 12h13M7 19h13"/><path d="m3 5 .6.6L5 4m-2 8 .6.6L5 11m-2 8 .6.6L5 18"/>',
        "profile": '<path d="M6 3h9l3 3v15H6z"/><path d="M15 3v4h4M9 11h6M9 15h6"/>',
        "check": '<path d="M12 22s8-3.5 8-10V5l-8-3-8 3v7c0 6.5 8 10 8 10Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/>',
        "folder": '<path d="M3 6h7l2 2h9v11H3z"/>',
        "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.6v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/>',
        "monitor": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
        "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
        "voice": '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/>',
    }
    body = paths.get(name, paths["check"])
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>'


def _brand() -> str:
    return '<span class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m8 5 11 7-11 7z"/><path d="M4 4v16"/></svg></span><div>AI-Video Studio<small>本地控制台</small></div>'


def _nav(active: str) -> str:
    links = [
        ("home", "/", "home", "首页"),
        ("new", "/wizard?step=1", "plus", "新建视频"),
        ("tasks", "/tasks", "task", "任务"),
        ("profiles", "/profiles", "profile", "配置档"),
        ("environment", "/environment", "check", "环境检查"),
        ("settings", "/settings", "folder", "工作区"),
    ]
    rows = "".join(
        f'<a href="{href}" {"aria-current=page" if key == active else ""}>{_icon(icon)}<span>{label}</span></a>'
        for key, href, icon, label in links
    )
    return f'<aside class="sidebar"><div class="brand">{_brand()}</div><nav class="nav">{rows}</nav><div class="sidebar-foot"><span class="status-dot"></span>本机服务 · v{APP_VERSION}</div></aside>'


def _layout(active: str, title: str, body: str, paths: Any) -> str:
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)} · AI-Video Studio</title><link rel="stylesheet" href="/assets/studio.css"></head><body><div class="app-shell">{_nav(active)}<main class="main"><header class="topbar"><strong>{_escape(title)}</strong><div class="location">工作区：{_escape(paths.workspace)}</div></header><div class="content">{body}</div></main></div><script src="/assets/studio.js" defer></script></body></html>'''


def _notice(message: str, kind: str = "") -> str:
    return f'<div class="notice {kind}">{_escape(message)}</div>' if message else ""


def _csrf(token: str) -> str:
    return f'<input type="hidden" name="csrf" value="{_escape(token)}">'


def _setup_page(csrf: str, settings: Mapping[str, Any], env: Mapping[str, Any], message: str = "") -> str:
    checks = "".join(
        f'<div class="check-row"><div><b>{_escape(label)}</b><small>{_escape(row.get("detail"))}</small></div><span class="state {"ok" if row.get("status")=="ok" else "warn"}">{_escape(row.get("status"))}</span></div>'
        for key, label in (("platform", "Windows 10/11 x64"), ("python", "Python"), ("ffmpeg", "FFmpeg"), ("gpu", "GPU / Whisper"), ("jianying", "剪映草稿"), ("comfyui", "ComfyUI"))
        for row in [env[key]]
    )
    legacy = CODE_ROOT / ".env.local"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>首次设置 · AI-Video Studio</title><link rel="stylesheet" href="/assets/studio.css"></head><body><div class="setup-page"><section class="setup-intro"><div><div class="brand">{_brand()}</div><h1>把复杂配置留在这里。</h1><p>完成一次环境向导后，新视频只需选择脚本、画面、声音和输出，不必再编辑 YAML、密钥文件或 ComfyUI JSON。</p></div><small>所有服务只监听 127.0.0.1；凭据使用当前 Windows 用户 DPAPI 加密。</small></section><main class="setup-main"><form class="setup-card" method="post" action="/setup">{_csrf(csrf)}<div class="setup-step">首次环境向导</div><h2>选择工作区并检查工具</h2>{_notice(message, "error" if message else "")}<div class="form-grid"><label class="field"><span>用户工作区</span><input name="workspace" value="{_escape(settings.get('workspace'))}" required><small>项目、输出、缓存、模型和工作流都存放在这里。</small></label><label class="field"><span>剪映草稿目录</span><input name="jianying_draft_root" value="{_escape(settings.get('jianying_draft_root'))}" required><small>已自动按当前 Windows 用户检测。</small></label></div><div class="check-list">{checks}</div><div class="form-grid"><label class="field"><span>DeepSeek / 兼容模型密钥</span><input type="password" name="DEEPSEEK_API_KEY" autocomplete="new-password" placeholder="留空则稍后配置"><small>保存后只显示“已配置”。</small></label><label class="field"><span>OpenAI 密钥（可选）</span><input type="password" name="OPENAI_API_KEY" autocomplete="new-password" placeholder="仅使用本机 ComfyUI 可留空"></label></div>{f'<p><label class="toggle"><input type="checkbox" name="import_legacy_env" value="1" checked>导入现有 .env.local 到 DPAPI</label></p><p><label class="toggle"><input type="checkbox" name="delete_legacy_env" value="1">往返验证后删除旧明文 .env.local</label></p>' if legacy.is_file() else ''}<div class="notice">Whisper、CUDA、ComfyUI 和大模型不会静默下载。环境页会显示体积提示和手动选择入口。</div><button class="button primary" type="submit">保存并进入控制台</button></form></main></div></body></html>'''


def _home_page(csrf: str, paths: Any, settings: Mapping[str, Any], message: str = "") -> str:
    env = inspect_environment(paths)
    latest = latest_editing_draft()
    task = _latest_video_task()
    task_rows = ""
    if latest:
        task_rows += f'<div class="plain-row"><div><b>{_escape(latest.get("title") or "未命名脚本")}</b><span class="meta">脚本草稿 · {_escape(latest.get("mode"))}</span></div><span class="state warn">待锁稿</span><span class="meta">{_escape(latest.get("updated_at"))}</span><button class="button secondary" name="action" value="resume_draft">继续脚本</button></div>'
    if task:
        status = str(task.get("status"))
        label = "等待素材审核" if status == "waiting_for_review" else "构建失败"
        task_rows += f'<div class="plain-row"><div><b>{_escape(task.get("project_id"))}</b><span class="meta">{_escape(task.get("current_stage"))}</span></div><span class="state {"warn" if status=="waiting_for_review" else "fail"}">{label}</span><span class="meta">阶段级安全续跑</span><button class="button secondary" name="action" value="resume_video">继续任务</button></div>'
    if not task_rows:
        task_rows = '<div class="empty">没有未完成任务。可以从一个新选题开始。</div>'
    defaults = settings.get("defaults", {})
    profile_store = ProfileStore(paths)
    def profile_name(kind: str, key: str) -> str:
        try:
            return str(profile_store.get(kind, str(defaults.get(key)))["name"])
        except Exception:
            return "需要重新选择"
    environment_items = "".join(
        f'<div class="environment-item"><span class="environment-icon">{_icon(icon)}</span><div><b>{label}</b><div class="state {"ok" if row.get("status")=="ok" else "warn"}">{_escape(row.get("detail"))}</div></div></div>'
        for label, icon, row in (
            ("本机环境", "monitor", env["platform"]),
            ("ComfyUI", "image", env["comfyui"]),
            ("剪映", "task", env["jianying"]),
            ("语音服务", "voice", {"status": "ok", "detail": "Edge TTS 可配置"}),
        )
    )
    defaults_html = "".join(
        f'<div class="default-item"><span class="meta">{label}</span><b>{_escape(value)}</b></div>'
        for label, value in (
            ("文本模型", profile_name("llm", "llm")),
            ("生图", profile_name("image", "image")),
            ("音色", profile_name("voice", "voice")),
            ("字幕", profile_name("subtitle", "subtitle")),
        )
    )
    update = check_release_once(SettingsStore(paths.appdata_root))
    update_notice = ""
    if update.get("latest_version") and str(update["latest_version"]) != APP_VERSION:
        update_notice = _notice(
            f"发现新版本 {update['latest_version']}。请前往 GitHub Release 下载；程序不会自动覆盖本机文件。"
        )
    body = f'''{_notice(message, "success" if message else "")}{update_notice}<div class="title-row"><div><h1>今天想做什么？</h1><p class="lead">从脚本、画面、配音和字幕开始，一次锁定可复现的视频配置。</p></div><a class="button primary" href="/wizard?step=1">{_icon('plus')}新建视频</a></div><section class="section"><div class="section-head"><h2>环境状态</h2><a href="/environment">查看详情 →</a></div><div class="environment-rail">{environment_items}</div></section><form method="post" action="/home">{_csrf(csrf)}<section class="section"><div class="section-head"><h2>待处理任务</h2><a href="/tasks">查看全部 →</a></div><div class="plain-list">{task_rows}</div></section></form><section class="section"><div class="section-head"><h2>默认配置档</h2><a href="/profiles">管理配置档 →</a></div><div class="defaults">{defaults_html}</div></section>'''
    return _layout("home", "首页", body, paths)


def _tasks_page(csrf: str, paths: Any, message: str = "") -> str:
    tasks = list_reusable_tasks(paths.output_root, paths.project_root)
    cards = "".join(
        f'''<div class="plain-row"><div><b>{_escape(item['title'])}</b>
        <span class="meta">{_escape(item['task_id'])}</span></div>
        <span class="state ok">已完成</span>
        <span class="meta">{item['asset_count']} 张已选画面 · {item['ai_count']} 张 AI</span>
        <button class="button secondary" name="parent_task_id" value="{_escape(item['task_id'])}">修改脚本并复用画面</button></div>'''
        for item in tasks
    ) or '<div class="empty">暂时没有台账完整的 sourced 成片可供派生。</div>'
    body = f'''{_notice(message, "error" if message else "")}<div class="title-row"><div><h1>任务</h1>
    <p class="lead">从成功成片建立独立修订项目；父项目、父成片和版权台账保持只读。</p></div></div>
    <form method="post" action="/tasks/action">{_csrf(csrf)}<section class="section"><div class="section-head"><h2>可复用的历史成片</h2></div><div class="plain-list">{cards}</div></section></form>'''
    return _layout("tasks", "任务", body, paths)


def _wizard_chrome(step: int, body: str, csrf: str, data: Mapping[str, Any], message: str = "") -> str:
    labels = ("脚本", "AI 与画面", "声音与字幕", "输出与汇总")
    steps = "".join(
        f'<div class="wizard-step {"active" if index==step else "done" if index<step else ""}"><span>{index}</span>{label}</div>'
        for index, label in enumerate(labels, 1)
    )
    back = '<a class="button secondary" href="/">退出</a>' if step == 1 else f'<button type="button" class="button secondary" data-submit="back">上一步</button>'
    next_action = "lock" if step == 4 else "next"
    next_label = "锁定脚本并开始制作" if step == 4 else "下一步"
    revision_notice = ""
    if isinstance(data.get("revision"), Mapping):
        revision_notice = _notice(
            f"这是从任务 {data['revision'].get('parent_task_id')} 派生的修订版。锁稿后会先审核旧画面匹配，只为空缺镜头补图。"
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新建视频 · AI-Video Studio</title><link rel="stylesheet" href="/assets/studio.css"></head><body class="wizard-shell"><div class="wizard"><header class="wizard-top"><div class="wizard-brand">AI-Video Studio</div><div class="wizard-steps">{steps}</div><a class="button ghost" href="/">保存并退出</a></header><form id="wizard-form" method="post" action="/wizard/action"><input type="hidden" id="form-action" name="action" value="next">{_csrf(csrf)}<input type="hidden" name="step" value="{step}"><div class="wizard-body"><div class="wizard-panel">{revision_notice}{_notice(message, "error" if message else "")}{body}</div></div><footer class="wizard-footer">{back}<button type="button" class="button primary" data-submit="{next_action}">{next_label}</button></footer></form></div><script src="/assets/studio.js" defer></script></body></html>'''


def _options(catalog: Mapping[str, Mapping[str, Any]], selected: str, *, settings_data: bool = False, previews: Mapping[str, Path] | None = None) -> str:
    rows = []
    for profile_id, profile in catalog.items():
        extras = ""
        if settings_data:
            extras += f' data-settings="{_escape(json.dumps(profile.get("settings", {}), ensure_ascii=False))}"'
        if previews is not None and profile_id in previews:
            extras += f' data-preview="/voice-preview/{_escape(profile_id)}"'
        rows.append(f'<option value="{_escape(profile_id)}" {"selected" if profile_id==selected else ""}{extras}>{_escape(profile.get("name") or profile_id)}</option>')
    return "".join(rows)


def _wizard_step(step: int, data: Mapping[str, Any], csrf: str, paths: Any, message: str = "") -> str:
    store = ProfileStore(paths)
    if step == 1:
        mode = str(data.get("mode", "direct"))
        duration = data.get("duration_seconds")
        stats = script_stats(str(data.get("final_script") or data.get("original_script") or ""), int(duration or 45))
        issues = "".join(f'<li><b>{_escape(item.get("category"))}</b>：{_escape(item.get("message"))}</li>' for item in data.get("analysis", {}).get("issues", [])) or '<li>尚未进行 AI 审核。</li>'
        risks = "".join(f'<li>{_escape(item.get("message"))}</li>' for item in data.get("analysis", {}).get("risks", [])) or '<li>无已记录风险；这不代表完成了史实核查。</li>'
        body = f'''<h1>先把故事说好。</h1><p class="lead">选择你提供脚本、让 AI 审稿，或从一个选题生成初稿。</p><section class="section"><div class="section-body"><div class="mode-switch">{''.join(f'<label><input type="radio" name="mode" value="{value}" {"checked" if mode==value else ""}><b>{label}</b><small>{desc}</small></label>' for value,label,desc in (("direct","直接使用","粘贴完整成稿，不调用 AI"),("review","审核润色","保留原稿，AI 给出问题和建议稿"),("topic","从选题生成","输入选题和资料，AI 生成初稿")))}</div><div class="form-grid three"><label class="field"><span>视频标题</span><input name="title" maxlength="160" value="{_escape(data.get('title'))}" required></label><label class="field"><span>目标时长</span><select name="duration_seconds" required><option value="">请选择</option>{''.join(f'<option value="{value}" {"selected" if duration==value else ""}>{value} 秒</option>' for value in (30,45,60,90))}</select></label><label class="field"><span>脚本模型</span><select name="script_llm_profile">{_options(store.list('llm'), str(data.get('script_llm_profile','deepseek_default')))}</select></label></div></div></section><section class="section" data-mode="topic"><div class="section-body"><div class="form-grid"><label class="field"><span>选题</span><textarea name="topic" placeholder="例如：古罗马人为什么收集尿液？">{_escape(data.get('topic'))}</textarea></label><label class="field"><span>补充资料 / 已知事实</span><textarea name="source_material">{_escape(data.get('source_material'))}</textarea></label><label class="field"><span>必须讲</span><textarea name="must_include">{_escape(data.get('must_include'))}</textarea></label><label class="field"><span>不要讲</span><textarea name="avoid">{_escape(data.get('avoid'))}</textarea></label></div></div></section><div class="editor-split"><section class="section"><div class="section-head"><h2>原稿</h2></div><div class="section-body"><textarea class="script" name="original_script" placeholder="direct / review 模式在这里粘贴完整脚本">{_escape(data.get('original_script'))}</textarea></div></section><section class="section"><div class="section-head"><h2>最终稿</h2><button type="button" class="button secondary" data-submit="ai">生成 / 审核</button></div><div class="section-body"><textarea class="script" name="final_script" placeholder="建议稿和你的人工修改都保留在这里">{_escape(data.get('final_script'))}</textarea></div></section></div><div class="stats-line"><span id="script-stats">{stats['effective_chars']} 个有效字 · 预计 {stats['estimated_seconds']} 秒</span><span>建议 {stats['target_chars'][0]}–{stats['target_chars'][1]} 字</span></div><div class="editor-split"><section class="section"><div class="section-head"><h2>审稿问题</h2></div><div class="section-body"><ul class="issue-list">{issues}</ul></div></section><section class="section"><div class="section-head"><h2>疑似史实风险</h2></div><div class="section-body"><ul class="issue-list">{risks}</ul></div></section></div><details class="advanced"><summary>版本反馈、强调词和读音</summary><div class="section-body"><div class="form-grid"><label class="field"><span>按我的要求再改一版</span><textarea name="feedback" placeholder="更口语化，但保留最后一句"></textarea></label><label class="field"><span>强调词</span><textarea name="emphasis">{_escape('，'.join(data.get('suggestions',{}).get('emphasis',[])))}</textarea></label><label class="field"><span>专名</span><textarea name="proper_nouns">{_escape('，'.join(data.get('suggestions',{}).get('proper_nouns',[])))}</textarea></label><label class="field"><span>读音词典：原词=读法</span><textarea name="pronunciation">{_escape(chr(10).join(f'{k}={v}' for k,v in data.get('suggestions',{}).get('pronunciation',{}).items()))}</textarea></label></div></div></details>'''
    elif step == 2:
        llms = store.list("llm")
        images = store.list("image")
        workflows = store.list("comfyui_workflow")
        strategy = str(data.get("visual_strategy", "museum_and_ai"))
        candidates = int(data.get("candidates_per_shot", 4))
        body = f'''<h1>决定画面从哪里来。</h1><p class="lead">普通模式只显示推荐组合；模型地址、角色分配和调用上限都在高级设置里。</p><section class="section"><div class="section-body"><div class="form-grid"><label class="field"><span>画面策略</span><select name="visual_strategy"><option value="museum_and_ai" {"selected" if strategy=='museum_and_ai' else ""}>馆藏素材＋AI 候选（推荐）</option><option value="ai_only" {"selected" if strategy=='ai_only' else ""}>纯 AI · 跳过馆藏搜索</option><option value="local" {"selected" if strategy=='local' else ""}>本地 raw 素材</option></select><small>纯 AI 仍会先生成镜头意图，再为每镜生成候选。</small></label><label class="field" data-visual="museum_and_ai ai_only"><span>生图配置档</span><select name="image_profile">{_options(images, str(data.get('image_profile','comfyui_default')))}</select></label><label class="field" data-visual="museum_and_ai ai_only"><span>ComfyUI 工作流</span><select name="comfyui_workflow_profile">{_options(workflows, str(data.get('comfyui_workflow_profile','history_image_default')))}</select></label><label class="field" data-visual="museum_and_ai ai_only"><span>每镜 AI 候选</span><select name="candidates_per_shot">{''.join(f'<option value="{n}" {"selected" if candidates==n else ""}>{n} 张</option>' for n in range(1,9))}</select><small>本机 ComfyUI 可动态覆盖所有镜头；外部服务受固定调用上限约束。</small></label></div></div></section><div class="cost-box"><div><span class="meta">目标时长</span><strong>{_escape(data.get('duration_seconds') or '—')} 秒</strong></div><div><span class="meta">预计镜头</span><strong>约 {max(8, round(int(data.get('duration_seconds') or 45)/3.2))} 个</strong></div><div><span class="meta">预计 AI 图</span><strong>约 {0 if strategy=='local' else max(8, round(int(data.get('duration_seconds') or 45)/3.2))*candidates} 张</strong></div></div><details class="advanced"><summary>高级：按角色分配文本模型</summary><div class="section-body"><div class="form-grid three"><label class="field"><span>默认模型</span><select name="llm_profile">{_options(llms, str(data.get('llm_profile','deepseek_default')))}</select></label><label class="field"><span>镜头规划</span><select name="visual_llm_profile">{_options(llms, str(data.get('visual_llm_profile','deepseek_default')))}</select></label><label class="field"><span>候选复核</span><select name="semantic_llm_profile">{_options(llms, str(data.get('semantic_llm_profile','deepseek_default')))}</select></label></div><p class="notice">文本模型测试必须返回短 JSON；未通过验证的配置档不能用于新任务。</p></div></details>'''
    elif step == 3:
        voices = store.list("voice")
        subtitles = store.list("subtitle")
        voice_id = str(data.get("voice_profile", "yunyang_soft"))
        subtitle_id = str(data.get("subtitle_profile", "social_pink"))
        subtitle = subtitles.get(subtitle_id, next(iter(subtitles.values())))
        style = {**subtitle.get("settings", {}), **dict(data.get("subtitle_overrides") or {})}
        previews = voice_preview_files()
        body = f'''<h1>把声音和字幕调成你的风格。</h1><p class="lead">音色、语速和字幕配置都会进入快照；右侧 9:16 画布实时预览常见字幕长度。</p><div class="subtitle-layout"><div class="subtitle-controls"><h2>声音</h2><div class="form-grid"><label class="field"><span>Edge 音色配置档</span><select id="voice-profile" name="voice_profile">{_options(voices, voice_id, previews=previews)}</select></label><div class="field"><span>试听</span><audio id="voice-preview" controls preload="none" src="{f'/voice-preview/{voice_id}' if voice_id in previews else ''}"></audio></div></div><h2>字幕</h2><label class="field"><span>字幕模板</span><select name="subtitle_profile" id="subtitle-profile">{_options(subtitles, subtitle_id, settings_data=True)}</select></label><details class="advanced" open><summary>可视化微调</summary><div class="form-grid three"><label class="field"><span>字体</span><input name="subtitle_font_name" value="{_escape(style.get('font_name','Microsoft YaHei'))}"></label><label class="field"><span>字号</span><input type="number" name="subtitle_font_size" min="36" max="180" value="{_escape(style.get('font_size',118))}"></label><label class="field"><span>底边距</span><input type="number" name="subtitle_margin_bottom" min="0" max="1000" value="{_escape(style.get('margin_bottom',460))}"></label><label class="field"><span>正文色</span><input type="color" name="subtitle_base_color" value="{_escape(style.get('base_color','#FFE3EC'))}"></label><label class="field"><span>强调色</span><input type="color" name="subtitle_highlight_color" value="{_escape(style.get('highlight_color','#FFF3B0'))}"></label><label class="field"><span>描边色</span><input type="color" name="subtitle_outline_color" value="{_escape(style.get('outline_color','#FF5C91'))}"></label><label class="field"><span>描边</span><input type="number" name="subtitle_outline" min="0" max="20" value="{_escape(style.get('outline',8))}"></label><label class="field"><span>阴影</span><input type="number" name="subtitle_shadow" min="0" max="20" value="{_escape(style.get('shadow',5))}"></label><label class="field"><span>每行字数</span><input type="number" name="subtitle_max_chars" min="1" max="24" value="{_escape(style.get('max_chars_per_line',8))}"></label><label class="field"><span>行数</span><select name="subtitle_max_lines"><option value="1" {"selected" if int(style.get('max_lines',1))==1 else ""}>1 行</option><option value="2" {"selected" if int(style.get('max_lines',1))==2 else ""}>2 行</option></select></label><label class="field"><span>淡入 ms</span><input type="number" name="subtitle_fade_in_ms" min="0" max="2000" value="{_escape(style.get('fade_in_ms',150))}"></label><label class="field"><span>淡出 ms</span><input type="number" name="subtitle_fade_out_ms" min="0" max="2000" value="{_escape(style.get('fade_out_ms',150))}"></label></div></details></div><div class="subtitle-preview"><h2>实时预览（9:16）</h2><div class="phone"><div class="preview-subtitle fade" id="preview-subtitle">风起长安梦未央</div></div><div class="preview-tabs"><button type="button" class="active" data-preview="eight">8 字</button><button type="button" data-preview="short">短句</button><button type="button" data-preview="keyword">关键词</button></div><small>MP4 与剪映仍以字体文件和映射检查为准。</small></div></div>'''
    else:
        strategy_labels = {"museum_and_ai":"馆藏＋AI","ai_only":"纯 AI","local":"本地素材"}
        duration = int(data.get("duration_seconds") or 45)
        shots = max(8, round(duration/3.2))
        candidates = int(data.get("candidates_per_shot", 4))
        body = f'''<h1>最后检查一次。</h1><p class="lead">锁稿后会创建独立项目和不可变配置快照，然后才进入配音和 16 阶段视频流水线。</p><section class="section"><div class="section-head"><h2>{_escape(data.get('title') or '未命名视频')}</h2><span>{duration} 秒</span></div><div class="plain-list"><div class="plain-row"><div><b>脚本</b><span class="meta">{_escape(data.get('mode'))}</span></div><span>{script_stats(str(data.get('final_script') or data.get('original_script') or ''),duration)['effective_chars']} 字</span><span class="state ok">已保存</span><a href="/wizard?step=1">修改</a></div><div class="plain-row"><div><b>AI 与画面</b><span class="meta">{strategy_labels.get(str(data.get('visual_strategy')), '—')}</span></div><span>约 {shots} 镜头</span><span>{0 if data.get('visual_strategy')=='local' else shots*candidates} 张 AI 候选</span><a href="/wizard?step=2">修改</a></div><div class="plain-row"><div><b>声音与字幕</b><span class="meta">配置将固化</span></div><span>{_escape(data.get('voice_profile'))}</span><span>{_escape(data.get('subtitle_profile'))}</span><a href="/wizard?step=3">修改</a></div></div></section><section class="section"><div class="section-body"><div class="form-grid"><label class="toggle"><input type="hidden" name="create_jianying_draft_present" value="1"><input type="checkbox" name="create_jianying_draft" value="1" {"checked" if data.get('create_jianying_draft',True) else ""}>生成剪映可编辑草稿</label><label class="toggle"><input type="hidden" name="ai_disclosure_present" value="1"><input type="checkbox" name="ai_disclosure" value="1" {"checked" if data.get('ai_disclosure',True) else ""}>使用 AI 图时添加片尾说明</label></div><div class="notice">预计图像数量按镜头估算。外部付费服务仍受配置档固定上限控制；本机 ComfyUI 不产生 API 费用。</div><div class="notice">publish_ready 仍为 false：最终史实核对、授权抽查和平台发布标识需要人工完成。</div></div></section>'''
    return _wizard_chrome(step, body, csrf, data, message)


def _environment_page(paths: Any) -> str:
    env = inspect_environment(paths)
    labels = {"platform":"系统","python":"Python","ffmpeg":"FFmpeg","ffprobe":"FFprobe","gpu":"GPU / CUDA","jianying":"剪映","moneyprinter":"MoneyPrinterTurbo","whisper":"Whisper 模型","fonts":"字体","comfyui":"ComfyUI","workspace":"工作区"}
    rows = "".join(f'<div class="plain-row"><div><b>{labels[key]}</b><span class="meta">{_escape(row.get("path") or "")}</span></div><span class="state {"ok" if row.get("status")=="ok" else "warn"}">{_escape(row.get("status"))}</span><span>{_escape(row.get("detail"))}</span><span></span></div>' for key,row in env.items() if isinstance(row,dict) and "status" in row)
    links = '<div class="download-links"><a href="https://www.python.org/downloads/release/python-31210/" rel="noreferrer">Python 3.12.10 官方页</a><a href="https://github.com/BtbN/FFmpeg-Builds/releases" rel="noreferrer">FFmpeg 构建发布页</a><a href="https://github.com/comfyanonymous/ComfyUI" rel="noreferrer">ComfyUI 官方仓库</a><a href="https://huggingface.co/Systran/faster-whisper-large-v3" rel="noreferrer">Whisper large-v3 模型页</a></div>'
    body = f'<div class="title-row"><div><h1>环境检查</h1><p class="lead">大模型、CUDA 和 Whisper 不会静默下载；缺失项不会阻止你先完成配置。</p></div></div><section class="section"><div class="plain-list">{rows}</div></section><div class="notice">发行包只携带 Python 安装介质和独立 FFmpeg 组件；Whisper、CUDA、ComfyUI、大模型与剪映字体需要用户自行准备或明确确认下载。</div><section class="section"><div class="section-head"><h2>官方安装来源</h2></div><div class="section-body">{links}<p class="meta">大型组件请先查看体积，再由你明确决定下载或手动选择已有目录。</p></div></section>'
    return _layout("environment", "环境检查", body, paths)


def _profiles_page(csrf: str, paths: Any, query: Mapping[str, list[str]], message: str = "") -> str:
    kind = str(query.get("kind", ["llm"])[0])
    if kind not in PROFILE_KINDS:
        kind = "llm"
    store = ProfileStore(paths)
    validation = ValidationStore(paths)
    labels = {"llm":"文本模型","image":"生图服务","comfyui_workflow":"ComfyUI 工作流","voice":"Edge 音色","subtitle":"字幕样式"}
    kinds = "".join(f'<a class="{"active" if item==kind else ""}" href="/profiles?kind={item}">{labels[item]}</a>' for item in PROFILE_KINDS)
    rows = ""
    for profile in store.list(kind).values():
        tested = validation.status(kind, str(profile["id"]), profile_fingerprint(profile))
        paid = kind == "image" and profile.get("protocol") == "images_compatible"
        test_button = (
            "<button class='button secondary' name='action' value='audition_voice'>试听</button>"
            if kind == "voice"
            else (
                f"<input type='hidden' name='confirm_cost' value='1'><button class='button secondary' name='action' value='test' {'data-confirm=\"这次测试会生成一张图片，可能产生费用。确认继续？\"' if paid else ''}>测试</button>"
                if kind in {"llm", "image", "comfyui_workflow"}
                else (
                    "<button class='button secondary' name='action' value='test'>检查字体</button>"
                    if kind == "subtitle" else ""
                )
            )
        )
        rows += f'<div class="profile-row"><div><b>{_escape(profile.get("name"))}</b><small>{_escape(profile.get("id"))} · {"内置" if profile.get("builtin") else "自定义"} · {"已验证" if tested else "未验证"}</small></div><form method="post" action="/profiles/action">{_csrf(csrf)}<input type="hidden" name="kind" value="{kind}"><input type="hidden" name="profile_id" value="{_escape(profile.get("id"))}">{test_button}</form></div>'
    create_form = ""
    if kind == "llm":
        create_form = '<div class="form-grid three"><label class="field"><span>ID</span><input name="id" placeholder="my_model" required></label><label class="field"><span>名称</span><input name="name" required></label><label class="field"><span>协议</span><select name="protocol"><option value="chat_completions">Chat Completions</option><option value="openai_responses">OpenAI Responses</option></select></label><label class="field"><span>Base URL</span><input name="base_url" value="https://api.deepseek.com" required></label><label class="field"><span>模型 ID</span><input name="model" required></label><label class="field"><span>密钥引用</span><input name="secret_ref" value="DEEPSEEK_API_KEY" required></label><label class="field"><span>超时秒</span><input type="number" name="timeout_seconds" value="120"></label><label class="field"><span>最大输出</span><input type="number" name="max_tokens" value="12000"></label></div>'
    elif kind == "image":
        create_form = '<div class="form-grid three"><label class="field"><span>ID</span><input name="id" required></label><label class="field"><span>名称</span><input name="name" required></label><label class="field"><span>协议</span><select name="protocol"><option value="comfyui_local">本机 ComfyUI</option><option value="images_compatible">Images 兼容服务</option></select></label><label class="field"><span>服务地址 / Base URL</span><input name="base_url" value="http://127.0.0.1:8000"></label><label class="field"><span>模型 ID</span><input name="model"></label><label class="field"><span>固定单次上限</span><input type="number" name="max_images_per_run" value="4" min="1"></label><label class="field"><span>密钥引用</span><input name="secret_ref"></label></div>'
    elif kind == "voice":
        voices = list_edge_voices(paths)
        voice_options = ''.join(f'<option value="{_escape(item["short_name"])}">{_escape(item["locale"])} · {_escape(item["gender"])} · {_escape(item["short_name"])}</option>' for item in voices if item["locale"].startswith("zh-"))
        create_form = f'<div class="form-grid three"><label class="field"><span>ID</span><input name="id" required></label><label class="field"><span>名称</span><input name="name" required></label><label class="field"><span>Edge voice</span><select name="voice">{voice_options}</select></label><label class="field"><span>语速</span><input name="rate" value="+0%"></label><label class="field"><span>音高</span><input name="pitch" value="+0Hz"></label></div>'
    elif kind == "comfyui_workflow":
        create_form = '<div class="form-grid"><label class="field"><span>ID</span><input name="id" required></label><label class="field"><span>名称</span><input name="name" required></label><label class="field"><span>API 格式 JSON</span><input type="file" name="workflow" accept="application/json,.json" required></label><label class="field"><span>提示词标记</span><input name="prompt_marker" value="__AI_VIDEO_PROMPT__"></label></div><details class="advanced"><summary>高级：自定义节点绑定</summary><p class="meta">自动扫描失败时填写 node_id:input_name；标准节点请留空。</p><div class="form-grid three"><label class="field"><span>Prompt</span><input name="binding_prompt"></label><label class="field"><span>Seed</span><input name="binding_seed"></label><label class="field"><span>Width</span><input name="binding_width"></label><label class="field"><span>Height</span><input name="binding_height"></label><label class="field"><span>Output</span><input name="binding_output"></label></div></details>'
    else:
        create_form = '<div class="form-grid three"><label class="field"><span>ID</span><input name="id" required></label><label class="field"><span>名称</span><input name="name" required></label><label class="field"><span>基于预设</span><select name="preset"><option value="social_pink">social_pink</option><option value="history_clean">history_clean</option><option value="history_keyword">history_keyword</option><option value="history_hook">history_hook</option></select></label><label class="field"><span>字号</span><input type="number" name="font_size" value="118"></label><label class="field"><span>正文色</span><input type="color" name="base_color" value="#FFE3EC"></label><label class="field"><span>强调色</span><input type="color" name="highlight_color" value="#FFF3B0"></label><label class="field"><span>每行字数</span><input type="number" name="max_chars_per_line" value="8"></label><label class="field"><span>行数</span><input type="number" name="max_lines" value="1"></label></div>'
    enctype = ' enctype="multipart/form-data"' if kind == "comfyui_workflow" else ""
    body = f'''{_notice(message, "success" if message and "成功" in message else "error" if message else "")}<div class="title-row"><div><h1>配置档</h1><p class="lead">全局默认值与每条视频分离；锁稿时保存完整非密钥快照。</p></div><a class="button secondary" href="/settings">导入 / 导出配置包</a></div><div class="profile-grid"><nav class="profile-kinds">{kinds}</nav><div><div class="profile-list">{rows}</div><section class="section"><div class="section-head"><h2>新建{labels[kind]}配置档</h2></div><form method="post" action="/profiles/action"{enctype}><div class="section-body">{_csrf(csrf)}<input type="hidden" name="kind" value="{kind}">{create_form}<p><button class="button primary" name="action" value="save">保存配置档</button></p></div></form></section></div></div>'''
    return _layout("profiles", "配置档", body, paths)


def _settings_page(csrf: str, paths: Any, settings: Mapping[str, Any], secret_status: Mapping[str, Any], message: str = "") -> str:
    profile_store = ProfileStore(paths)
    defaults = dict(settings.get("defaults") or {})
    secret_rows = "".join(f'<div class="plain-row"><div><b>{name}</b><span class="meta">{"已配置" if name in secret_status else "未配置"}</span></div><span></span><span></span><input type="password" name="secret_{name}" placeholder="留空保持不变"></div>' for name in ("DEEPSEEK_API_KEY","OPENAI_API_KEY","SMITHSONIAN_API_KEY","OPENVERSE_API_TOKEN"))
    body = f'''{_notice(message, "success" if message else "")}<div class="title-row"><div><h1>工作区与设置</h1><p class="lead">本机绝对路径不会进入配置包；密钥不会回显或写入日志。</p></div></div><form method="post" action="/settings/action">{_csrf(csrf)}<section class="section"><div class="section-head"><h2>路径</h2></div><div class="section-body"><div class="form-grid"><label class="field"><span>工作区</span><input name="workspace" value="{_escape(settings.get('workspace'))}"></label><label class="field"><span>剪映草稿目录</span><input name="jianying_draft_root" value="{_escape(settings.get('jianying_draft_root'))}"></label></div></div></section><section class="section"><div class="section-head"><h2>新视频默认配置</h2></div><div class="section-body"><div class="form-grid three"><label class="field"><span>文本模型</span><select name="default_llm">{_options(profile_store.list('llm'),str(defaults.get('llm','deepseek_default')))}</select></label><label class="field"><span>生图服务</span><select name="default_image">{_options(profile_store.list('image'),str(defaults.get('image','comfyui_default')))}</select></label><label class="field"><span>ComfyUI 工作流</span><select name="default_workflow">{_options(profile_store.list('comfyui_workflow'),str(defaults.get('comfyui_workflow','history_image_default')))}</select></label><label class="field"><span>音色</span><select name="default_voice">{_options(profile_store.list('voice'),str(defaults.get('voice','yunyang_soft')))}</select></label><label class="field"><span>字幕</span><select name="default_subtitle">{_options(profile_store.list('subtitle'),str(defaults.get('subtitle','social_pink')))}</select></label><label class="field"><span>画面策略</span><select name="default_visual_strategy"><option value="museum_and_ai" {"selected" if defaults.get('visual_strategy')=='museum_and_ai' else ""}>馆藏＋AI</option><option value="ai_only" {"selected" if defaults.get('visual_strategy')=='ai_only' else ""}>纯 AI</option><option value="local" {"selected" if defaults.get('visual_strategy')=='local' else ""}>本地素材</option></select></label><label class="field"><span>每镜候选</span><select name="default_candidates">{''.join(f'<option value="{n}" {"selected" if int(defaults.get("candidates_per_shot",4))==n else ""}>{n}</option>' for n in range(1,9))}</select></label></div></div></section><section class="section"><div class="section-head"><h2>API 凭据</h2></div><div class="plain-list">{secret_rows}</div></section><section class="section"><div class="section-body"><label class="toggle"><input type="checkbox" name="updates_enabled" value="1" {"checked" if settings.get('updates',{}).get('enabled',True) else ""}>每天最多检查一次 GitHub Release 新版本</label></div></section><button class="button primary" name="action" value="save">保存设置</button></form><section class="section"><div class="section-head"><h2>配置包</h2></div><div class="section-body"><div class="form-grid"><form method="post" action="/settings/action">{_csrf(csrf)}<input type="hidden" name="action" value="export"><button class="button secondary">导出不含密钥和路径的配置包</button></form><form method="post" action="/settings/action"><input type="hidden" name="action" value="import">{_csrf(csrf)}<label class="field"><span>配置包路径</span><input name="bundle_path" placeholder="D:\\profiles.zip"></label><button class="button secondary">导入并重新绑定密钥</button></form></div></div></section>'''
    return _layout("settings", "工作区与设置", body, paths)


def _form_to_profile(kind: str, values: Mapping[str, list[str]]) -> dict[str, Any]:
    value = lambda name, default="": values.get(name, [default])[0]
    base = {"schema_version": 1, "kind": kind, "id": value("id"), "name": value("name")}
    if kind == "llm":
        base.update({"protocol":value("protocol"),"base_url":value("base_url"),"model":value("model"),"json_mode":"response_format","timeout_seconds":int(value("timeout_seconds","120")),"max_tokens":int(value("max_tokens","12000")),"secret_ref":value("secret_ref")})
    elif kind == "image":
        protocol = value("protocol")
        base.update({"protocol":protocol,"server_url":value("base_url") if protocol=="comfyui_local" else None,"base_url":value("base_url") if protocol=="images_compatible" else None,"model":value("model"),"max_images_per_run":int(value("max_images_per_run","4")),"secret_ref":value("secret_ref"),"size":"1024x1536","quality":"medium","output_format":"jpeg"})
    elif kind == "voice":
        base.update({"provider":"edge_tts","voice":value("voice"),"rate":value("rate","+0%"),"pitch":value("pitch","+0Hz")})
    elif kind == "subtitle":
        base.update({"preset":value("preset","social_pink"),"settings":{"font_size":int(value("font_size","118")),"base_color":value("base_color","#FFE3EC"),"highlight_color":value("highlight_color","#FFF3B0"),"max_chars_per_line":int(value("max_chars_per_line","8")),"max_lines":int(value("max_lines","1"))}})
    return base


def run_studio_console(
    *,
    open_browser: bool = True,
    ready_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    configure_workbench_paths()
    csrf = secrets.token_urlsafe(24)
    paths = get_studio_paths(CODE_ROOT)
    settings_store = SettingsStore(paths.appdata_root)
    secret_store = SecretStore(paths.secrets_path)
    current = latest_editing_draft()
    result: dict[str, Any] = {"status":"closed","project":None,"resume_task":None,"options":{}}
    holder: dict[str, ThreadingHTTPServer] = {}
    messages: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def _headers(self, content_type: str, length: int, *, cache: bool = False) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

        def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
            data = body.encode("utf-8")
            self.send_response(status); self._headers(content_type, len(data)); self.end_headers(); self.wfile.write(data)

        def _send_bytes(self, status: int, body: bytes, content_type: str, *, cache: bool = False) -> None:
            self.send_response(status); self._headers(content_type, len(body), cache=cache); self.end_headers(); self.wfile.write(body)

        def _read_raw(self) -> bytes | None:
            try: length = int(self.headers.get("Content-Length", "0"))
            except ValueError: self._send(400,"bad request","text/plain"); return None
            if length < 0 or length > MAX_REQUEST_BYTES: self._send(413,"request too large","text/plain"); return None
            return self.rfile.read(length)

        def _values(self) -> dict[str, list[str]] | None:
            raw = self._read_raw()
            if raw is None: return None
            try: values = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True, max_num_fields=MAX_FORM_FIELDS)
            except (UnicodeDecodeError, ValueError): self._send(400,"invalid form data","text/plain"); return None
            if values.get("csrf",[""])[0] != csrf: self._send(403,"forbidden","text/plain"); return None
            return values

        def _multipart(self) -> tuple[dict[str, list[str]], dict[str, bytes]] | None:
            raw = self._read_raw()
            if raw is None: return None
            header = f"Content-Type: {self.headers.get('Content-Type','')}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            message = BytesParser(policy=email_policy).parsebytes(header + raw)
            values: dict[str,list[str]] = {}; files: dict[str,bytes] = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name: continue
                payload = part.get_payload(decode=True) or b""
                filename = part.get_filename()
                if filename: files[str(name)] = payload
                else: values.setdefault(str(name),[]).append(payload.decode("utf-8"))
            if values.get("csrf",[""])[0] != csrf: self._send(403,"forbidden","text/plain"); return None
            return values, files

        def _shutdown(self) -> None:
            threading.Thread(target=holder["server"].shutdown, daemon=True).start()

        def do_GET(self) -> None:  # noqa: N802
            nonlocal paths, current
            parsed = urllib.parse.urlsplit(self.path); query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/favicon.ico":
                self._send_bytes(204, b"", "image/x-icon", cache=True); return
            if parsed.path.startswith("/assets/"):
                name = Path(parsed.path).name
                if name not in {"studio.css","studio.js"}: self._send(404,"not found","text/plain"); return
                file = WEB_ROOT / name
                self._send_bytes(200,file.read_bytes(),"text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8",cache=True); return
            if parsed.path.startswith("/voice-preview/"):
                profile_id = urllib.parse.unquote(parsed.path.removeprefix("/voice-preview/"))
                if not re.fullmatch(r"[a-z0-9_-]+",profile_id): self._send(404,"not found","text/plain"); return
                preview = voice_preview_files().get(profile_id)
                if preview is None:
                    try: preview = audition_voice_profile(ProfileStore(paths).get("voice",profile_id),paths)
                    except Exception: self._send(404,"not found","text/plain"); return
                self._send_bytes(200,preview.read_bytes(),"audio/mpeg",cache=True); return
            settings = settings_store.load()
            if not settings.get("initialized") and parsed.path != "/setup":
                self._send(200,_setup_page(csrf,settings,inspect_environment(paths),messages.pop("setup",""))); return
            if parsed.path == "/setup": self._send(200,_setup_page(csrf,settings,inspect_environment(paths),messages.pop("setup",""))); return
            if parsed.path == "/": self._send(200,_home_page(csrf,paths,settings,messages.pop("home",""))); return
            if parsed.path == "/wizard":
                if current is None: current = latest_editing_draft() or create_draft()
                step = min(4,max(1,int(query.get("step",["1"])[0])))
                self._send(200,_wizard_step(step,current,csrf,paths,messages.pop("wizard",""))); return
            if parsed.path == "/environment": self._send(200,_environment_page(paths)); return
            if parsed.path == "/profiles": self._send(200,_profiles_page(csrf,paths,query,messages.pop("profiles",""))); return
            if parsed.path == "/settings": self._send(200,_settings_page(csrf,paths,settings,secret_store.status(),messages.pop("settings",""))); return
            if parsed.path == "/tasks": self._send(200,_tasks_page(csrf,paths,messages.pop("tasks",""))); return
            self._send(404,"not found","text/plain")

        def do_POST(self) -> None:  # noqa: N802
            nonlocal paths, current
            parsed = urllib.parse.urlsplit(self.path)
            try:
                if parsed.path == "/setup":
                    values = self._values()
                    if values is None: return
                    value = lambda name: values.get(name,[""])[0]
                    workspace = Path(value("workspace")).expanduser()
                    draft_root = Path(value("jianying_draft_root")).expanduser()
                    if not workspace.is_absolute() or not draft_root.is_absolute(): raise StudioConsoleError("工作区和剪映目录必须是绝对路径。")
                    settings_store.initialize(workspace,jianying_draft_root=draft_root)
                    paths = get_studio_paths(CODE_ROOT); ensure_workspace(paths.workspace); configure_workbench_paths()
                    workflow_target = paths.workflow_root / "history_image_api.json"
                    if not workflow_target.exists() and (CODE_ROOT/"history_image_api.json").is_file(): shutil.copy2(CODE_ROOT/"history_image_api.json",workflow_target)
                    if value("import_legacy_env")=="1": import_legacy_env(CODE_ROOT/".env.local",secret_store)
                    for name in ("DEEPSEEK_API_KEY","OPENAI_API_KEY"):
                        if value(name): secret_store.set(name,value(name))
                    if value("delete_legacy_env")=="1" and (CODE_ROOT/".env.local").is_file():
                        imported = import_legacy_env(CODE_ROOT/".env.local",secret_store)
                        if imported: (CODE_ROOT/".env.local").unlink()
                    messages["home"]="首次设置已完成。"
                    self.send_response(303); self.send_header("Location","/"); self.end_headers(); return
                if parsed.path == "/home":
                    values = self._values()
                    if values is None: return
                    action = values.get("action",[""])[0]
                    if action == "resume_draft": current = latest_editing_draft() or create_draft(); self.send_response(303); self.send_header("Location","/wizard?step=1"); self.end_headers(); return
                    if action == "resume_video": result["status"]="resume_video"; result["resume_task"]=_latest_video_task(); self._send(200,"<meta charset=utf-8><h2>正在安全续跑任务，可关闭页面。</h2>"); self._shutdown(); return
                if parsed.path == "/tasks/action":
                    values = self._values()
                    if values is None: return
                    current = create_revision_draft(values.get("parent_task_id", [""])[0])
                    self.send_response(303); self.send_header("Location", "/wizard?step=1"); self.end_headers(); return
                if parsed.path == "/wizard/action":
                    values = self._values()
                    if values is None: return
                    if current is None: current = create_draft()
                    action = values.get("action",["next"])[0]; step = min(4,max(1,int(values.get("step",["1"])[0])))
                    update_from_form(current,values)
                    if action == "autosave": self._send(204,""); return
                    if action == "ai":
                        if current.get("duration_seconds") not in ALLOWED_DURATIONS or not current.get("title"): raise ScriptWorkbenchError("生成或审核前必须填写标题并选择时长。")
                        if current.get("mode")=="direct": raise ScriptWorkbenchError("direct 模式不会调用 AI。")
                        run_ai_revision(current,values.get("feedback",[""])[0]); messages["wizard"]="AI 版本已生成，请人工检查最终稿。"
                    elif action == "back": step=max(1,step-1)
                    elif action == "next": step=min(4,step+1)
                    elif action == "lock":
                        store = ProfileStore(paths); validation = ValidationStore(paths)
                        required = {
                            "image": {str(current.get("image_profile") or "comfyui_default")},
                        }
                        required_llms = set()
                        if str(current.get("mode") or "direct") != "direct":
                            required_llms.add(str(current.get("script_llm_profile") or "deepseek_default"))
                        if str(current.get("visual_strategy") or "museum_and_ai") == "local":
                            required.pop("image", None)
                        else:
                            required_llms.update({str(current.get("visual_llm_profile") or "deepseek_default"), str(current.get("semantic_llm_profile") or "deepseek_default")})
                        if required_llms:
                            required["llm"] = required_llms
                        missing = []
                        for profile_kind, profile_ids in required.items():
                            for profile_id in profile_ids:
                                profile = store.get(profile_kind, profile_id)
                                if not validation.status(profile_kind, profile_id, profile_fingerprint(profile)):
                                    missing.append(f"{profile_kind}:{profile_id}")
                        image_id = str(current.get("image_profile") or "comfyui_default")
                        if image_id in required.get("image", set()) and store.get("image", image_id).get("protocol") == "comfyui_local":
                            workflow_id = str(current.get("comfyui_workflow_profile") or "history_image_default")
                            workflow = store.get("comfyui_workflow", workflow_id)
                            if not validation.status("comfyui_workflow", workflow_id, profile_fingerprint(workflow)):
                                missing.append(f"comfyui_workflow:{workflow_id}")
                        if missing:
                            raise StudioConsoleError("以下配置档尚未通过短测试，不能开始构建：" + "、".join(sorted(missing)) + "。请先到配置档页面测试。")
                        if bool(current.get("create_jianying_draft", True)):
                            subtitle_id = str(current.get("subtitle_profile") or "social_pink")
                            subtitle = store.get("subtitle", subtitle_id)
                            if not validation.status("subtitle", subtitle_id, profile_fingerprint(subtitle)):
                                raise StudioConsoleError(f"字幕配置 subtitle:{subtitle_id} 尚未通过字体与剪映映射检查。请先到配置档页面检查字体。")
                        project=lock_draft(current,values); result["status"]="locked"; result["project"]=str(project); result["options"]={"visual_mode":"local" if current.get("visual_strategy")=="local" else "sourced","skip_draft":not bool(current.get("create_jianying_draft",True))}; self._send(200,"<meta charset=utf-8><h2>配置已锁定，正在进入视频流水线。</h2>"); self._shutdown(); return
                    self.send_response(303); self.send_header("Location",f"/wizard?step={step}"); self.end_headers(); return
                if parsed.path == "/profiles/action":
                    if self.headers.get("Content-Type","").startswith("multipart/form-data"):
                        parsed_form=self._multipart()
                        if parsed_form is None:return
                        values,files=parsed_form
                    else:
                        values=self._values(); files={}
                        if values is None:return
                    value=lambda name,default="":values.get(name,[default])[0]; kind=value("kind"); action=value("action")
                    store=ProfileStore(paths)
                    if action=="save" and kind=="comfyui_workflow":
                        if "workflow" not in files: raise ProfileError("请选择 API 格式 JSON。")
                        temporary=paths.workflow_root/f".upload-{secrets.token_hex(5)}.json"; temporary.parent.mkdir(parents=True,exist_ok=True); temporary.write_bytes(files["workflow"])
                        bindings = {}
                        for binding_name in ("prompt", "seed", "width", "height", "output"):
                            raw_binding = value(f"binding_{binding_name}").strip()
                            if raw_binding:
                                node_id, separator, input_name = raw_binding.partition(":")
                                if not separator or not node_id or not input_name:
                                    raise ProfileError(f"{binding_name} 绑定必须使用 node_id:input_name。")
                                bindings[binding_name] = [node_id, input_name]
                        try: import_workflow_profile(temporary,value("id"),value("name"),paths=paths,marker=value("prompt_marker","__AI_VIDEO_PROMPT__"),bindings=bindings or None)
                        finally: temporary.unlink(missing_ok=True)
                    elif action=="save": ProfileStore(paths).save(_form_to_profile(kind,values))
                    elif action=="audition_voice": audition_voice_profile(store.get("voice",value("profile_id")),paths)
                    elif action=="test":
                        profile=store.get(kind,value("profile_id"))
                        if kind=="llm": test_llm_profile(profile,paths)
                        elif kind=="image" and profile.get("protocol")=="images_compatible":
                            if value("confirm_cost")!="1": raise ProviderTestError("外部测试可能计费；请在配置档详情中明确确认后测试。")
                            test_external_image_profile(profile,paths)
                        elif kind=="image": test_comfyui_profile(profile,store.get("comfyui_workflow",settings_store.load().get("defaults",{}).get("comfyui_workflow","history_image_default")),paths)
                        elif kind=="comfyui_workflow":
                            image_id=str(settings_store.load().get("defaults",{}).get("image","comfyui_default")); image_profile=store.get("image",image_id)
                            if image_profile.get("protocol")!="comfyui_local": image_profile=store.get("image","comfyui_default")
                            test_comfyui_profile(image_profile,profile,paths)
                        elif kind=="subtitle": test_subtitle_profile(profile,paths)
                    messages["profiles"]="操作成功。"
                    self.send_response(303);self.send_header("Location",f"/profiles?kind={kind}");self.end_headers();return
                if parsed.path == "/settings/action":
                    values=self._values()
                    if values is None:return
                    value=lambda name,default="":values.get(name,[default])[0];action=value("action")
                    if action=="save":
                        settings=settings_store.load();new_workspace=Path(value("workspace",settings["workspace"])).expanduser().resolve()
                        if str(new_workspace)!=str(Path(settings["workspace"]).resolve()): raise StudioConsoleError("移动已有工作区请使用迁移工具，不能仅修改路径。")
                        settings["jianying_draft_root"]=str(Path(value("jianying_draft_root",settings["jianying_draft_root"])).expanduser().resolve());settings["updates"]["enabled"]=value("updates_enabled")=="1";settings_store.save(settings)
                        settings["defaults"].update({"llm":value("default_llm",settings["defaults"]["llm"]),"script_llm":value("default_llm",settings["defaults"]["llm"]),"visual_llm":value("default_llm",settings["defaults"]["llm"]),"semantic_llm":value("default_llm",settings["defaults"]["llm"]),"image":value("default_image",settings["defaults"]["image"]),"comfyui_workflow":value("default_workflow",settings["defaults"]["comfyui_workflow"]),"voice":value("default_voice",settings["defaults"]["voice"]),"subtitle":value("default_subtitle",settings["defaults"]["subtitle"]),"visual_strategy":value("default_visual_strategy",settings["defaults"]["visual_strategy"]),"candidates_per_shot":int(value("default_candidates",str(settings["defaults"]["candidates_per_shot"])))})
                        settings_store.save(settings)
                        for name in ("DEEPSEEK_API_KEY","OPENAI_API_KEY","SMITHSONIAN_API_KEY","OPENVERSE_API_TOKEN"):
                            if value(f"secret_{name}"):secret_store.set(name,value(f"secret_{name}"))
                        messages["settings"]="设置已保存。"
                    elif action=="export":
                        output=paths.export_root/"ai-video-profiles.zip";export_profile_bundle(output,ProfileStore(paths));messages["settings"]=f"配置包已导出：{output}"
                    elif action=="import": import_profile_bundle(Path(value("bundle_path")).expanduser(),ProfileStore(paths));messages["settings"]="配置包已导入；请重新绑定密钥并逐项测试。"
                    self.send_response(303);self.send_header("Location","/settings");self.end_headers();return
                self._send(404,"not found","text/plain")
            except (StudioConsoleError,StudioSettingsError,ProfileError,ProviderTestError,ScriptWorkbenchError,ValueError,OSError) as exc:
                target="wizard" if parsed.path.startswith("/wizard") else "profiles" if parsed.path.startswith("/profiles") else "settings" if parsed.path.startswith("/settings") else "tasks" if parsed.path.startswith("/tasks") else "setup"
                messages[target]=str(exc)
                location=f"/wizard?step={values.get('step',['1'])[0]}" if target=="wizard" and 'values' in locals() and values else f"/{target}" if target!="profiles" else f"/profiles?kind={values.get('kind',['llm'])[0]}"
                self.send_response(303);self.send_header("Location",location);self.end_headers()

        def log_message(self,*_:Any)->None:return

    server=ThreadingHTTPServer(("127.0.0.1",0),Handler);holder["server"]=server
    url=f"http://127.0.0.1:{server.server_address[1]}/";print(f"[AI-Video Studio] 仅本机可访问：{url}")
    if ready_callback:ready_callback(url)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close()
    return result
