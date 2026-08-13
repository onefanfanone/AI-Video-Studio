from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image


class ComfyUIError(RuntimeError):
    """A resumable error from the local ComfyUI API."""


def validate_server_url(config: dict[str, Any]) -> str:
    server_url = str(config.get("server_url", "http://127.0.0.1:8000")).rstrip("/")
    parsed = urllib.parse.urlparse(server_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ComfyUIError("ComfyUI server_url 的端口无效。") from exc
    if parsed.scheme not in {"http", "https"} or not port:
        raise ComfyUIError("ComfyUI server_url 必须包含 http(s) 协议和明确端口。")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ComfyUIError("ComfyUI provider 只允许连接本机 127.0.0.1/localhost/::1。")
    return server_url


def resolve_workflow_path(config: dict[str, Any], project_dir: Path) -> Path:
    raw = str(config.get("workflow_file") or "").strip()
    if not raw:
        raise ComfyUIError("ComfyUI 生图缺少 workflow_file 配置。")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ComfyUIError(f"找不到 ComfyUI API 工作流：{path}")
    return path


def load_api_workflow(config: dict[str, Any], project_dir: Path) -> tuple[dict[str, Any], Path]:
    path = resolve_workflow_path(config, project_dir)
    try:
        workflow = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComfyUIError(f"无法读取 ComfyUI API 工作流：{path.name}（{exc}）") from exc
    if not isinstance(workflow, dict) or not workflow:
        raise ComfyUIError("ComfyUI 工作流不是有效的 API 格式 JSON 对象。")
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            raise ComfyUIError(
                f"ComfyUI 工作流节点 {node_id} 缺少 class_type；请使用“导出工作流（API格式）”。"
            )
        if not isinstance(node.get("inputs"), dict):
            raise ComfyUIError(f"ComfyUI 工作流节点 {node_id} 缺少 inputs。")
    return workflow, path


def workflow_fingerprint(config: dict[str, Any], project_dir: Path) -> str:
    _, path = load_api_workflow(config, project_dir)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(
    url: str,
    *,
    data: bytes | None = None,
    method: str = "GET",
    timeout: float = 30,
) -> tuple[bytes, Any]:
    headers = {"Accept": "application/json", "User-Agent": "AI-Video/3.1"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ComfyUIError(f"ComfyUI API 返回 HTTP {exc.code}：{body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ComfyUIError(
            "无法连接本机 ComfyUI。请先启动 ComfyUI，并确认地址为 "
            f"{url.split('/prompt', 1)[0].split('/object_info', 1)[0]}（{type(exc).__name__}）。"
        ) from exc


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
    raw, _ = _request(url, data=data, method=method, timeout=timeout)
    try:
        result = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ComfyUIError("ComfyUI API 未返回有效 JSON。") from exc
    if not isinstance(result, dict):
        raise ComfyUIError("ComfyUI API 返回结构不是 JSON 对象。")
    return result


def _replace_marker(value: Any, marker: str, prompt: str) -> tuple[Any, int]:
    if isinstance(value, str) and marker in value:
        return value.replace(marker, prompt), 1
    if isinstance(value, list):
        result: list[Any] = []
        count = 0
        for item in value:
            replaced, found = _replace_marker(item, marker, prompt)
            result.append(replaced)
            count += found
        return result, count
    if isinstance(value, dict):
        result_dict: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            replaced, found = _replace_marker(item, marker, prompt)
            result_dict[key] = replaced
            count += found
        return result_dict, count
    return value, 0


def prepare_workflow(
    workflow: dict[str, Any],
    prompt: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    prepared = json.loads(json.dumps(workflow, ensure_ascii=False))
    marker = str(config.get("prompt_marker", "__AI_VIDEO_PROMPT__"))
    bindings = dict(config.get("bindings") or {})
    prepared, replacements = _replace_marker(prepared, marker, prompt)
    if replacements != 1:
        prompt_binding = bindings.get("prompt")
        if (
            replacements == 0
            and isinstance(prompt_binding, list)
            and len(prompt_binding) == 2
            and str(prompt_binding[0]) in prepared
            and str(prompt_binding[1]) in prepared[str(prompt_binding[0])]["inputs"]
        ):
            prepared[str(prompt_binding[0])]["inputs"][str(prompt_binding[1])] = prompt
        else:
            raise ComfyUIError(
                f"ComfyUI 工作流应恰好包含一个提示词标记 {marker}，实际找到 {replacements} 个；"
                "也可以在高级配置中指定 prompt 节点和 input。"
            )

    width = int(config.get("width", 768))
    height = int(config.get("height", 1344))
    sampler_count = 0
    size_count = 0
    save_count = 0
    prefix = f"AI-Video-History/{uuid.uuid4().hex}"
    for node in prepared.values():
        class_type = str(node["class_type"])
        inputs = node["inputs"]
        if class_type in {"KSampler", "KSamplerAdvanced"}:
            if "seed" in inputs:
                inputs["seed"] = int(seed)
                sampler_count += 1
            elif "noise_seed" in inputs:
                inputs["noise_seed"] = int(seed)
                sampler_count += 1
        if "width" in inputs and "height" in inputs and class_type.startswith("Empty"):
            inputs["width"] = width
            inputs["height"] = height
            size_count += 1
        if class_type == "SaveImage":
            inputs["filename_prefix"] = prefix
            save_count += 1
    def apply_binding(name: str, value: Any) -> bool:
        binding = bindings.get(name)
        if not isinstance(binding, list) or len(binding) != 2:
            return False
        node_id, input_name = map(str, binding)
        if node_id not in prepared or input_name not in prepared[node_id]["inputs"]:
            raise ComfyUIError(f"ComfyUI {name} 绑定不存在：{node_id}:{input_name}")
        prepared[node_id]["inputs"][input_name] = value
        return True
    if sampler_count < 1 and apply_binding("seed", int(seed)):
        sampler_count = 1
    if size_count < 1:
        width_bound = apply_binding("width", width)
        height_bound = apply_binding("height", height)
        if width_bound and height_bound:
            size_count = 1
    if save_count < 1:
        output_binding = bindings.get("output")
        if isinstance(output_binding, list) and len(output_binding) == 2:
            node_id = str(output_binding[0])
            if node_id in prepared:
                save_count = 1
    if sampler_count < 1:
        raise ComfyUIError("ComfyUI 工作流中没有可设置确定性 seed 的采样器。")
    if size_count < 1:
        raise ComfyUIError("ComfyUI 工作流中没有可设置 width/height 的空图节点。")
    if save_count < 1:
        raise ComfyUIError("ComfyUI 工作流中没有 SaveImage 输出节点。")
    return prepared


def _models_and_sampler(workflow: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    models: set[str] = set()
    sampler: dict[str, Any] = {}
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if key in {"ckpt_name", "unet_name", "clip_name", "vae_name", "model_name"}:
                if isinstance(value, str):
                    models.add(value)
        if str(node.get("class_type")) in {"KSampler", "KSamplerAdvanced"}:
            for key in ("steps", "cfg", "sampler_name", "scheduler", "denoise"):
                if key in inputs:
                    sampler[key] = inputs[key]
    return sorted(models), sampler


def _history_image(record: dict[str, Any]) -> dict[str, str] | None:
    outputs = record.get("outputs") or {}
    for output in outputs.values():
        for image in output.get("images") or []:
            if image.get("filename"):
                return {
                    "filename": str(image["filename"]),
                    "subfolder": str(image.get("subfolder") or ""),
                    "type": str(image.get("type") or "output"),
                }
    return None


def generate_image(
    prompt: str,
    config: dict[str, Any],
    project_dir: Path,
    *,
    seed: int,
) -> tuple[bytes, dict[str, Any]]:
    workflow, workflow_path = load_api_workflow(config, project_dir)
    server_url = validate_server_url(config)
    object_info = _request_json(
        f"{server_url}/object_info",
        timeout=float(config.get("connect_timeout_seconds", 20)),
    )
    missing = sorted(
        {
            str(node["class_type"])
            for node in workflow.values()
            if str(node["class_type"]) not in object_info
        }
    )
    if missing:
        raise ComfyUIError("本机 ComfyUI 缺少工作流节点：" + ", ".join(missing))

    prepared = prepare_workflow(workflow, prompt, config, seed)
    queued = _request_json(
        f"{server_url}/prompt",
        payload={"prompt": prepared, "client_id": uuid.uuid4().hex},
        timeout=float(config.get("connect_timeout_seconds", 20)),
    )
    prompt_id = str(queued.get("prompt_id") or "")
    if not prompt_id:
        error = queued.get("error") or queued.get("node_errors") or "未返回 prompt_id"
        raise ComfyUIError(f"ComfyUI 拒绝了工作流：{error}")

    deadline = time.monotonic() + float(config.get("timeout_seconds", 900))
    image_ref: dict[str, str] | None = None
    while time.monotonic() < deadline:
        history = _request_json(f"{server_url}/history/{prompt_id}", timeout=20)
        record = history.get(prompt_id)
        if isinstance(record, dict):
            image_ref = _history_image(record)
            if image_ref:
                break
            status = record.get("status") or {}
            if status.get("status_str") == "error":
                messages = status.get("messages") or []
                raise ComfyUIError(f"ComfyUI 执行工作流失败：{messages[-1] if messages else 'unknown'}")
            if status.get("completed") is True:
                raise ComfyUIError("ComfyUI 工作流已结束，但没有生成图片。")
        time.sleep(float(config.get("poll_interval_seconds", 1.0)))
    if not image_ref:
        raise ComfyUIError(
            f"等待 ComfyUI 生图超过 {int(config.get('timeout_seconds', 900))} 秒；任务可以续跑。"
        )

    query = urllib.parse.urlencode(image_ref)
    raw_image, _ = _request(f"{server_url}/view?{query}", timeout=60)
    try:
        with Image.open(io.BytesIO(raw_image)) as image:
            width, height = image.size
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=95, optimize=True)
            image_bytes = output.getvalue()
    except (OSError, ValueError) as exc:
        raise ComfyUIError("ComfyUI 返回的结果不是可读取的图片。") from exc

    models, sampler = _models_and_sampler(prepared)
    metadata = {
        "provider": "comfyui_local",
        "model": ", ".join(models) or "ComfyUI workflow",
        "models": models,
        "request_id": prompt_id,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "quality": "local_workflow",
        "output_format": "jpeg",
        "generated_at": int(time.time()),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "seed": int(seed),
        "sampler": sampler,
        "workflow_file": workflow_path.name,
        "workflow_sha256": hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
        "comfyui_output": image_ref,
    }
    return image_bytes, metadata
