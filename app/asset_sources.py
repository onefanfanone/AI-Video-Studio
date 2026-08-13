from __future__ import annotations

import concurrent.futures
import hashlib
import html
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageFile


class AssetSourceError(RuntimeError):
    pass


RIGHTS_LINKS = {
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "pdm-1.0": "https://creativecommons.org/publicdomain/mark/1.0/",
}
PROVIDER_TRUST = {"met": 24, "smithsonian": 24, "wikimedia": 20, "openverse": 0}


def normalize_http_url(value: str) -> str:
    """Percent-encode provider URLs without double-encoding existing escapes."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise AssetSourceError("素材 URL 包含不允许的控制字符。")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return raw
    if any(character.isspace() for character in parsed.netloc):
        raise AssetSourceError("素材 URL 的主机名包含空白字符。")
    path = urllib.parse.quote(parsed.path, safe="/:@!$&'()*+,;=-._~%")
    query = urllib.parse.quote(parsed.query, safe="=&?/:;+,%@[]")
    fragment = urllib.parse.quote(parsed.fragment, safe="=&?/:;+,%@[]")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, path, query, fragment)
    )


def normalize_rights(value: str, url: str = "") -> str | None:
    text = " ".join((value, url)).strip().lower()
    if "creativecommons.org/publicdomain/zero/1.0" in text or re.search(
        r"\bcc0(?:\s*1\.0)?\b", text
    ):
        return "cc0-1.0"
    if "creativecommons.org/publicdomain/mark/1.0" in text or any(
        marker in text for marker in ("public domain mark", "public domain", "pdm 1.0")
    ):
        return "pdm-1.0"
    return None


def _plain(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return html.unescape(re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    url = normalize_http_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-Video/3.0", **(headers or {})},
    )
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if attempt < 2:
            time.sleep(attempt + 1)
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = urllib.parse.urlencode(
        [(key, "***" if key.lower() in {"api_key", "token", "access_token"} else value) for key, value in query]
    )
    safe_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, safe_query, parsed.fragment)
    )
    raise AssetSourceError(f"请求失败（已重试两次）：{safe_url}；{last_error}")


def _https(value: str) -> bool:
    return value.lower().startswith("https://")


def _candidate(
    *,
    provider: str,
    source_id: str,
    title: str,
    creator: str,
    institution: str,
    source_page: str,
    download_url: str,
    thumbnail_url: str,
    rights_code: str | None,
    rights_url: str,
    width: int,
    height: int,
    raw: dict[str, Any],
    selectable: bool = True,
) -> dict[str, Any]:
    source_page = normalize_http_url(source_page)
    download_url = normalize_http_url(download_url)
    thumbnail_url = normalize_http_url(thumbnail_url or download_url)
    stable = hashlib.sha256(
        f"{provider}\0{source_id}\0{download_url}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "asset_id": f"{provider}-{stable}",
        "provider": provider,
        "source_id": str(source_id),
        "title": title or "Untitled",
        "creator": creator or "Unknown",
        "institution": institution,
        "source_page": source_page,
        "download_url": download_url,
        "thumbnail_url": thumbnail_url,
        "rights_code": rights_code,
        "rights_url": rights_url or (RIGHTS_LINKS.get(rights_code or "", "")),
        "width": int(width or 0),
        "height": int(height or 0),
        "mime": "",
        "selectable": bool(selectable and rights_code and source_page and download_url),
        "requires_reverification": provider == "openverse",
        "ai_generated": False,
        "score": 0,
        "score_detail": {},
        "raw_metadata": raw,
    }


def normalize_met_object(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("isPublicDomain") is not True or not item.get("primaryImage"):
        return None
    source_page = str(item.get("objectURL") or "")
    download = str(item.get("primaryImage") or "")
    if not (_https(source_page) and _https(download)):
        return None
    return _candidate(
        provider="met",
        source_id=str(item.get("objectID", "")),
        title=str(item.get("title") or "Untitled"),
        creator=str(item.get("artistDisplayName") or "Unknown"),
        institution="The Metropolitan Museum of Art",
        source_page=source_page,
        download_url=download,
        thumbnail_url=str(item.get("primaryImageSmall") or download),
        rights_code="pdm-1.0",
        rights_url=RIGHTS_LINKS["pdm-1.0"],
        width=0,
        height=0,
        raw=item,
    )


def normalize_smithsonian_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    content = row.get("content", {})
    media = (
        content.get("descriptiveNonRepeating", {})
        .get("online_media", {})
        .get("media", [])
    )
    title = _plain(content.get("descriptiveNonRepeating", {}).get("title")) or str(
        row.get("title") or "Untitled"
    )
    source_page = str(
        content.get("descriptiveNonRepeating", {}).get("record_link")
        or row.get("url")
        or ""
    )
    creator = "Unknown"
    for name in content.get("freetext", {}).get("name", []) or []:
        if "maker" in str(name.get("label", "")).lower() or creator == "Unknown":
            creator = _plain(name.get("content")) or creator
    results: list[dict[str, Any]] = []
    for index, image in enumerate(media):
        usage = image.get("usage", {})
        rights = normalize_rights(_plain(usage.get("access")), _plain(usage.get("text")))
        download = str(image.get("content") or image.get("resources", [{}])[-1].get("url") or "")
        thumbnail = str(image.get("thumbnail") or download)
        if rights != "cc0-1.0" or not (_https(source_page) and _https(download)):
            continue
        results.append(
            _candidate(
                provider="smithsonian",
                source_id=f"{row.get('id', '')}:{index}",
                title=title,
                creator=creator,
                institution=str(row.get("unitCode") or "Smithsonian Institution"),
                source_page=source_page,
                download_url=download,
                thumbnail_url=thumbnail,
                rights_code=rights,
                rights_url=RIGHTS_LINKS[rights],
                width=int(image.get("width") or 0),
                height=int(image.get("height") or 0),
                raw={"row": row, "media_index": index},
            )
        )
    return results


def normalize_wikimedia_page(page: dict[str, Any]) -> dict[str, Any] | None:
    info = (page.get("imageinfo") or [{}])[0]
    metadata = info.get("extmetadata") or {}
    rights_text = " ".join(
        _plain(metadata.get(name))
        for name in ("LicenseShortName", "License", "UsageTerms", "Copyrighted")
    )
    rights_url = _plain(metadata.get("LicenseUrl"))
    rights = normalize_rights(rights_text, rights_url)
    source_page = str(info.get("descriptionurl") or info.get("descriptionshorturl") or "")
    download = str(info.get("url") or "")
    creator = _plain(metadata.get("Artist"))
    if not (rights and creator and _https(source_page) and _https(download)):
        return None
    return _candidate(
        provider="wikimedia",
        source_id=str(page.get("pageid") or page.get("title") or ""),
        title=_plain(metadata.get("ObjectName")) or str(page.get("title") or "Untitled"),
        creator=creator,
        institution=_plain(metadata.get("Credit")) or "Wikimedia Commons",
        source_page=source_page,
        download_url=download,
        thumbnail_url=str(info.get("thumburl") or download),
        rights_code=rights,
        rights_url=rights_url or RIGHTS_LINKS[rights],
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        raw=page,
    )


def normalize_openverse_result(item: dict[str, Any]) -> dict[str, Any] | None:
    rights = normalize_rights(str(item.get("license") or ""), str(item.get("license_url") or ""))
    source_page = str(item.get("foreign_landing_url") or "")
    download = str(item.get("url") or "")
    if not (rights and _https(source_page) and _https(download)):
        return None
    candidate = _candidate(
        provider="openverse",
        source_id=str(item.get("id") or ""),
        title=str(item.get("title") or "Untitled"),
        creator=str(item.get("creator") or "Unknown"),
        institution=str(item.get("source") or "Openverse discovery"),
        source_page=source_page,
        download_url=download,
        thumbnail_url=str(item.get("thumbnail") or download),
        rights_code=rights,
        rights_url=str(item.get("license_url") or RIGHTS_LINKS[rights]),
        width=int(item.get("width") or 0),
        height=int(item.get("height") or 0),
        raw=item,
        selectable=False,
    )
    candidate["rejection_reason"] = "Openverse 仅用于发现，尚未由上游馆藏 provider 复核"
    return candidate


def search_met(query: str, limit: int, _: dict[str, str]) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"hasImages": "true", "q": query})
    result = _request_json(f"https://collectionapi.metmuseum.org/public/collection/v1/search?{params}")
    ids = (result.get("objectIDs") or [])[: max(limit * 2, 8)]
    candidates: list[dict[str, Any]] = []
    for object_id in ids:
        item = _request_json(
            f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        )
        candidate = normalize_met_object(item)
        if candidate:
            candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def search_smithsonian(query: str, limit: int, env: dict[str, str]) -> list[dict[str, Any]]:
    key = env.get("SMITHSONIAN_API_KEY", "")
    if not key:
        raise AssetSourceError("缺少 SMITHSONIAN_API_KEY，已跳过 Smithsonian")
    params = urllib.parse.urlencode({"q": query, "api_key": key, "rows": max(limit, 10)})
    result = _request_json(f"https://api.si.edu/openaccess/api/v1.0/search?{params}")
    candidates: list[dict[str, Any]] = []
    for row in result.get("response", {}).get("rows", []):
        candidates.extend(normalize_smithsonian_row(row))
        if len(candidates) >= limit:
            break
    return candidates[:limit]


def search_wikimedia(query: str, limit: int, _: dict[str, str]) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6, "gsrlimit": max(limit * 2, 10),
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 480, "origin": "*",
        }
    )
    result = _request_json(f"https://commons.wikimedia.org/w/api.php?{params}")
    candidates = [
        candidate
        for page in result.get("query", {}).get("pages", [])
        if (candidate := normalize_wikimedia_page(page)) is not None
    ]
    return candidates[:limit]


def search_openverse(query: str, limit: int, env: dict[str, str]) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"q": query, "license": "cc0,pdm", "page_size": min(max(limit, 1), 20)}
    )
    headers = {}
    if env.get("OPENVERSE_API_TOKEN"):
        headers["Authorization"] = f"Bearer {env['OPENVERSE_API_TOKEN']}"
    result = _request_json(f"https://api.openverse.org/v1/images/?{params}", headers=headers)
    return [
        candidate
        for item in result.get("results", [])
        if (candidate := normalize_openverse_result(item)) is not None
    ][:limit]


SEARCHERS: dict[str, Callable[[str, int, dict[str, str]], list[dict[str, Any]]]] = {
    "met": search_met,
    "smithsonian": search_smithsonian,
    "wikimedia": search_wikimedia,
    "openverse": search_openverse,
}


def _keywords(intent: dict[str, Any]) -> set[str]:
    terms = [
        *intent.get("search_terms_en", []), *intent.get("search_terms_zh", []),
        *intent.get("objects", []), intent.get("era", ""), intent.get("location", ""),
    ]
    return {
        token.lower()
        for term in terms
        for token in re.findall(r"[\w\u4e00-\u9fff]{2,}", str(term))
    }


def historical_consistency_reason(
    candidate: dict[str, Any], intent: dict[str, Any], mode: str = "off"
) -> str | None:
    """Deprecated compatibility hook; semantic review owns topic-specific checks."""
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode not in {"off", "strict"}:
        raise AssetSourceError("historical_consistency 必须是 off 或 strict。")
    return None


def score_candidate(
    candidate: dict[str, Any],
    intent: dict[str, Any],
    historical_consistency: str = "off",
) -> dict[str, Any]:
    historical_consistency_reason(candidate, intent, historical_consistency)
    haystack = " ".join(
        str(candidate.get(key, ""))
        for key in ("title", "creator", "institution", "source_page")
    ).lower()
    keywords = _keywords(intent)
    matches = sum(1 for word in keywords if word in haystack)
    relevance = min(42, (8 + matches * 9) if matches else 0)
    width = int(candidate.get("width") or 0)
    height = int(candidate.get("height") or 0)
    resolution = 12 if max(width, height) >= 1600 and min(width, height) >= 800 else 6
    if not width or not height:
        resolution = 0
    ratio = width / height if height else 1.0
    crop = max(0, round(12 - abs(math.log(max(ratio, 0.05) / (9 / 16))) * 4))
    rights = 10 if candidate.get("rights_code") in RIGHTS_LINKS else 0
    trust = PROVIDER_TRUST.get(str(candidate.get("provider")), 0)
    total = relevance + resolution + crop + rights + trust
    if not candidate.get("selectable"):
        total = 0
    candidate["score"] = min(100, total)
    candidate["score_detail"] = {
        "text_relevance": relevance,
        "institution_trust": trust,
        "resolution": resolution,
        "vertical_crop": crop,
        "rights": rights,
        "historical_consistency": "deprecated_dynamic_review_pending",
    }
    return candidate


def probe_remote_dimensions(url: str) -> tuple[int, int, str]:
    """Read only enough of an HTTPS image stream for Pillow to parse its header."""
    if not _https(url):
        return 0, 0, ""
    try:
        url = normalize_http_url(url)
    except AssetSourceError:
        return 0, 0, ""
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Video/3.0"})
    parser = ImageFile.Parser()
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            mime = str(response.headers.get_content_type())
            remaining = 768 * 1024
            while remaining > 0 and parser.image is None:
                chunk = response.read(min(16 * 1024, remaining))
                if not chunk:
                    break
                parser.feed(chunk)
                remaining -= len(chunk)
            if parser.image is None:
                return 0, 0, mime
            width, height = parser.image.size
            return int(width), int(height), mime
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, 0, ""


def _valid_dimensions(candidate: dict[str, Any], search: dict[str, Any]) -> bool:
    width = int(candidate.get("width") or 0)
    height = int(candidate.get("height") or 0)
    return max(width, height) >= int(search.get("min_long_edge", 1024)) and min(
        width, height
    ) >= int(search.get("min_short_edge", 640))


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((16, 16))
        pixels = list(gray.get_flattened_data())
    average = sum(pixels) / len(pixels)
    return "".join("1" if value >= average else "0" for value in pixels)


def search_assets(
    scene_plan: dict[str, Any],
    search_config: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    providers = [str(item) for item in search_config.get("providers", ["met", "smithsonian", "wikimedia", "openverse"])]
    limit = int(search_config.get("candidates_per_shot", 6))
    threshold = int(search_config.get("recommendation_threshold", 70))
    historical_consistency = str(search_config.get("historical_consistency", "off"))
    allowed_rights = {
        str(item) for item in search_config.get("allowed_rights", ["pdm-1.0", "cc0-1.0"])
    }
    provider_status: dict[str, dict[str, Any]] = {
        name: {"status": "pending", "message": "", "count": 0, "successful_queries": 0, "failed_queries": 0}
        for name in providers
    }
    shots: list[dict[str, Any]] = []
    any_success = False
    previous_recommendation: str | None = None
    for intent in scene_plan.get("shots", []):
        queries = [str(item).strip() for item in intent.get("search_terms_en", []) if str(item).strip()]
        query = " ".join(queries[:3]) or str(intent.get("narration") or "")
        found: list[dict[str, Any]] = []
        shot_successes: set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(providers) or 1)) as pool:
            futures = {
                pool.submit(SEARCHERS[name], query, limit, env): name
                for name in providers
                if name in SEARCHERS
            }
            for future, name in futures.items():
                try:
                    values = future.result()
                    found.extend(values)
                    status = provider_status[name]
                    status["status"] = "succeeded"
                    status["count"] = int(status["count"]) + len(values)
                    status["successful_queries"] = int(status["successful_queries"]) + 1
                    shot_successes.add(name)
                    if name in {"met", "smithsonian", "wikimedia"}:
                        any_success = True
                except Exception as exc:
                    status = provider_status[name]
                    skipped = "缺少" in str(exc)
                    if status["status"] != "succeeded":
                        status["status"] = "skipped" if skipped else "failed"
                    status["message"] = str(exc)
                    status["failed_queries"] = int(status["failed_queries"]) + (0 if skipped else 1)
        unique: dict[str, dict[str, Any]] = {}
        for candidate in found:
            if candidate.get("rights_code") not in allowed_rights:
                candidate["selectable"] = False
                candidate["rejection_reason"] = "权利标识不在项目允许列表中"
            identity = hashlib.sha256(
                re.sub(r"^https?://", "", candidate.get("download_url", "")).lower().encode("utf-8")
            ).hexdigest()
            if identity not in unique or candidate.get("selectable"):
                unique[identity] = score_candidate(candidate, intent, historical_consistency)
        ranked = sorted(unique.values(), key=lambda item: (-int(item["score"]), item["asset_id"]))
        for candidate in ranked[: max(limit * 2, 8)]:
            if not candidate.get("width") or not candidate.get("height"):
                width, height, mime = probe_remote_dimensions(str(candidate.get("download_url", "")))
                candidate["width"] = width
                candidate["height"] = height
                candidate["mime"] = mime
                score_candidate(candidate, intent, historical_consistency)
        ranked = sorted(ranked, key=lambda item: (-int(item["score"]), item["asset_id"]))
        qualified = [item for item in ranked if item.get("selectable") and _valid_dimensions(item, search_config)]
        recommendation = next(
            (
                item["asset_id"]
                for item in qualified
                if int(item["score"]) >= threshold
                and item["asset_id"] != previous_recommendation
            ),
            None,
        )
        previous_recommendation = recommendation
        shots.append(
            {
                "shot_id": intent["shot_id"],
                "intent_id": intent["intent_id"],
                "start": intent.get("start", 0.0),
                "end": intent.get("end", 0.0),
                "narration": intent.get("narration", ""),
                "query": query,
                "successful_providers": sorted(shot_successes),
                "museum_source_succeeded": bool(
                    shot_successes & {"met", "smithsonian", "wikimedia"}
                ),
                "recommended_asset_id": recommendation,
                "candidates": ranked[:limit],
            }
        )
    return {
        "schema_version": 1,
        "searched_at": datetime.now().astimezone().isoformat(),
        "at_least_one_source_succeeded": any_success,
        "provider_status": provider_status,
        "shots": shots,
    }


def download_bytes(
    url: str,
    *,
    limit_bytes: int = 80 * 1024 * 1024,
    total_timeout: float = 180.0,
) -> tuple[bytes, str]:
    if not _https(url):
        raise AssetSourceError("只允许下载 HTTPS 原图。")
    url = normalize_http_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Video/3.0"})
    started = time.monotonic()
    chunks: list[bytes] = []
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            mime = str(response.headers.get_content_type())
            while True:
                if time.monotonic() - started > total_timeout:
                    raise AssetSourceError(
                        f"素材下载超过 {int(total_timeout)} 秒总时限。"
                    )
                chunk = response.read(min(256 * 1024, limit_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > limit_bytes:
                    raise AssetSourceError("素材文件超过 80MB 限制。")
            data = b"".join(chunks)
    except AssetSourceError:
        raise
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AssetSourceError(f"素材下载失败：{exc}") from exc
    return data, mime


def verify_image_bytes(
    data: bytes,
    mime: str,
    *,
    min_long_edge: int,
    min_short_edge: int,
) -> tuple[int, int, str]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            image_format = str(image.format or "").upper()
    except Exception as exc:
        raise AssetSourceError("下载内容不是可读取的图片，或魔数与文件内容不符。") from exc
    if image_format not in {"JPEG", "PNG", "WEBP"}:
        raise AssetSourceError(f"不支持的图片格式：{image_format}")
    if mime and not mime.startswith("image/"):
        raise AssetSourceError(f"来源返回了非图片 MIME：{mime}")
    if max(width, height) < min_long_edge or min(width, height) < min_short_edge:
        raise AssetSourceError(
            f"原图分辨率 {width}x{height} 低于 {min_long_edge}/{min_short_edge} 要求。"
        )
    return width, height, image_format.lower().replace("jpeg", "jpg")
