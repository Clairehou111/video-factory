from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import re
import sys
import stat
from datetime import datetime
from pathlib import Path
from typing import Any


MODERN_BROWSER_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "referer": "https://member.bilibili.com/platform/home",
}


def _load_biliup(source_dir: Path):
    sys.path.insert(0, str(source_dir))
    from biliup.plugins.bili_webup import BiliBili, Data

    return BiliBili, Data


def _read_account(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    cookies = value.get("cookie_info", {}).get("cookies", [])
    names = {item.get("name") for item in cookies if isinstance(item, dict)}
    if not {"SESSDATA", "bili_jct"} <= names:
        raise RuntimeError("Bilibili account file is missing SESSDATA or bili_jct")
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError("Bilibili credential env file must have permission 600")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise RuntimeError(f"Invalid credential env line for {key.strip() or 'unknown key'}")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _account_from_env(path: Path) -> dict[str, Any]:
    values = _read_env_file(path)
    sessdata = _normalize_cookie_value("SESSDATA", values.get("BILIBILI_SESSDATA", ""))
    bili_jct = _normalize_cookie_value("bili_jct", values.get("BILIBILI_BILI_JCT", ""))
    dede_user_id = _normalize_cookie_value("DedeUserID", values.get("BILIBILI_DEDE_USER_ID", ""))
    if len(sessdata) < 20 or sessdata == "SESSDATA":
        raise RuntimeError("BILIBILI_SESSDATA is missing or invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", bili_jct):
        raise RuntimeError("BILIBILI_BILI_JCT must be a 32-character hexadecimal value")
    if not dede_user_id.isdigit():
        raise RuntimeError("BILIBILI_DEDE_USER_ID must be numeric")
    return {
        "cookie_info": {"cookies": [
            {"name": "SESSDATA", "value": sessdata},
            {"name": "bili_jct", "value": bili_jct},
            {"name": "DedeUserID", "value": dede_user_id},
        ]},
        "token_info": {"access_token": "", "refresh_token": ""},
    }


def _resolve_account(account_file: Path, env_file: Path | None) -> dict[str, Any]:
    if env_file and env_file.is_file():
        return _account_from_env(env_file)
    return _read_account(account_file)


def _cookie_dict(account: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["name"]): str(item["value"])
        for item in account["cookie_info"]["cookies"]
        if isinstance(item, dict) and item.get("name") and item.get("value")
    }


def _normalize_cookie_value(name: str, raw: str) -> str:
    value = raw.strip().strip('"').strip("'")
    if ";" in value or f"{name}=" in value:
        pairs = {}
        for part in value.split(";"):
            key, separator, candidate = part.strip().partition("=")
            if separator:
                pairs[key] = candidate
        value = pairs.get(name, value.removeprefix(f"{name}="))
    return value.strip().strip('"').strip("'")


def _validate_creator_access(source_dir: Path, account: dict[str, Any]) -> None:
    BiliBili, Data = _load_biliup(source_dir)
    with BiliBili(Data()) as client:
        _prepare_client(client)
        try:
            client.login_by_cookies(account)
        except json.JSONDecodeError as error:
            raise RuntimeError("Bilibili nav endpoint returned a non-JSON response") from error
        try:
            result = client.tid_archive(_cookie_dict(account))
        except json.JSONDecodeError as error:
            raise RuntimeError("Bilibili creator endpoint returned a non-JSON response") from error
    if result.get("code") != 0:
        raise RuntimeError(f"Bilibili creator access check failed with code {result.get('code')}")


def _prepare_client(client: Any) -> None:
    session = getattr(client, "_BiliBili__session")
    session.headers.update(MODERN_BROWSER_HEADERS)


def login(source_dir: Path, account_file: Path, env_file: Path | None) -> None:
    if env_file and env_file.is_file():
        _validate_creator_access(source_dir, _account_from_env(env_file))
        print(json.dumps({"status": "login_completed", "creator_access": True, "source": "env_file"}))
        return
    sessdata = _normalize_cookie_value(
        "SESSDATA", getpass.getpass("请输入 Bilibili SESSDATA（输入不会显示）: ")
    )
    bili_jct = _normalize_cookie_value(
        "bili_jct", getpass.getpass("请输入 Bilibili bili_jct（输入不会显示）: ")
    )
    dede_user_id = _normalize_cookie_value(
        "DedeUserID", getpass.getpass("请输入 Bilibili DedeUserID（输入不会显示）: ")
    )
    if len(sessdata) < 20 or sessdata == "SESSDATA":
        raise RuntimeError("SESSDATA 格式不正确；请复制 Cookie 的 Value 列")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", bili_jct):
        raise RuntimeError("bili_jct 格式不正确；应为 32 位十六进制 Value")
    if not dede_user_id.isdigit():
        raise RuntimeError("DedeUserID 格式不正确；应为纯数字 Value")
    account = {
        "cookie_info": {
            "cookies": [
                {"name": "SESSDATA", "value": sessdata},
                {"name": "bili_jct", "value": bili_jct},
                {"name": "DedeUserID", "value": dede_user_id},
            ]
        },
        "token_info": {"access_token": "", "refresh_token": ""},
    }
    _validate_creator_access(source_dir, account)
    account_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = account_file.with_suffix(account_file.suffix + ".tmp")
    temporary.write_text(json.dumps(account, ensure_ascii=False), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, account_file)
    account_file.chmod(0o600)
    print(json.dumps({"status": "login_completed", "creator_access": True}))


def check(source_dir: Path, account_file: Path, env_file: Path | None) -> None:
    _validate_creator_access(source_dir, _resolve_account(account_file, env_file))
    print(json.dumps({"valid": True, "creator_access": True}))


def _source_url(description: str) -> str:
    match = re.search(r"https?://[^\s｜]+", description)
    return match.group(0).rstrip(".,;，。；") if match else ""


def _request_preupload(client: Any, video_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    line = _select_upload_line(client)
    query = {
        "r": line["os"], "profile": "ugcupos/bup", "ssl": 0,
        "version": "2.8.12", "build": 2081200,
        "name": video_path.name, "size": video_path.stat().st_size,
    }
    session = getattr(client, "_BiliBili__session")
    response = session.get(
        f"https://member.bilibili.com/preupload?{line['query']}",
        params=query, timeout=10,
    )
    try:
        result = response.json()
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Bilibili preupload returned HTTP {response.status_code} with non-JSON content"
        ) from error
    if not isinstance(result, dict):
        raise RuntimeError("Bilibili preupload returned an unexpected response")
    return response, result, line


def _select_upload_line(client: Any) -> dict[str, Any]:
    """Choose the first healthy UPOS line without letting one bad CDN abort the probe.

    Upstream ``BiliBili.probe`` returns immediately on a non-200 response and
    propagates TLS/network errors from any individual line. Bilibili can keep
    an expired or temporarily broken CDN in the probe list, so a single bad
    endpoint must not make every otherwise healthy upload line unusable.
    """
    session = getattr(client, "_BiliBili__session")
    response = session.get(
        "https://member.bilibili.com/preupload", params={"r": "probe"}, timeout=5,
    )
    try:
        payload = response.json()
    except json.JSONDecodeError as error:
        raise RuntimeError("Bilibili upload line probe returned non-JSON content") from error
    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not isinstance(lines, list):
        raise RuntimeError("Bilibili upload line probe returned no usable lines")
    failures: list[str] = []
    for line in lines:
        if not isinstance(line, dict) or line.get("os") != "upos":
            continue
        query = line.get("query")
        probe_url = line.get("probe_url")
        if not isinstance(query, str) or not isinstance(probe_url, str):
            continue
        url = probe_url if probe_url.startswith("https://") else f"https:{probe_url}"
        try:
            probe = session.get(url, timeout=5)
        except Exception as error:
            failures.append(f"{query}: {error.__class__.__name__}")
            continue
        if probe.status_code == 200:
            return line
        failures.append(f"{query}: HTTP {probe.status_code}")
    detail = "; ".join(failures[:4]) or "no UPOS entries"
    raise RuntimeError(f"Bilibili upload line probe returned no healthy UPOS line ({detail})")


def _ensure_preupload_accepted(http_status: int, result: dict[str, Any]) -> None:
    if result.get("code") not in (None, 0) or "chunk_size" not in result:
        code = result.get("code", "unknown")
        message = result.get("message") or "missing upload parameters"
        raise RuntimeError(
            "Bilibili preupload rejected upload before transfer: "
            f"HTTP {http_status}, code {code}, message {message}"
        )


def upload(
    source_dir: Path,
    account_file: Path,
    video_path: Path,
    title: str,
    description: str,
    tid: int,
    tags: list[str],
    thumbnail: Path | None,
    schedule: str | None,
    env_file: Path | None,
) -> None:
    account = _resolve_account(account_file, env_file)
    BiliBili, Data = _load_biliup(source_dir)
    video = Data()
    with BiliBili(video) as client:
        _prepare_client(client)
        client.login_by_cookies(account)
        creator = client.tid_archive(_cookie_dict(account))
        if creator.get("code") != 0:
            raise RuntimeError(f"Bilibili creator access check failed with code {creator.get('code')}")
        response, preupload, line = _request_preupload(client, video_path)
        _ensure_preupload_accepted(response.status_code, preupload)
        if line.get("os") != "upos":
            raise RuntimeError(f"Bilibili selected unsupported upload line {line.get('os')}")
        with video_path.open("rb") as video_file:
            part = asyncio.run(
                client.upos(video_file, video_path.stat().st_size, preupload, tasks=3)
            )
        part["title"] = video_path.stem[:80]
        video.append(part)
        video.title = title[:80]
        video.desc = description
        video.desc_v2 = [{"raw_text": description, "biz_id": "", "type": 1}]
        video.copyright = 2
        video.source = _source_url(description)
        video.tid = tid
        video.set_tag(tags)
        if schedule:
            video.dtime = int(datetime.fromisoformat(schedule).timestamp())
        if thumbnail:
            video.cover = client.cover_up(str(thumbnail)).replace("http:", "")
        result = client.submit("web")
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    print(json.dumps({
        "code": result.get("code"),
        "aid": data.get("aid") or result.get("aid"),
        "bvid": data.get("bvid") or result.get("bvid"),
    }, ensure_ascii=False))


def preflight_upload(
    source_dir: Path, account_file: Path, env_file: Path | None, video_path: Path,
) -> None:
    account = _resolve_account(account_file, env_file)
    BiliBili, Data = _load_biliup(source_dir)
    with BiliBili(Data()) as client:
        _prepare_client(client)
        client.login_by_cookies(account)
        response, result, line = _request_preupload(client, video_path)
    print(json.dumps({
        "http_status": response.status_code,
        "code": result.get("code"),
        "message": result.get("message"),
        "keys": sorted(key for key in result if key not in {"auth", "endpoint", "endpoints", "upos_uri"}),
        "has_chunk_size": "chunk_size" in result,
        "line_os": line.get("os"),
    }, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--account-file", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("login")
    subparsers.add_parser("check")
    preflight_parser = subparsers.add_parser("preflight-upload")
    preflight_parser.add_argument("--file", required=True, type=Path)
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--file", required=True, type=Path)
    upload_parser.add_argument("--title", required=True)
    upload_parser.add_argument("--desc", required=True)
    upload_parser.add_argument("--tid", required=True, type=int)
    upload_parser.add_argument("--tags", default="")
    upload_parser.add_argument("--thumbnail", type=Path)
    upload_parser.add_argument("--schedule")
    return parser


def main() -> None:
    logging.getLogger("biliup").setLevel(logging.WARNING)
    args = build_parser().parse_args()
    if args.action == "login":
        login(args.source_dir, args.account_file, args.env_file)
    elif args.action == "check":
        check(args.source_dir, args.account_file, args.env_file)
    elif args.action == "preflight-upload":
        preflight_upload(args.source_dir, args.account_file, args.env_file, args.file)
    else:
        upload(
            args.source_dir, args.account_file, args.file, args.title, args.desc,
            args.tid, [tag.strip() for tag in args.tags.split(",") if tag.strip()],
            args.thumbnail, args.schedule, args.env_file,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Bilibili direct publisher failed: {error}", file=sys.stderr)
        raise SystemExit(1)
