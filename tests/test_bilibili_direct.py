import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.bilibili_direct import (
    _account_from_env,
    _ensure_preupload_accepted,
    _normalize_cookie_value,
    _select_upload_line,
)


class BilibiliDirectTest(unittest.TestCase):
    def test_normalizes_plain_cookie_value(self) -> None:
        self.assertEqual(_normalize_cookie_value("bili_jct", " abc123 "), "abc123")

    def test_normalizes_named_cookie_value(self) -> None:
        self.assertEqual(
            _normalize_cookie_value("SESSDATA", "SESSDATA=token%2Cvalue; Path=/"),
            "token%2Cvalue",
        )

    def test_extracts_value_from_complete_cookie_header(self) -> None:
        self.assertEqual(
            _normalize_cookie_value(
                "DedeUserID", "SESSDATA=secret; bili_jct=csrf; DedeUserID=123456"
            ),
            "123456",
        )

    def test_loads_secure_env_credentials(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "bilibili.env"
            path.write_text(
                "BILIBILI_SESSDATA=" + "s" * 40 + "\n"
                "BILIBILI_BILI_JCT=0123456789abcdef0123456789abcdef\n"
                "BILIBILI_DEDE_USER_ID=123456\n",
                encoding="utf-8",
            )
            os.chmod(path, 0o600)
            account = _account_from_env(path)
            self.assertEqual(
                [item["name"] for item in account["cookie_info"]["cookies"]],
                ["SESSDATA", "bili_jct", "DedeUserID"],
            )

    def test_rejects_env_credentials_with_group_permissions(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "bilibili.env"
            path.write_text("BILIBILI_SESSDATA=secret\n", encoding="utf-8")
            os.chmod(path, 0o640)
            with self.assertRaisesRegex(RuntimeError, "permission 600"):
                _account_from_env(path)

    def test_preupload_rate_limit_fails_before_transfer(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"before transfer: HTTP 406, code 601, message 您上传视频过快",
        ):
            _ensure_preupload_accepted(
                406,
                {"code": 601, "message": "您上传视频过快，请您稍作休息后再继续"},
            )

    def test_preupload_accepts_upload_parameters(self) -> None:
        _ensure_preupload_accepted(200, {"chunk_size": 10_485_760})

    def test_line_probe_skips_broken_cdn(self) -> None:
        class Response:
            def __init__(self, status_code=200, payload=None):
                self.status_code = status_code
                self._payload = payload

            def json(self):
                return self._payload

        lines = [
            {"os": "upos", "query": "upcdn=broken", "probe_url": "//broken.invalid/OK"},
            {"os": "upos", "query": "upcdn=healthy", "probe_url": "//healthy.invalid/OK"},
        ]

        class Session:
            def get(self, url, **kwargs):
                if url == "https://member.bilibili.com/preupload":
                    self.asserted_params = kwargs["params"]
                    return Response(payload={"lines": lines})
                if url == "https://broken.invalid/OK":
                    raise OSError("expired certificate")
                return Response(status_code=200)

        class Client:
            _BiliBili__session = Session()

        selected = _select_upload_line(Client())
        self.assertEqual(selected["query"], "upcdn=healthy")
        self.assertEqual(Client._BiliBili__session.asserted_params, {"r": "probe"})


if __name__ == "__main__":
    unittest.main()
