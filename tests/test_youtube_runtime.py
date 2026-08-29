import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.youtube_runtime import (
    BGUTIL_VERSION, YTDLP_EJS_VERSION, YTDLP_VERSION,
    ManagedYouTubeRuntime, YouTubeRuntimeSettings,
)


class YouTubeRuntimeTest(unittest.TestCase):
    def test_status_and_extractor_args_use_mweb_script_provider(self) -> None:
        with TemporaryDirectory() as temp:
            settings = YouTubeRuntimeSettings(Path(temp))
            settings.executable.parent.mkdir(parents=True)
            settings.executable.write_text("", encoding="utf-8")
            settings.provider_server.mkdir(parents=True)
            settings.installation_path.write_text(json.dumps({
                "yt_dlp": YTDLP_VERSION,
                "yt_dlp_ejs": YTDLP_EJS_VERSION,
                "bgutil_provider": BGUTIL_VERSION,
                "provider_mode": "script",
            }), encoding="utf-8")
            runtime = ManagedYouTubeRuntime(settings)

            self.assertTrue(runtime.status()["installed"])
            self.assertTrue(runtime.status()["version_pins_valid"])
            self.assertEqual(runtime.require_executable(), str(settings.executable))
            arguments = runtime.extractor_arguments()
            self.assertIn("youtube:player_client=mweb", arguments)
            self.assertTrue(any("youtubepot-bgutilscript:server_home=" in item for item in arguments))

    def test_explicit_token_stays_on_mweb_and_omits_script_provider(self) -> None:
        with TemporaryDirectory() as temp, patch.dict(
            os.environ, {"VIDEO_FACTORY_YOUTUBE_PO_TOKEN": "token-value"}, clear=False,
        ):
            runtime = ManagedYouTubeRuntime(YouTubeRuntimeSettings(Path(temp)))
            arguments = runtime.extractor_arguments()
            self.assertIn("youtube:player_client=mweb;po_token=mweb.gvs+token-value", arguments)
            self.assertFalse(any("youtubepot-bgutilscript" in item for item in arguments))


if __name__ == "__main__":
    unittest.main()
