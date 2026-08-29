import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.tracks import TrackSegment, build_crossfade_track


class TrackCompositionTests(unittest.TestCase):
    def test_zero_fade_uses_clean_concat_without_overlapping_text_frames(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            segments = [
                TrackSegment(root / "one.mp4", 2.0),
                TrackSegment(root / "two.mp4", 2.0),
            ]
            with patch("video_factory.tracks.subprocess.run") as run:
                build_crossfade_track(segments, root / "out.mp4", fade_seconds=0)
            command = " ".join(str(item) for item in run.call_args.args[0])
            self.assertIn("concat=n=2:v=1:a=0", command)
            self.assertNotIn("xfade", command)


if __name__ == "__main__":
    unittest.main()
