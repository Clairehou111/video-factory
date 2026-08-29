import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.github_context import enrich_github_context
from video_factory.models import Candidate, SourceType
from video_factory.storage import Workspace


class GitHubContextTest(unittest.TestCase):
    def test_archives_only_readme_linked_context_and_discovers_official_source(self) -> None:
        readme = "Details: [vendor notes](docs/vendor-notes.md) and [website](https://example.com)."
        vendor_notes = b"Source: [Official docs](https://support.vendor.example/marks)."
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            candidate = Candidate("c", SourceType.GITHUB, "https://github.com/a/b", "a/b")
            evidence, sources, actions = enrich_github_context(
                candidate, readme, "a", "b", "main", workspace, root,
                lambda _url, _accept: vendor_notes,
            )
        self.assertEqual([item.source_kind for item in evidence], ["github:linked_context"])
        self.assertEqual(sources, ["https://support.vendor.example/marks"])
        self.assertEqual(actions[0]["status"], "archived")

    def test_project_without_context_link_does_not_gain_background_sources(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            candidate = Candidate("c", SourceType.GITHUB, "https://github.com/a/b", "a/b")
            evidence, sources, actions = enrich_github_context(
                candidate, "# Tool\nRun it with `python cli.py`.", "a", "b", "main",
                workspace, root, lambda _url, _accept: self.fail("must not fetch"),
            )
        self.assertEqual((evidence, sources, actions), ([], [], []))


if __name__ == "__main__":
    unittest.main()
