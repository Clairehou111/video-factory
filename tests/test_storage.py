import sqlite3
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.storage import Workspace


class WorkspaceStorageTest(unittest.TestCase):
    def test_save_retries_a_transient_concurrent_sqlite_write_lock(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            with closing(sqlite3.connect(workspace.db_path)) as db:
                self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            locked = threading.Event()

            def hold_write_lock() -> None:
                db = sqlite3.connect(workspace.db_path, timeout=0.01)
                try:
                    db.execute("BEGIN EXCLUSIVE")
                    db.execute(
                        "INSERT OR REPLACE INTO records"
                        "(kind, id, payload, created_at) VALUES (?, ?, ?, datetime('now'))",
                        ("lock", "holder", "{}"),
                    )
                    locked.set()
                    time.sleep(0.18)
                    db.commit()
                finally:
                    db.close()

            thread = threading.Thread(target=hold_write_lock)
            thread.start()
            self.assertTrue(locked.wait(timeout=1))
            with patch.dict("os.environ", {
                "VIDEO_FACTORY_SQLITE_BUSY_TIMEOUT_SECONDS": "0.02",
                "VIDEO_FACTORY_SQLITE_WRITE_ATTEMPTS": "5",
            }):
                workspace.save_discovery_candidate({"id": "candidate", "value": 1})
            thread.join(timeout=1)

            self.assertEqual(
                workspace.load_discovery_candidate("candidate")["value"], 1,
            )


if __name__ == "__main__":
    unittest.main()
