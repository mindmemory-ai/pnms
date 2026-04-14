import json
import tempfile
import unittest
from pathlib import Path

import sqlite3

from pnms import (
    LIBRARY_VERSION,
    MEMORY_FORMAT_VERSION,
    get_library_version,
    get_memory_format_version,
    get_versions,
    peek_checkpoint_versions,
)
from pnms.versioning import MEMORY_FORMAT_VERSION as V_FMT


class VersioningTests(unittest.TestCase):
    def test_getters_match_constants(self):
        self.assertEqual(get_library_version(), LIBRARY_VERSION)
        self.assertEqual(get_memory_format_version(), MEMORY_FORMAT_VERSION)
        d = get_versions()
        self.assertEqual(d["library"], LIBRARY_VERSION)
        self.assertEqual(d["memory_format"], MEMORY_FORMAT_VERSION)

    def test_peek_checkpoint_versions_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info = peek_checkpoint_versions(tmpdir)
            self.assertIsNone(info["memory_format_in_meta"])
            self.assertIsNone(info["memory_format_in_graph"])
            self.assertEqual(info["current_library"], LIBRARY_VERSION)
            self.assertEqual(info["current_memory_format"], V_FMT)

    def test_peek_checkpoint_versions_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "meta.json").write_text(
                json.dumps(
                    {
                        "memory_format_version": "9.9.9",
                        "pnms_library_version": "0.0.1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            db = root / "graph.db"
            conn = sqlite3.connect(str(db))
            try:
                conn.execute(
                    "CREATE TABLE pnms_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.execute(
                    "INSERT INTO pnms_meta (key, value) VALUES ('memory_format_version', ?)",
                    (V_FMT,),
                )
                conn.commit()
            finally:
                conn.close()

            info = peek_checkpoint_versions(root)
            self.assertEqual(info["memory_format_in_meta"], "9.9.9")
            self.assertEqual(info["library_saved_in_meta"], "0.0.1")
            self.assertEqual(info["memory_format_in_graph"], V_FMT)


if __name__ == "__main__":
    unittest.main()
