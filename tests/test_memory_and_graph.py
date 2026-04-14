import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from pnms import PNMSConfig, MemoryStore, MemoryGraph, MemorySlot, MEMORY_FORMAT_VERSION


class MemoryStoreAndGraphTests(unittest.TestCase):
    def test_write_and_merge_thresholds(self):
        cfg = PNMSConfig(embed_dim=4, write_new_slot_threshold=0.5, merge_similarity_threshold=0.8)
        store = MemoryStore(cfg, user_id="u1", graph=None)
        k1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        k2 = torch.tensor([0.9, 0.1, 0.0, 0.0])
        k3 = torch.tensor([0.0, 1.0, 0.0, 0.0])

        s1 = store.write(k1, "slot1")
        # k2 与 k1 相似度较高，应更新同一槽而不新增
        s2 = store.write(k2, "slot2")
        self.assertEqual(store.num_slots, 1)
        self.assertIs(s1, s2)
        self.assertEqual(s2.content, "slot2")
        self.assertGreaterEqual(s2.version, 1)

        # k3 与 k1 相似度较低，应新增槽
        s3 = store.write(k3, "slot3")
        self.assertEqual(store.num_slots, 2)
        self.assertNotEqual(s3.slot_id, s1.slot_id)

    def test_evict_and_graph_remove(self):
        cfg = PNMSConfig(embed_dim=4, max_slots_per_user=1)
        graph = MemoryGraph()
        store = MemoryStore(cfg, user_id="u1", graph=graph)
        k1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        k2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        s1 = store.write(k1, "slot1")
        # 记录一次自环外的共现边
        graph.record_cooccurrence([s1.slot_id])
        # 写入第二个槽会触发淘汰逻辑
        s2 = store.write(k2, "slot2")
        self.assertEqual(store.num_slots, 1)
        self.assertIn(s2.slot_id, store.slot_by_id)

    def test_graph_persistence_sqlite_roundtrip(self):
        graph = MemoryGraph()
        graph.record_cooccurrence(["a", "b", "c"])
        # 记录一次，边数应大于 0
        self.assertGreater(len(graph._edges), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            graph.save(db_path)
            self.assertTrue(db_path.exists())
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.execute(
                    "SELECT value FROM pnms_meta WHERE key = 'memory_format_version'"
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], MEMORY_FORMAT_VERSION)
            finally:
                conn.close()

            graph2 = MemoryGraph()
            graph2.load(db_path)
            self.assertEqual(len(graph2._edges), len(graph._edges))


if __name__ == "__main__":
    unittest.main()

