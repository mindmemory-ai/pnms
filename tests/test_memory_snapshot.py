"""记忆槽与个人状态 checkpoint 往返。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pnms import PNMS, PNMSConfig, PersistenceError, SimpleQueryEncoder


class MemorySnapshotTests(unittest.TestCase):
    def test_save_concept_modules_includes_memory_slots(self):
        cfg = PNMSConfig()
        cfg.embed_dim = 64
        cfg.concept_checkpoint_dir = None  # set per-tempdir below
        enc = SimpleQueryEncoder(embed_dim=64, vocab_size=10000)
        with tempfile.TemporaryDirectory() as tmpdir:
            ck = Path(tmpdir)
            cfg.concept_checkpoint_dir = str(ck)
            p = PNMS(config=cfg, user_id="u1", encoder=enc, device=torch.device("cpu"))
            p.handle_query(
                "hello",
                llm=lambda q, ctx: "world",
                content_to_remember="remember this",
            )
            self.assertGreater(p.store.num_slots, 0)
            p.save_concept_modules()

            slots_file = ck / "memory_slots.json"
            sess_file = ck / "memory_session.pt"
            self.assertTrue(slots_file.is_file())
            self.assertTrue(sess_file.is_file())

            p2 = PNMS(config=cfg, user_id="u1", encoder=enc, device=torch.device("cpu"))
            self.assertEqual(p2.store.num_slots, p.store.num_slots)
            self.assertEqual(p2._round_counter, p._round_counter)

    def test_load_rejects_newer_checkpoint_format(self):
        cfg = PNMSConfig()
        cfg.embed_dim = 64
        enc = SimpleQueryEncoder(embed_dim=64, vocab_size=10000)
        with tempfile.TemporaryDirectory() as tmpdir:
            ck = Path(tmpdir)
            cfg.concept_checkpoint_dir = str(ck)
            p = PNMS(config=cfg, user_id="u1", encoder=enc, device=torch.device("cpu"))
            p.handle_query("hello", llm=lambda q, ctx: "world", content_to_remember="remember this")
            p.save_concept_modules()

            meta_path = ck / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["memory_format_version"] = "9.0.0"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaises(PersistenceError):
                p.load_concept_modules(expected_memory_format_version="1.0.0")


if __name__ == "__main__":
    unittest.main()
