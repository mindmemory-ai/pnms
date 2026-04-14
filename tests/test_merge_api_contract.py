import tempfile
import unittest
from pathlib import Path

import torch

from pnms import (
    ErrorCodes,
    PNMS,
    PNMSConfig,
    PersistenceError,
    SimpleQueryEncoder,
)


class MergeApiContractTests(unittest.TestCase):
    def _build_engine(self) -> PNMS:
        cfg = PNMSConfig()
        cfg.embed_dim = 64
        enc = SimpleQueryEncoder(embed_dim=64, vocab_size=10000)
        return PNMS(config=cfg, user_id="u1", encoder=enc, device=torch.device("cpu"))

    def test_merge_rejects_missing_source_checkpoint(self):
        engine = self._build_engine()
        with self.assertRaises(PersistenceError) as cm:
            engine.merge_memories("/tmp/not-exist-merge-source")
        self.assertEqual(cm.exception.code, ErrorCodes.MERGE_CHECKPOINT_NOT_FOUND)

    def test_merge_rejects_newer_source_version(self):
        engine = self._build_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(PersistenceError) as cm:
                engine.merge_memories(str(root), source_memory_format_version="9.0.0")
            self.assertEqual(cm.exception.code, ErrorCodes.MERGE_VERSION_INCOMPATIBLE)

    def test_merge_completes_when_no_source_slots(self):
        engine = self._build_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            engine.merge_memories(str(root), source_memory_format_version="1.0.0")

    def test_merge_imports_slots_from_source_checkpoint(self):
        cfg = PNMSConfig()
        cfg.embed_dim = 64
        enc = SimpleQueryEncoder(embed_dim=64, vocab_size=10000)
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "remote"
            cfg.concept_checkpoint_dir = str(src)
            remote = PNMS(config=cfg, user_id="u1", encoder=enc, device=torch.device("cpu"))
            remote.handle_query("a", llm=lambda q, ctx: "b", content_to_remember="from remote")
            remote.save_concept_modules()

            engine = self._build_engine()
            n0 = engine.store.num_slots
            engine.merge_memories(str(src), source_memory_format_version="1.0.0")
            self.assertGreater(engine.store.num_slots, n0)


if __name__ == "__main__":
    unittest.main()
