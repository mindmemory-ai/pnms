import json
import tempfile
import unittest
from pathlib import Path

import torch

from pnms import ConceptModuleManager, MEMORY_FORMAT_VERSION


class ConceptPersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        embed_dim = 4
        mgr = ConceptModuleManager(embed_dim=embed_dim, concept_dim=2, top_m=1)
        center = torch.ones(embed_dim)
        slot_ids = ["s1", "s2"]
        mgr.add_module("c1", center, slot_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            mgr.save(ckpt_dir)

            meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta.get("memory_format_version"), MEMORY_FORMAT_VERSION)
            self.assertIn("pnms_library_version", meta)

            mgr2 = ConceptModuleManager(embed_dim=embed_dim, concept_dim=2, top_m=1)
            mgr2.load(ckpt_dir)

            mods = mgr2.retrieve_modules(torch.ones(embed_dim), top_m=1)
            self.assertTrue(mods)
            mid, _ = mods[0]
            self.assertEqual(mid, "c1")


if __name__ == "__main__":
    unittest.main()

