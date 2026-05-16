import tempfile
import unittest
from pathlib import Path

import torch

from extract_teacher import ShardWriter


class ExtractTeacherShardWriterTests(unittest.TestCase):
    def test_shard_writer_flushes_by_shard_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ShardWriter(tmpdir, shard_size=2, rank=0)
            writer.add({"id": 0, "q_embeds": torch.zeros(1, 1)})
            writer.add({"id": 1, "q_embeds": torch.ones(1, 1)})
            writer.add({"id": 2, "q_embeds": torch.full((1, 1), 2)})
            writer.close()

            shard_paths = sorted(Path(tmpdir).glob("teacher_shard_rank0_*.pt"))
            self.assertEqual(len(shard_paths), 2)
            shard0 = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
            shard1 = torch.load(shard_paths[1], map_location="cpu", weights_only=False)
            self.assertEqual(len(shard0), 2)
            self.assertEqual(len(shard1), 1)


if __name__ == "__main__":
    unittest.main()
