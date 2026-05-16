import tempfile
import unittest
from pathlib import Path

import torch

from pack_teacher_cache import pack_teacher_cache


class PackTeacherCacheTests(unittest.TestCase):
    def test_pack_teacher_cache_groups_small_files_into_shards(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            for idx in range(3):
                torch.save(
                    {
                        "segments": [torch.full((1, 2), idx)],
                        "q_embeds": torch.full((2, 2), idx),
                        "teacher_q_hidden": [torch.full((2, 2), idx)],
                    },
                    Path(input_dir) / f"{idx:06d}.pt",
                )

            shard_paths = pack_teacher_cache(input_dir, output_dir, shard_size=2)

            self.assertEqual(len(shard_paths), 2)
            shard0 = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
            shard1 = torch.load(shard_paths[1], map_location="cpu", weights_only=False)
            self.assertEqual(len(shard0), 2)
            self.assertEqual(len(shard1), 1)


if __name__ == "__main__":
    unittest.main()
