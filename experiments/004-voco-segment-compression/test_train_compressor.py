import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from train_compressor import TeacherCacheDataset, validate_cache_dir


class TrainCompressorCacheTests(unittest.TestCase):
    def test_validate_cache_dir_rejects_nfs(self):
        with self.assertRaises(ValueError):
            validate_cache_dir(
                "/beegfs_hdd/data/nfs_share/users/lifanhong/nishome/video/teacher_cache_10k"
            )

    def test_teacher_cache_dataset_reads_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "teacher_shard_000.pt"
            shard_samples = [
                {
                    "segments": [torch.zeros(2, 4)],
                    "q_embeds": torch.zeros(3, 4),
                    "teacher_q_hidden": [torch.zeros(3, 4)],
                },
                {
                    "segments": [torch.ones(2, 4)],
                    "q_embeds": torch.ones(3, 4),
                    "teacher_q_hidden": [torch.ones(3, 4)],
                },
            ]
            torch.save(shard_samples, shard_path)

            dataset = TeacherCacheDataset(tmpdir)

            self.assertEqual(len(dataset), 2)
            sample0 = dataset[0]
            sample1 = dataset[1]
            self.assertTrue(torch.equal(sample0["q_embeds"], torch.zeros(3, 4)))
            self.assertTrue(torch.equal(sample1["q_embeds"], torch.ones(3, 4)))

    def test_teacher_cache_dataset_rejects_non_sharded_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save({"q_embeds": torch.zeros(1, 1)}, Path(tmpdir) / "000000.pt")

            with self.assertRaises(ValueError):
                TeacherCacheDataset(tmpdir)

    def test_teacher_cache_dataset_reuses_loaded_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "teacher_shard_000.pt"
            shard_samples = [
                {
                    "segments": [torch.zeros(2, 4)],
                    "q_embeds": torch.full((3, 4), idx, dtype=torch.float32),
                    "teacher_q_hidden": [torch.zeros(3, 4)],
                }
                for idx in range(2)
            ]
            torch.save(shard_samples, shard_path)

            real_load = torch.load
            load_calls = []

            def counting_load(*args, **kwargs):
                load_calls.append(Path(args[0]).name)
                return real_load(*args, **kwargs)

            with mock.patch("train_compressor.torch.load", side_effect=counting_load):
                dataset = TeacherCacheDataset(tmpdir)
                init_calls = list(load_calls)
                sample0 = dataset[0]
                sample1 = dataset[1]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(init_calls, ["teacher_shard_000.pt"])
            self.assertEqual(load_calls[1:], ["teacher_shard_000.pt"])
            self.assertTrue(torch.equal(sample0["q_embeds"], torch.zeros(3, 4)))
            self.assertTrue(torch.equal(sample1["q_embeds"], torch.ones(3, 4)))


if __name__ == "__main__":
    unittest.main()
