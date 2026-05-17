import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from compressor import InterSegmentAttention
from train_compressor import (
    TeacherCacheDataset,
    accumulate_segment_losses,
    compute_resume_position,
    compress_segments,
    validate_cache_dir,
)


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

    def test_teacher_cache_dataset_max_samples_stops_indexing_early(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard0 = Path(tmpdir) / "teacher_shard_000.pt"
            shard1 = Path(tmpdir) / "teacher_shard_001.pt"
            sample = {
                "segments": [torch.zeros(2, 4)],
                "q_embeds": torch.zeros(3, 4),
                "teacher_q_hidden": [torch.zeros(3, 4)],
            }
            torch.save([sample, sample], shard0)
            torch.save([sample, sample], shard1)

            real_load = torch.load
            load_calls = []

            def counting_load(*args, **kwargs):
                load_calls.append(Path(args[0]).name)
                return real_load(*args, **kwargs)

            with mock.patch("train_compressor.torch.load", side_effect=counting_load):
                dataset = TeacherCacheDataset(tmpdir, max_samples=1)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(load_calls, ["teacher_shard_000.pt"])

    def test_teacher_cache_dataset_shard_size_hint_only_loads_last_shard_per_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample = {
                "segments": [torch.zeros(2, 4)],
                "q_embeds": torch.zeros(3, 4),
                "teacher_q_hidden": [torch.zeros(3, 4)],
            }
            torch.save([sample, sample], Path(tmpdir) / "teacher_shard_rank0_000.pt")
            torch.save([sample, sample], Path(tmpdir) / "teacher_shard_rank0_001.pt")
            torch.save([sample], Path(tmpdir) / "teacher_shard_rank0_002.pt")
            torch.save([sample], Path(tmpdir) / "teacher_shard_rank1_000.pt")

            real_load = torch.load
            load_calls = []

            def counting_load(*args, **kwargs):
                load_calls.append(Path(args[0]).name)
                return real_load(*args, **kwargs)

            with mock.patch("train_compressor.torch.load", side_effect=counting_load):
                dataset = TeacherCacheDataset(tmpdir, shard_size_hint=2)

            self.assertEqual(len(dataset), 6)
            self.assertEqual(
                load_calls,
                ["teacher_shard_rank0_002.pt", "teacher_shard_rank1_000.pt"],
            )


class InterSegmentAttentionTests(unittest.TestCase):
    def test_inter_segment_attention_preserves_segment_shapes(self):
        mixer = InterSegmentAttention(dim=4, n_heads=2, n_layers=1, ffn_mult=1)
        tokens = torch.randn(6, 4)
        segment_lengths = [2, 4]

        outputs = mixer(tokens, segment_lengths)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (2, 4))
        self.assertEqual(outputs[1].shape, (4, 4))

    def test_compress_segments_applies_interaction_after_all_segments(self):
        class AddOneCompressor(torch.nn.Module):
            def forward(self, dense_vision):
                return dense_vision + 1

        class RecordingMixer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_shape = None
                self.seen_lengths = None

            def forward(self, tokens, segment_lengths):
                self.seen_shape = tuple(tokens.shape)
                self.seen_lengths = list(segment_lengths)
                return [part + 10 for part in torch.split(tokens, segment_lengths)]

        segments = [torch.zeros(2, 4), torch.ones(3, 4)]
        mixer = RecordingMixer()

        outputs = compress_segments(AddOneCompressor(), mixer, segments)

        self.assertEqual(mixer.seen_shape, (5, 4))
        self.assertEqual(mixer.seen_lengths, [2, 3])
        self.assertTrue(torch.equal(outputs[0], torch.full((2, 4), 11.0)))
        self.assertTrue(torch.equal(outputs[1], torch.full((3, 4), 12.0)))

    def test_accumulate_segment_losses_backprops_shared_interaction_graph_once(self):
        mixer = InterSegmentAttention(dim=4, n_heads=2, n_layers=1, ffn_mult=1)
        tokens = torch.randn(5, 4, requires_grad=True)
        outputs = mixer(tokens, [2, 3])
        losses = [out.square().mean() for out in outputs]

        total_loss, loss_value, n_segs = accumulate_segment_losses(losses)
        total_loss.backward()

        self.assertEqual(n_segs, 2)
        self.assertGreater(loss_value, 0.0)
        self.assertIsNotNone(tokens.grad)


class ResumePositionTests(unittest.TestCase):
    def test_epoch_checkpoint_resumes_from_next_epoch(self):
        resume = compute_resume_position(
            ckpt_epoch=2, ckpt_step=None, steps_per_epoch=2474,
        )

        self.assertEqual(resume["start_epoch"], 2)
        self.assertEqual(resume["skip_steps"], 0)
        self.assertEqual(resume["initial_global_step"], 4948)

    def test_step_checkpoint_resumes_inside_current_epoch(self):
        resume = compute_resume_position(
            ckpt_epoch=3, ckpt_step=5500, steps_per_epoch=2474,
        )

        self.assertEqual(resume["start_epoch"], 2)
        self.assertEqual(resume["skip_steps"], 552)
        self.assertEqual(resume["initial_global_step"], 4948)

    def test_first_epoch_step_checkpoint_resumes_inside_first_epoch(self):
        resume = compute_resume_position(
            ckpt_epoch=1, ckpt_step=500, steps_per_epoch=2474,
        )

        self.assertEqual(resume["start_epoch"], 0)
        self.assertEqual(resume["skip_steps"], 500)
        self.assertEqual(resume["initial_global_step"], 0)


if __name__ == "__main__":
    unittest.main()
