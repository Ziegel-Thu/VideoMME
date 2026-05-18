import unittest

from eval_compressor import get_checkpoint_config


class EvalCompressorCheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_none_inter_layers_means_zero(self):
        ckpt = {"K_seg": 8, "n_layers": 1, "inter_layers": None}

        config = get_checkpoint_config(ckpt)

        self.assertEqual(config, (8, 1, 0))


if __name__ == "__main__":
    unittest.main()
