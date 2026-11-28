import pathlib
import sys
import tempfile
import unittest

import torch


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_v3 import load_data, resolve_device  # noqa: E402


class RuntimeDataTests(unittest.TestCase):
    def test_token_cache_is_sampling_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            data = pathlib.Path(directory) / "tiny.txt"
            cache = pathlib.Path(directory) / "tiny.tokens.pt"
            data.write_bytes(b"abc" * 333_334)

            first, vocab_a = load_data(str(data), 16, 4,
                                       cache_path=str(cache))
            self.assertTrue(cache.exists())
            torch.manual_seed(123)
            xa, ya = first("train")

            second, vocab_b = load_data(str(data), 16, 4,
                                        cache_path=str(cache))
            torch.manual_seed(123)
            xb, yb = second("train")

            self.assertEqual(vocab_a, vocab_b)
            self.assertTrue(torch.equal(xa, xb))
            self.assertTrue(torch.equal(ya, yb))
            self.assertEqual(xb.dtype, torch.long)

    def test_explicit_unavailable_accelerator_fails(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda")


if __name__ == "__main__":
    unittest.main()
