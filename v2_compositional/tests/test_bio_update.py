import json
import math
import pathlib
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bio_update import (  # noqa: E402
    capped_simplex,
    mirror_allocation,
    validate_trace,
    weighted_scale_std,
)


class BioUpdateMathTests(unittest.TestCase):
    def test_mirror_step_conserves_simplex(self):
        phi0 = [0.1, 0.2, 0.3, 0.4]
        phi = mirror_allocation(phi0, phi0, [1.0, -0.5, 0.25, -0.75], 0.1, 0.3)
        self.assertAlmostEqual(sum(phi), 1.0, places=14)
        self.assertTrue(all(value > 0.0 for value in phi))
        self.assertGreater(weighted_scale_std(phi, phi0), 0.0)

    def test_zero_signal_keeps_parameter_share_baseline(self):
        phi0 = [0.05, 0.15, 0.30, 0.50]
        phi = mirror_allocation(phi0, phi0, [0.0] * 4, 0.1, 0.3)
        for actual, expected in zip(phi, phi0):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_trust_one_is_frozen_control(self):
        phi0 = [0.1, 0.2, 0.3, 0.4]
        projected = capped_simplex([0.8, 0.1, 0.05, 0.05], phi0, trust=1.0)
        for actual, expected in zip(projected, phi0):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_capped_projection_preserves_budget(self):
        phi0 = [0.1, 0.2, 0.3, 0.4]
        projected = capped_simplex([0.9, 0.04, 0.03, 0.03], phi0, trust=2.0)
        scales = [p / p0 for p, p0 in zip(projected, phi0)]
        self.assertAlmostEqual(sum(projected), 1.0, places=13)
        self.assertAlmostEqual(sum(p0 * scale for p0, scale in zip(phi0, scales)), 1.0, places=13)
        self.assertLessEqual(max(scales), 2.0 + 1e-12)

    def test_trace_validator_detects_action_and_conservation(self):
        rows = [
            {"step": 0, "G": 3, "sum_phi": 1.0, "budget_gap": 0.0, "scale_std": 0.0},
            {"step": 1, "G": 3, "sum_phi": 1.0, "budget_gap": 1e-15, "scale_std": 0.02},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            verdict = validate_trace(str(path), require_active=True, minimum_scale_std=0.005)
        self.assertTrue(verdict["ok"], verdict["errors"])
        self.assertEqual(verdict["rows"], 2)

    def test_trace_validator_rejects_missing_action(self):
        row = {"step": 0, "G": 3, "sum_phi": 1.0, "budget_gap": 0.0, "scale_std": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            verdict = validate_trace(str(path), require_active=True, minimum_scale_std=0.005)
        self.assertFalse(verdict["ok"])
        self.assertTrue(any("inactive" in error for error in verdict["errors"]))

    def test_trace_validator_requires_split_resync(self):
        row = {"step": 0, "G": 3, "sum_phi": 1.0, "budget_gap": 0.0,
               "scale_std": 0.02, "resync": False}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            verdict = validate_trace(str(path), require_resync=True)
        self.assertFalse(verdict["ok"])
        self.assertTrue(any("resynchronisation" in error for error in verdict["errors"]))


if __name__ == "__main__":
    unittest.main()
