import unittest

import numpy as np

from dptig.metrics import diversity, subsample_scores
from dptig.stats import cohens_d, correlations, fisher, mann_whitney, paired_t


class MetricTests(unittest.TestCase):
    def test_identical_is_zero_orthogonal_is_one(self):
        self.assertAlmostEqual(diversity(np.ones((5, 4))), 0.0, places=9)
        self.assertAlmostEqual(diversity(np.eye(4)), 1.0, places=9)

    def test_scale_invariant_and_opposite(self):
        self.assertAlmostEqual(diversity(np.array([[1., 0], [-3., 0]])), 2.0, places=9)
        f = np.random.default_rng(0).normal(size=(6, 8))
        self.assertAlmostEqual(diversity(f), diversity(f * 7.0), places=9)

    def test_matches_naive_loop(self):
        f = np.random.default_rng(1).normal(size=(7, 5))
        s = [1 - f[i] @ f[j] / np.linalg.norm(f[i]) / np.linalg.norm(f[j]) for i in range(7) for j in range(i + 1, 7)]
        self.assertAlmostEqual(diversity(f), float(np.mean(s)), places=9)

    def test_subsample_pairable_and_deterministic(self):
        f = np.random.default_rng(2).normal(size=(30, 6))
        np.testing.assert_array_equal(subsample_scores(f, 5, 10, 3), subsample_scores(f, 5, 10, 3))
        self.assertEqual(len(subsample_scores(f, 4, 99, 0)), 4)

    def test_needs_two(self):
        with self.assertRaises(ValueError):
            diversity(np.ones((1, 3)))


class StatsTests(unittest.TestCase):
    def test_cohens_d_known(self):
        self.assertAlmostEqual(cohens_d([1, 2, 3], [2, 3, 4]), -1.0, places=9)

    def test_fisher_direction(self):
        r = fisher(90, 100, 50, 100)
        self.assertGreater(r["odds_ratio"], 1)
        self.assertLess(r["p"], 0.001)

    def test_paired_and_mw_and_corr(self):
        rng = np.random.default_rng(0)
        a = rng.normal(1, .1, 30); b = a - 0.5 + rng.normal(0, .05, 30)
        self.assertLess(paired_t(a, b)["p"], 1e-6)
        self.assertLess(mann_whitney(a, b)["p"], 1e-6)
        self.assertGreater(correlations(a, b)["pearson"], 0.8)


if __name__ == "__main__":
    unittest.main()


class AnalyzeTests(unittest.TestCase):
    def test_compare_without_diversity(self):
        import json, tempfile
        from pathlib import Path
        from dptig.analyze import compare
        with tempfile.TemporaryDirectory() as t:
            for name, rate in (("fp", 3), ("dp", 8)):
                d = Path(t) / name; d.mkdir()
                rows = [dict(n=i, seed_ok=True, runs={lv: dict(failed=i % 10 < rate, iterations=5 + i % 7, file=None)
                                                      for lv in ("low", "high")}) for i in range(100)]
                (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            r = compare(Path(t) / "fp", Path(t) / "dp")
            self.assertGreater(r["low"]["effectiveness"]["odds_ratio"], 1)
            self.assertIn("iterations", r["high"])


class SmallSampleTests(unittest.TestCase):
    def test_paired_t_degenerate(self):
        r = paired_t([0.3] * 30, [0.2] * 30)
        self.assertTrue(r.get("degenerate"))
        self.assertTrue(np.isnan(r["t"]))

    def test_diversity_insufficient_images(self):
        """Fewer failure images than m: full-set D reported, no test (was t~1e15, p=0)."""
        import json, tempfile
        from pathlib import Path
        from unittest import mock
        from PIL import Image
        from dptig import analyze

        class FakeExt:
            def __init__(self, name): self.name = name
            def __call__(self, imgs): return np.random.default_rng(len(imgs)).normal(size=(len(imgs), 8))

        with tempfile.TemporaryDirectory() as t:
            for name in ("fp", "dp"):
                d = Path(t) / name; (d / "low").mkdir(parents=True)
                rows = []
                for i in range(4):
                    Image.new("RGB", (8, 8), (i * 50, 0, 0)).save(d / "low" / f"{i}.png")
                    rows.append(dict(n=i, seed_ok=True, runs={"low": dict(failed=True, iterations=3, file=f"low/{i}.png")}))
                (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with mock.patch.object(analyze, "ClipExtractor", lambda dev: FakeExt("clip")), \
                    mock.patch.object(analyze, "ResNetExtractor", lambda dev: FakeExt("resnet50")):
                r = analyze.compare(Path(t) / "fp", Path(t) / "dp", levels=("low",), diversity=True, m=20)
                r2 = analyze.compare(Path(t) / "fp", Path(t) / "dp", levels=("low",), diversity=True, m=2, k=10)
        self.assertTrue(r["low"]["diversity_clip"]["insufficient"])
        self.assertIn("D_dp", r["low"]["diversity_clip"])
        self.assertNotIn("insufficient", r2["low"]["diversity_clip"])
