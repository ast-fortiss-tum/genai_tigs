import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dptig.config import load_config
from dptig.fitness import margin_fitness, target_logit_fitness
from dptig.ga import GAParams, run_search
from dptig.prompts import FixedPromptProvider
from dptig.runner import run_task

ROOT = Path(__file__).resolve().parent.parent


class FitnessTests(unittest.TestCase):
    def test_margin_matches_legacy(self):
        def legacy(logit, label):  # verbatim from tig_sd_cifar10.py
            best1, best2 = np.argsort(-logit)[:2]
            other = best2 if best1 == label else best1
            return logit[label] - logit[other]
        rng = np.random.default_rng(0)
        s = rng.normal(size=(50, 10))
        for lab in (0, 3, 9):
            np.testing.assert_allclose(margin_fitness(s, lab), [legacy(r, lab) for r in s])

    def test_sign_means_misclassified(self):
        s = np.array([[5., 1., 0.], [1., 5., 0.]])
        f = margin_fitness(s, 0)
        self.assertGreater(f[0], 0); self.assertLess(f[1], 0)

    def test_softmax_margin_bounded(self):
        f = margin_fitness(np.array([[100., 0., 0.]]), 0, softmax=True)
        self.assertTrue(0 < f[0] <= 1)

    def test_target_logit(self):
        np.testing.assert_array_equal(target_logit_fitness(np.array([[1., -2.], [3., 4.]]), 1), [-2., 4.])


def fake_evaluate_factory(threshold, calls):
    """fitness = threshold - mean(latent): decreases as the latent mean grows, so the GA can 'succeed'."""
    def ev(latents):
        calls.append(latents.clone())
        fit = (threshold - latents.float().mean(dim=(1, 2, 3))).numpy()
        return fit, [f"img{i}" for i in range(len(fit))]
    return ev


class GATests(unittest.TestCase):
    def setUp(self):
        self.z0 = torch.zeros(1, 4, 8, 8)

    def test_stops_on_negative_fitness(self):
        calls = []
        # init pop mean ~ delta noise ~ 0 ; threshold tiny so a few mutations suffice
        p = GAParams(delta=0.5, bounds=(-5, 5), max_iter=250)
        r = run_search(self.z0, fake_evaluate_factory(0.05, calls), p, seed=1)
        self.assertTrue(r.failed)
        self.assertEqual(r.iterations, len(calls))
        self.assertLess(r.best_fitness, 0)
        self.assertTrue(r.best_payload.startswith("img"))

    def test_budget_exhausted_returns_max_iter(self):
        calls = []
        p = GAParams(delta=1e-4, bounds=(-5, 5), max_iter=7)
        r = run_search(self.z0, fake_evaluate_factory(1e6, calls), p, seed=1)
        self.assertFalse(r.failed); self.assertEqual(r.iterations, 7); self.assertEqual(len(calls), 7)
        self.assertTrue(all(c.shape == (25, 4, 8, 8) for c in calls))

    def test_clamping_and_pop_size(self):
        calls = []
        p = GAParams(delta=10.0, bounds=(-0.1, 0.1), max_iter=5)
        run_search(self.z0, fake_evaluate_factory(1e6, calls), p, seed=2)
        for c in calls[1:]:                       # generations after the first are clamped
            self.assertLessEqual(c.abs().max().item(), 0.1 + 1e-6)

    def test_deterministic_given_seed(self):
        a, b = [], []
        p = GAParams(delta=0.3, bounds=(-5, 5), max_iter=4)
        run_search(self.z0, fake_evaluate_factory(1e6, a), p, seed=3)
        run_search(self.z0, fake_evaluate_factory(1e6, b), p, seed=3)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_adaptive_delta_doubles_on_stall(self):
        # constant fitness -> best never improves after iteration 1 -> delta doubles each generation.
        # The mutation noise std seen in the population's deviation from parents therefore grows.
        calls = []
        def ev(latents):
            calls.append(latents.clone()); return np.ones(len(latents)), list(range(len(latents)))
        p = GAParams(delta=0.01, bounds=(-1e9, 1e9), max_iter=6)
        run_search(self.z0, ev, p, seed=0)
        spread = [c[10:].std().item() for c in calls[1:]]   # offspring rows
        self.assertTrue(all(b > a for a, b in zip(spread, spread[1:])), spread)


class FakeGen:
    def latent(self, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, 4, 8, 8, generator=g)
    def seed_image(self, prompt, latent):
        return Image.new("RGB", (16, 16), (int(abs(latent.sum().item())) % 255,) * 3)
    def batch(self, prompt, latents):
        return [Image.new("RGB", (16, 16), (int(abs(z.sum().item() * 50)) % 255,) * 3) for z in latents]


class FakeClf:
    """10-class 'classifier': class 0 score falls as the image gets brighter -> perturbations can flip it."""
    def scores(self, images):
        out = []
        for im in images:
            v = np.asarray(im)[0, 0, 0] / 255.0
            s = np.zeros(10); s[0] = 1.0 - 2 * v + 0.5; s[1] = 0.3
            out.append(s)
        return np.array(out)


class RunnerTests(unittest.TestCase):
    def test_end_to_end_with_fakes_and_resume(self):
        cfg = load_config(ROOT / "configs" / "cifar10.yaml")
        cfg.search["max_iter"] = 5
        cfg.classes = cfg.classes[:1]                      # single class, label 0
        with tempfile.TemporaryDirectory() as d:
            p = run_task(cfg, FixedPromptProvider(cfg), FakeGen(), FakeClf(), Path(d), num_seeds=4, log=lambda m: None)
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            self.assertEqual([r["n"] for r in rows], [0, 1, 2, 3])
            for r in rows:
                self.assertEqual(r["method"], "fp")
                self.assertEqual(r["prompt"], "A photo of A1plane0 cifar10_0")
                if r["seed_ok"]:
                    self.assertEqual(set(r["runs"]), {"low", "high"})
                    self.assertAlmostEqual(r["runs"]["low"]["delta"] * 10, r["runs"]["high"]["delta"], places=6)
            run_task(cfg, FixedPromptProvider(cfg), FakeGen(), FakeClf(), Path(d), num_seeds=4, log=lambda m: None)
            self.assertEqual(len(p.read_text().splitlines()), 4)   # resume: nothing re-run

    def test_config_deltas_match_legacy_ranges(self):
        c = load_config(ROOT / "configs" / "cifar10.yaml")
        self.assertAlmostEqual(c.delta("low"), 0.00109171276092529, places=9)    # legacy perturbation_size
        i = load_config(ROOT / "configs" / "imagenet_pizza.yaml")
        self.assertAlmostEqual(i.delta("high"), 0.0108763074874878, places=9)    # legacy perturbation_size


class ResizeTests(unittest.TestCase):
    def test_nearest_matches_cv2_formula(self):
        from dptig.classifiers import _nearest_resize
        a = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
        r = _nearest_resize(a, 32)
        self.assertEqual(r.shape, (32, 32, 3))
        np.testing.assert_array_equal(r[1, 2], a[16, 32])      # floor(1*16)=16, floor(2*16)=32
        try:
            import cv2
            np.testing.assert_array_equal(r, cv2.resize(a, (32, 32), interpolation=cv2.INTER_NEAREST))
        except ImportError:
            pass


class RealCifarWeights(unittest.TestCase):
    @unittest.skipUnless((ROOT / "weights/classifiers/cifar10/CIFAR10_cifar10_train.pynet.pth").exists(), "no weights")
    def test_state_dict_loads_and_predicts(self):
        from dptig.classifiers import load_classifier
        clf = load_classifier({"kind": "cifar10_vgg", "weights": "classifiers/cifar10/CIFAR10_cifar10_train.pynet.pth"},
                              torch.device("cpu"))
        s = clf.scores([Image.new("RGB", (512, 512), (120, 60, 30))] * 2)
        self.assertEqual(s.shape, (2, 10)); self.assertTrue(np.isfinite(s).all())


if __name__ == "__main__":
    unittest.main()
