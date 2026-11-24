"""dp-paper arm (Maryam's DP scripts) and the authors' Table-3 cluster diversity."""
import unittest

import numpy as np

from dptig.config import load_config
from dptig.llm_prompt import GeneratedPrompt, PaperPromptGenerator, extract_paper_prompt
from dptig.metrics import cluster_diversity
from dptig.prompts import make_provider

TASKS = ["cifar10", "imagenet_pizza", "imagenet_teddy", "celebahq"]


class FakeRaw:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def complete_raw(self, prompt, seed, max_tokens=16):
        self.calls.append((prompt, seed))
        return self.outputs.pop(0)


class TestPaperPrompts(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(extract_paper_prompt("  A photo of a cat on grass "), "A photo of a cat on grass")
        self.assertEqual(extract_paper_prompt("Here is:\na photo of x"), "a photo of x")
        self.assertEqual(extract_paper_prompt("Sure! A photo of"), "")

    def test_retry_then_fallback(self):
        b = FakeRaw(["nope", "A photo of a dog in water"])
        g = PaperPromptGenerator(b, "Class: {category} {phrase}")
        r = g.generate(dict(category="dog", descriptor="water", phrase="in water"), seed=3, fallback="FB")
        self.assertEqual((r.prompt, r.attempts, r.fallback), ("A photo of a dog in water", 2, False))
        self.assertIn("Class: dog in water", b.calls[0][0])
        self.assertTrue(b.calls[0][0].startswith("<|start_header_id|>user"))
        self.assertNotEqual(b.calls[0][1], b.calls[1][1])
        g2 = PaperPromptGenerator(FakeRaw(["x"] * 5), "{category}")
        r2 = g2.generate(dict(category="dog", descriptor="", phrase=""), seed=0, fallback="FB")
        self.assertEqual((r2.prompt, r2.fallback), ("FB", True))

    def test_all_configs_paired_and_formatted(self):
        class Rec:
            def generate(self, v, seed, fallback):
                self.v = v
                return GeneratedPrompt("A photo of x", 1, False)
        for t in TASKS:
            c = load_config(f"configs/{t}.yaml")
            rec = Rec()
            paper, fp = make_provider("dp-paper", c, rec), make_provider("fp", c)
            for n in range(20):
                sp = paper.get(n)
                self.assertEqual(sp.expected_label, fp.get(n).expected_label)   # paired with FP
                c.paper_dp["instruction"].format(**rec.v)                       # no missing placeholder
            sd = c.sd_for("dp-paper")
            self.assertEqual(sd["lora"], c.sd["lora"])
            if t != "celebahq":
                self.assertEqual(len(sd["adapter_weights"]), 2)
            self.assertNotIn("style_lora", c.sd_for("fp"))

    def test_teddy_trigger(self):
        c = load_config("configs/imagenet_teddy.yaml")
        self.assertEqual(c.classes[0].fixed_prompt, "A photo of 1toy_bear teddy_bear")


class TestClusterDiversity(unittest.TestCase):
    def test_properties(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(12, 8)) * 10
        spread = centers[rng.integers(0, 12, 60)] + rng.normal(size=(60, 8)) * 0.1   # covers all clusters
        tight = np.repeat(centers[:1], 60, axis=0) + rng.normal(size=(60, 8)) * 0.1  # one cluster
        d = cluster_diversity({"a": spread, "b": tight}, combos={"ab": ["a", "b"]})
        self.assertEqual(len(d["a"]), 6)
        self.assertTrue(all(0 <= x <= 1 for v in d.values() for x in v))
        self.assertGreater(np.mean(d["a"]), np.mean(d["b"]) + 0.3)
        self.assertLess(np.mean(d["b"]), 0.2)
        # deterministic
        self.assertEqual(d, cluster_diversity({"a": spread, "b": tight}, combos={"ab": ["a", "b"]}))


if __name__ == "__main__":
    unittest.main()
