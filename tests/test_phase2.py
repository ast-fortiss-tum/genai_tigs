import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dptig.config import load_config
from dptig.descriptors import extract_terms
from dptig.llm_prompt import (INSTRUCTION, LazyBackend, PromptGenerator, clean, fallback_prompt,
                              user_message, validate_prompt)
from dptig.prompts import DiversePromptProvider, FixedPromptProvider, make_provider

ROOT = Path(__file__).resolve().parent.parent


class Stub:
    """Scripted LLM: returns the queued outputs in order, recording every call."""
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
    def complete(self, instruction, user, seed):
        self.calls.append((instruction, user, seed))
        return self.outputs.pop(0)


class ValidateTests(unittest.TestCase):
    def test_paper_examples_valid(self):
        for p, c, s in [("A photo of a pizza with pepperoni topping.", "pizza", "pepperoni"),
                        ("A photo of a teddy bear sitting on a chair.", "teddy bear", "chair"),
                        ("A photo of a dog on road.", "dog", "road")]:
            self.assertTrue(validate_prompt(p, c, s)[0], p)

    def test_rejections(self):
        v = lambda p: validate_prompt(p, "dog", "road")
        self.assertFalse(v("")[0])
        self.assertIn("begin", v("Picture of a dog on road")[1])
        self.assertIn("category", v("A photo of a cat on road")[1])
        self.assertIn("descriptor", v("A photo of a dog on grass")[1])
        self.assertIn("words", v("A photo of a dog standing on a long winding road near the hills")[1])

    def test_word_limit_boundary(self):
        eleven = "A photo of a dog on road " + "x " * 4
        self.assertEqual(len(eleven.split()), 11)
        self.assertTrue(validate_prompt(eleven.strip(), "dog", "road")[0])
        self.assertFalse(validate_prompt(eleven.strip() + " y", "dog", "road")[0])

    def test_clean(self):
        self.assertEqual(clean('  "A photo of a dog on road."\nSure! more text'), "A photo of a dog on road.")
        self.assertEqual(clean("\n\n  "), "")

    def test_instruction_matches_fig3_rules(self):
        for frag in ["under 12 words", "Output only the prompt", "A photo of a [target category]"]:
            self.assertIn(frag, INSTRUCTION)


class GeneratorTests(unittest.TestCase):
    def test_retries_then_succeeds_with_distinct_seeds(self):
        st = Stub(["A cat on road", "Sure, here: A photo of a dog on road", "A photo of a dog on road."])
        g = PromptGenerator(st, max_retries=5).generate("dog", "road", seed=7)
        self.assertEqual((g.prompt, g.attempts, g.fallback), ("A photo of a dog on road.", 3, False))
        self.assertEqual(len({c[2] for c in st.calls}), 3)            # fresh sampling seed per attempt
        self.assertEqual(st.calls[0][2], 7)
        self.assertEqual(st.calls[0][1], user_message("dog", "road"))

    def test_fallback_after_exhausting_retries(self):
        g = PromptGenerator(Stub(["bad"] * 3), max_retries=3).generate("airplane", "grass", seed=1)
        self.assertTrue(g.fallback)
        self.assertEqual(g.prompt, "A photo of an airplane grass")
        self.assertTrue(validate_prompt(g.prompt, "airplane", "grass")[0])

    def test_uses_phrase_for_llm_but_keyword_for_validation(self):
        st = Stub(["A photo of a pizza on a plate."])
        g = PromptGenerator(st).generate("pizza", "plate", 0, descriptor_text="on a plate")
        self.assertFalse(g.fallback)
        self.assertIn("on a plate", st.calls[0][1])

    def test_cache_persists_and_skips_llm(self):
        with tempfile.TemporaryDirectory() as d:
            cp = Path(d) / "c.jsonl"
            PromptGenerator(Stub(["A photo of a dog on road."]), cp).generate("dog", "road", 3)
            st = Stub([])                                             # would raise if called
            g = PromptGenerator(st, cp).generate("dog", "road", 3)
            self.assertEqual(g.prompt, "A photo of a dog on road.")
            self.assertEqual(st.calls, [])
            self.assertEqual(len(cp.read_text().splitlines()), 1)

    def test_no_backend_and_cache_miss_is_an_error(self):
        with self.assertRaises(RuntimeError):
            PromptGenerator(None).generate("dog", "road", 0)

    def test_lazy_backend_loads_once_and_only_when_used(self):
        n = []
        lb = LazyBackend(lambda: (n.append(1), Stub(["x", "y"]))[1])
        self.assertEqual(n, [])
        lb.complete("i", "u", 0); lb.complete("i", "u", 1)
        self.assertEqual(n, [1])


class EchoLLM:
    """Plausible LLM: 'A photo of a <category> <descriptor text>.'"""
    def complete(self, instruction, user, seed):
        c = user.split("\n")[0].split(": ")[1]; s = user.split("\n")[1].split(": ")[1]
        return f"A photo of a {c} {s}."


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs" / "cifar10.yaml")

    def test_pairs_with_fp_and_is_deterministic(self):
        dp = DiversePromptProvider(self.cfg, PromptGenerator(EchoLLM()))
        fp = FixedPromptProvider(self.cfg)
        for n in range(30):
            a, b, f = dp.get(n), dp.get(n), fp.get(n)
            self.assertEqual(a.prompt, b.prompt)
            self.assertEqual((a.expected_label, a.category), (f.expected_label, f.category))   # paired arms
            self.assertIn(a.descriptor, ["water", "grass", "road"])
            self.assertTrue(a.prompt.startswith("A photo of"))
            self.assertIn("llm_fallback", a.meta)

    def test_covers_all_target_descriptor_combinations(self):
        dp = DiversePromptProvider(self.cfg, PromptGenerator(EchoLLM()))
        combos = Counter((dp.get(n).category, dp.get(n).descriptor) for n in range(600))
        self.assertEqual(len(combos), 30)                              # |C| x |S| = 10 x 3 (Eq. 1)

    def test_pizza_uses_phrases(self):
        cfg = load_config(ROOT / "configs" / "imagenet_pizza.yaml")
        dp = DiversePromptProvider(cfg, PromptGenerator(EchoLLM()))
        seen = {dp.get(n).prompt for n in range(40)}
        self.assertTrue(seen <= {"A photo of a pizza on a plate.", "A photo of a pizza with pepperoni topping.",
                                 "A photo of a pizza as a slice of pizza."}, seen)

    def test_make_provider(self):
        self.assertEqual(make_provider("dp", self.cfg, PromptGenerator(EchoLLM())).method, "dp")
        with self.assertRaises(ValueError):
            make_provider("dp", self.cfg)


class DescriptorTests(unittest.TestCase):
    def test_extract_terms(self):
        caps = ["a dog standing on the grass", "a dog running in grass near a road", "a black dog on a road"]
        r = dict(extract_terms(caps, exclude=["dog"], top_k=5))
        self.assertEqual(r, {"grass": 2, "road": 2})                  # min_count=2, class word + stopwords gone
        self.assertNotIn("dog", r)

    def test_counted_once_per_caption(self):
        self.assertEqual(extract_terms(["grass grass grass", "road"], min_count=1), [("grass", 1), ("road", 1)])


if __name__ == "__main__":
    unittest.main()
