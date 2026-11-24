"""Prompt providers: the single point where FP-TIG and DP-TIG differ."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .config import TaskConfig


@dataclass
class PromptSpec:
    prompt: str
    expected_label: int
    category: str
    descriptor: Optional[str] = None   # None for FP-TIG
    meta: dict = field(default_factory=dict)   # e.g. LLM attempts / fallback flag


class PromptProvider(Protocol):
    method: str

    def get(self, n: int) -> PromptSpec:
        """Prompt (+ expected label) for the n-th initial latent. Must be deterministic in n."""


class FixedPromptProvider:
    """FP-TIG: the dataset-specific concept prompt used during LoRA fine-tuning."""
    method = "fp"

    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg

    def get(self, n: int) -> PromptSpec:
        c = self.cfg.sample_class(n)
        return PromptSpec(c.fixed_prompt, c.label, c.name)


class DiversePromptProvider:
    """DP-TIG: one LLM-generated target-descriptor prompt per initial latent (Eq. 2).

    The class for latent n is the same one FP-TIG picks (cfg.sample_class), so the two arms are paired;
    the descriptor is drawn uniformly from the class's descriptors, deterministically in n.
    """
    method = "dp"

    def __init__(self, cfg: TaskConfig, generator, method: str = "dp"):
        self.cfg, self.generator, self.method = cfg, generator, method

    def get(self, n: int) -> PromptSpec:
        c = self.cfg.sample_class(n)
        if not c.descriptors:
            raise ValueError(f"class {c.name!r} has no descriptors configured")
        s = random.Random(f"{self.cfg.base_seed}-desc-{n}").choice(c.descriptors)
        phrase = c.descriptor_phrases.get(s)
        if getattr(self.generator, "uses_template_phrases", False):
            phrase = c.template_phrases.get(s) or phrase
        g = self.generator.generate(c.name, s, seed=self.cfg.base_seed + n, descriptor_text=phrase,
                                    fallback_text=c.template_phrases.get(s))
        return PromptSpec(g.prompt, c.label, c.name, s, dict(llm_attempts=g.attempts, llm_fallback=g.fallback))


class PaperPromptProvider:
    """dp-paper: Maryam's DP prompts. Same class per latent as FP/DP (paired). The descriptor is sampled as in DP
    unless paper_dp.sample_descriptor is false (CelebA: the LLM picks it from the lists in the instruction)."""
    method = "dp-paper"

    def __init__(self, cfg: TaskConfig, generator):
        self.cfg, self.generator = cfg, generator

    def get(self, n: int) -> PromptSpec:
        c = self.cfg.sample_class(n)
        s = None
        if self.cfg.paper_dp.get("sample_descriptor", True):
            s = random.Random(f"{self.cfg.base_seed}-desc-{n}").choice(c.descriptors)
        pd = self.cfg.paper_dp.get("phrases", {})
        phrase = (pd.get(s) or c.template_phrases.get(s) or c.descriptor_phrases.get(s) or s) if s else ""
        v = {**dict(category=c.name, descriptor=s or "", phrase=phrase), **c.paper_vars}
        g = self.generator.generate(v, seed=self.cfg.base_seed + n, fallback=f"{c.fixed_prompt} {phrase}".strip())
        return PromptSpec(g.prompt, c.label, c.name, s, dict(llm_attempts=g.attempts, llm_fallback=g.fallback))


def make_provider(method: str, cfg: TaskConfig, generator=None) -> PromptProvider:
    if method == "dp-paper":
        if generator is None:
            raise ValueError("dp-paper needs a PaperPromptGenerator")
        return PaperPromptProvider(cfg, generator)
    if method == "fp":
        return FixedPromptProvider(cfg)
    if method == "dp":
        if generator is None:
            raise ValueError("DP-TIG needs a PromptGenerator")
        return DiversePromptProvider(cfg, generator)
    if method == "dp-template":
        from .llm_prompt import TemplatePromptGenerator
        return DiversePromptProvider(cfg, TemplatePromptGenerator(), method="dp-template")
    raise ValueError(f"unknown method: {method}")
