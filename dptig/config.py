"""Task configuration (YAML) and path resolution."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def weights_root() -> Path:
    """Weights live outside the legacy tree; override with GENAI_TIGS_WEIGHTS."""
    return Path(os.environ.get("GENAI_TIGS_WEIGHTS", Path(__file__).resolve().parent.parent / "weights"))


def resolve_weight(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else weights_root() / p


@dataclass
class ClassSpec:
    label: int                         # classifier label (or attribute index for CelebA)
    name: str                          # natural-language name (used by DP-TIG)
    fixed_prompt: str                  # fine-tuning concept prompt (used by FP-TIG)
    descriptors: List[str] = field(default_factory=list)   # DP-TIG conditioning descriptors (keywords)
    template_phrases: Dict[str, str] = field(default_factory=dict)    # keyword -> phrase for the LLM-free template arm
    descriptor_phrases: Dict[str, str] = field(default_factory=dict)  # optional keyword -> phrase shown to the LLM
    paper_vars: Dict[str, str] = field(default_factory=dict)  # extra placeholders for the dp-paper instruction


@dataclass
class TaskConfig:
    name: str
    sd: dict
    classifier: dict
    classes: List[ClassSpec]
    search: dict
    fitness: str = "logit_margin"
    num_seeds: int = 100
    base_seed: int = 0
    llm: dict = field(default_factory=dict)
    # dp-paper arm: Maryam's DP scripts (github.com/deeptestai/genai_tigs_diversity): instruction template,
    # optional style LoRA + adapter weights, whether the descriptor is sampled (else the LLM picks it)
    paper_dp: dict = field(default_factory=dict)

    def sd_for(self, method: str) -> dict:
        """SD settings for an arm: dp-paper adds the style LoRA (if configured) on top of the concept LoRA."""
        if method != "dp-paper":
            return self.sd
        return {**self.sd, **{k: self.paper_dp[k] for k in ("style_lora", "adapter_weights") if k in self.paper_dp}}

    @property
    def bounds(self):
        return tuple(self.search["bounds"])

    def delta(self, level: str) -> float:
        """Perturbation step: latent range divided by delta_divisor[level] (paper Sec. 5.5)."""
        lo, hi = self.bounds
        return (hi - lo) / float(self.search["delta_divisor"][level]) * float(self.search.get("delta_scale", 1.0))

    def sample_class(self, n: int) -> ClassSpec:
        """Deterministic per-latent class choice, shared by FP and DP so arms are paired."""
        return random.Random(f"{self.base_seed}-class-{n}").choice(self.classes)


def load_config(path: str | os.PathLike) -> TaskConfig:
    raw = yaml.safe_load(Path(path).read_text())
    classes = [ClassSpec(**c) for c in raw.pop("classes")]
    return TaskConfig(classes=classes, **raw)
