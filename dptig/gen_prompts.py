"""Pre-generate DP-TIG prompts (needs the LLM, not the GPU/SD stack) into a JSONL cache.

    python -m dptig.gen_prompts --config configs/cifar10.yaml --num 100

Copy prompts/<task>_dp.jsonl next to the code on the machine that runs Stable Diffusion; the LLM is then
never loaded there (DP-TIG reads every prompt from the cache).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .llm_prompt import build_paper_generator, build_prompt_generator
from .prompts import DiversePromptProvider, PaperPromptProvider


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int)
    ap.add_argument("--cache")
    ap.add_argument("--llm-model")
    ap.add_argument("--method", choices=["dp", "dp-paper"], default="dp")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cache = Path(a.cache or f"prompts/{cfg.name}_{a.method}.jsonl")
    if a.method == "dp-paper":
        prov = PaperPromptProvider(cfg, build_paper_generator(cfg, cache, a.llm_model))
    else:
        prov = DiversePromptProvider(cfg, build_prompt_generator(cfg.llm, cache, a.llm_model))
    fb = 0
    for n in range(a.num or cfg.num_seeds):
        sp = prov.get(n)
        fb += sp.meta["llm_fallback"]
        print(f"{n:3d} [{sp.category} + {sp.descriptor}] {sp.prompt}" + ("  (FALLBACK)" if sp.meta["llm_fallback"] else ""))
    print(f"done; {fb} fallback prompts")


if __name__ == "__main__":
    main()
