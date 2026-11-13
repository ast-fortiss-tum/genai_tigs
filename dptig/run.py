"""CLI:  python -m dptig.run --config configs/cifar10.yaml --method fp --out runs/cifar10_fp"""
from __future__ import annotations

import argparse
from pathlib import Path

from .classifiers import load_classifier
from .config import load_config
from .generator import SDGenerator, pick_device
from .llm_prompt import build_prompt_generator
from .prompts import make_provider
from .runner import run_task


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["fp", "dp", "dp-template"], default="fp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-seeds", type=int)
    ap.add_argument("--prompt-cache", help="DP-TIG prompt cache (default prompts/<task>_dp.jsonl)")
    ap.add_argument("--llm-model", help="override llm.model (path under weights/ or absolute)")
    ap.add_argument("--levels", nargs="+", default=["low", "high"])
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    pg = None
    if a.method == "dp":
        pg = build_prompt_generator(cfg.llm, Path(a.prompt_cache or f"prompts/{cfg.name}_dp.jsonl"), a.llm_model)
    provider = make_provider(a.method, cfg, pg)
    device = pick_device()
    clf = load_classifier(cfg.classifier, device)
    gen = SDGenerator(cfg.sd, device)
    run_task(cfg, provider, gen, clf, Path(a.out), a.levels, a.num_seeds)


if __name__ == "__main__":
    main()
