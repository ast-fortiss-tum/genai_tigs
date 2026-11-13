"""Per-task experiment loop (paper Algorithm 1, outer part). Method-agnostic via PromptProvider."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch

from .config import TaskConfig
from .fitness import make_fitness
from .ga import GAParams, run_search
from .prompts import PromptProvider


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["n"] for l in path.read_text().splitlines() if l.strip()}


def run_task(cfg: TaskConfig, provider: PromptProvider, gen, clf, out_dir: Path,
             levels: Iterable[str] = ("low", "high"), num_seeds: Optional[int] = None,
             log: Callable[[str], None] = print) -> Path:
    """For each of ``num_seeds`` latents: build the prompt, generate + validate the seed, then run the
    GA once per perturbation level (seeds are shared across levels, as in the paper).

    Writes <out_dir>/records.jsonl (one row per latent, incl. rejected seeds -> RQ1) and PNGs/npy of
    every failure-inducing image. Re-running resumes: latents already in records.jsonl are skipped.
    """
    out_dir = Path(out_dir)
    (out_dir / "seeds").mkdir(parents=True, exist_ok=True)
    for lv in levels:
        (out_dir / lv).mkdir(exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    done = _load_done(rec_path)
    n_total = num_seeds or cfg.num_seeds
    s = cfg.search

    for n in range(n_total):
        if n in done:
            continue
        t0 = time.time()
        spec = provider.get(n)
        z0 = gen.latent(cfg.base_seed + n)
        init_img = gen.seed_image(spec.prompt, z0)
        fitness_fn = make_fitness(cfg.fitness, spec.expected_label)
        seed_scores = clf.scores([init_img])
        seed_label = int(np.argmax(seed_scores[0]))
        # multi-label (target_logit): a seed is valid when the attribute is predicted present (fitness >= 0)
        seed_ok = (float(fitness_fn(seed_scores)[0]) >= 0) if cfg.fitness == "target_logit" \
            else seed_label == spec.expected_label
        rec: Dict = dict(n=n, latent_seed=cfg.base_seed + n, method=provider.method, prompt=spec.prompt,
                         category=spec.category, descriptor=spec.descriptor, expected=spec.expected_label,
                         seed_label=seed_label, seed_ok=bool(seed_ok), prompt_meta=spec.meta, runs={})
        if seed_ok:
            init_img.save(out_dir / "seeds" / f"seed_{n}_X{spec.expected_label}.png")

            def evaluate(latents, _p=spec.prompt, _f=fitness_fn):
                imgs = gen.batch(_p, latents)
                sc = clf.scores(imgs)
                return _f(sc), list(zip(imgs, sc))

            for lv in levels:
                params = GAParams(delta=cfg.delta(lv), bounds=cfg.bounds, pop_size=s["pop_size"],
                                  best_left=s["best_left"], max_iter=s["max_iter"])
                res = run_search(z0, evaluate, params, seed=cfg.base_seed + n)
                img, sc = res.best_payload
                pred = int(np.argmax(sc))
                stem = f"image_{n}_iteration{res.iterations}_X{spec.expected_label}_Y{pred}"
                if res.failed:
                    img.save(out_dir / lv / f"{stem}.png")
                rec["runs"][lv] = dict(failed=res.failed, iterations=res.iterations, best_fitness=res.best_fitness,
                                       pred=pred, file=f"{lv}/{stem}.png" if res.failed else None,
                                       delta=params.delta)
        rec["seconds"] = round(time.time() - t0, 1)
        with rec_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"[{n + 1}/{n_total}] {spec.category!r} seed_ok={seed_ok} "
            + " ".join(f"{lv}:{'F' if r['failed'] else '-'}@{r['iterations']}" for lv, r in rec["runs"].items()))
    return rec_path
