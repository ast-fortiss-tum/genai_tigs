"""Seed-validity check for a task: FP seed images per class -> fraction accepted by the classifier (cf. Table 2,
'% Seeds'). Used to sanity-check a (re)trained LoRA + classifier before running the search.

    python -m dptig.seed_check --config configs/celebahq.yaml --per-class 20 --out runs/seedcheck_celebahq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .classifiers import load_classifier
from .config import load_config
from .fitness import make_fitness
from .generator import SDGenerator, pick_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    dev = pick_device()
    gen, clf = SDGenerator(cfg.sd, dev), load_classifier(cfg.classifier, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report, thumbs = {}, []
    for ci, c in enumerate(cfg.classes):
        fit = make_fitness(cfg.fitness, c.label)
        ok = []
        for k in range(a.per_class):
            img = gen.seed_image(c.fixed_prompt, gen.latent(cfg.base_seed + 10_000 * ci + k))
            sc = clf.scores([img])
            valid = float(fit(sc)[0]) >= 0 if cfg.fitness == "target_logit" else int(np.argmax(sc[0])) == c.label
            ok.append(bool(valid))
            img.save(out / f"{c.name.replace(' ', '_')}_{k}_{'ok' if valid else 'NO'}.png")
            thumbs.append(img.resize((128, 128)))
        report[c.name] = dict(prompt=c.fixed_prompt, valid=sum(ok), n=len(ok), rate=sum(ok) / len(ok))
        print(f"{c.name!r}: {sum(ok)}/{len(ok)} valid", flush=True)
    report["overall_rate"] = float(np.mean([r["rate"] for r in report.values()]))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    cols = a.per_class
    grid = Image.new("RGB", (cols * 128, len(cfg.classes) * 128))
    for i, t in enumerate(thumbs):
        grid.paste(t, ((i % cols) * 128, (i // cols) * 128))
    grid.save(out / "grid.png")
    print("overall", report["overall_rate"])


if __name__ == "__main__":
    main()
