"""Compare two arms (e.g. FP vs DP) from their run directories (records.jsonl + failure PNGs).
  python -m dptig.analyze --fp runs/cifar10_fp --dp runs/cifar10_dp [--diversity] [--device mps]
Per level: seed validity, effectiveness (Fisher), iterations (Mann-Whitney + d), diversity (paired t + d)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stats
from .metrics import ClipExtractor, ResNetExtractor, load_images, subsample_scores
from .metrics import diversity as eq6_diversity


def load(run: Path):
    return [json.loads(l) for l in (run / "records.jsonl").read_text().splitlines() if l.strip()]


def compare(fp: Path, dp: Path, levels=("low", "high"), diversity=False, device="cpu", k=30, m=20, seed=0) -> dict:
    A, B = load(fp), load(dp)
    out = dict(seeds=stats.fisher(sum(r["seed_ok"] for r in B), len(B), sum(r["seed_ok"] for r in A), len(A)))
    ext = {e.name: e for e in (ClipExtractor(device), ResNetExtractor(device))} if diversity else {}
    for lv in levels:
        runs = {n: [r for r in R if r["seed_ok"] and lv in r["runs"]] for n, R in (("fp", A), ("dp", B))}
        res = {}
        f = {n: [x["runs"][lv]["failed"] for x in R] for n, R in runs.items()}
        res["effectiveness"] = dict(fp=sum(f["fp"]), n_fp=len(f["fp"]), dp=sum(f["dp"]), n_dp=len(f["dp"]),
                                    **stats.fisher(sum(f["dp"]), len(f["dp"]), sum(f["fp"]), len(f["fp"])))
        it = {n: [x["runs"][lv]["iterations"] for x in R if x["runs"][lv]["failed"]] for n, R in runs.items()}
        if len(it["fp"]) > 1 and len(it["dp"]) > 1:
            res["iterations"] = stats.mann_whitney(it["dp"], it["fp"])
        if diversity:
            sc = {}
            files = {n: [d / x["runs"][lv]["file"] for x in runs[n] if x["runs"][lv]["failed"]]
                     for n, d in (("fp", fp), ("dp", dp))}
            nmin = min(len(v) for v in files.values())
            for en, e in ext.items():
                if nmin < 2:
                    continue
                feats = {n: e(load_images(v)) for n, v in files.items()}
                if nmin <= m:
                    # every size-m subsample would be the whole set -> identical scores, degenerate t-test.
                    # Report the full-set D (Eq. 6) descriptively, no test.
                    res[f"diversity_{en}"] = dict(insufficient=True, n_fp=len(files["fp"]), n_dp=len(files["dp"]),
                                                  m=m, D_dp=eq6_diversity(feats["dp"]), D_fp=eq6_diversity(feats["fp"]))
                    continue
                per = {n: subsample_scores(f, k, m, seed) for n, f in feats.items()}
                sc[en] = per
                res[f"diversity_{en}"] = dict(stats.paired_t(per["dp"], per["fp"]),
                                              n_fp=len(files["fp"]), n_dp=len(files["dp"]))
            if len(sc) == 2:
                res["diversity_corr"] = stats.correlations(
                    np.concatenate([sc["clip"]["dp"], sc["clip"]["fp"]]),
                    np.concatenate([sc["resnet50"]["dp"], sc["resnet50"]["fp"]]))
        out[lv] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", type=Path, required=True)
    ap.add_argument("--dp", type=Path, required=True)
    ap.add_argument("--diversity", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=30, help="diversity subsamples per condition")
    ap.add_argument("--m", type=int, default=20, help="images per subsample")
    a = ap.parse_args()
    print(json.dumps(compare(a.fp, a.dp, diversity=a.diversity, device=a.device, k=a.k, m=a.m), indent=1))


if __name__ == "__main__":
    main()
