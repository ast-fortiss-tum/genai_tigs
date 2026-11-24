"""Compare two arms (e.g. FP vs DP) from their run directories (records.jsonl + failure PNGs).
  python -m dptig.analyze --fp runs/cifar10_fp --dp runs/cifar10_dp [--diversity] [--device mps]
Per level: seed validity, effectiveness (Fisher), iterations (Mann-Whitney + d), diversity (paired t + d)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stats
from .metrics import ClipExtractor, ResNetExtractor, cluster_diversity, load_images, paper_view, subsample_scores
from .metrics import diversity as eq6_diversity


def load(run: Path):
    return [json.loads(l) for l in (run / "records.jsonl").read_text().splitlines() if l.strip()]


def compare(fp: Path, dp: Path, levels=("low", "high"), diversity=False, device="cpu", k=30, m=20, seed=0,
            table3=False, view="none", kmeans_seeds=(0,)) -> dict:
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
    if table3:
        out["table3"] = table3_diversity(fp, dp, runs_all={"fp": A, "dp": B}, levels=levels, device=device, view=view,
                                         kmeans_seeds=kmeans_seeds)
    return out


def table3_diversity(fp: Path, dp: Path, runs_all, levels=("low", "high"), device="cpu", view="none",
                     kmeans_seeds=(0,)) -> dict:
    """Authors' Table-3 procedure: CLIP (L2) / ResNet-50 V2 (raw) features of the failure images of FP_LOW, FP_HIGH,
    DP_LOW, DP_HIGH; cluster_diversity over K=2..12; paired t-test FP vs DP over the 6 K values (their
    paired-ttest.py). CAVEAT: the K values reuse the same images, so they are not independent samples."""
    import numpy as np
    files = {f"{a}_{lv}": [d / r["runs"][lv]["file"] for r in runs_all[a]
                           if r["seed_ok"] and lv in r["runs"] and r["runs"][lv]["failed"]]
             for a, d in (("fp", fp), ("dp", dp)) for lv in levels}
    if min(len(v) for v in files.values()) < 2:
        return dict(insufficient=True, n={g: len(v) for g, v in files.items()})
    imgs = {g: [paper_view(i, view) for i in load_images(v)] for g, v in files.items()}
    res = dict(n={g: len(v) for g, v in files.items()}, view=view)
    for name, ext, l2 in (("clip", ClipExtractor(device), True), ("resnet50", ResNetExtractor(device, "IMAGENET1K_V2"), False)):
        feats = {}
        for g, im in imgs.items():
            z = ext(im).astype(np.float32)   # float32 as in the authors' script: KMeans results depend on it
            feats[g] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9) if l2 else z
        combos = {"fp_overall": [f"fp_{lv}" for lv in levels], "dp_overall": [f"dp_{lv}" for lv in levels]}
        r = {}
        for ks in kmeans_seeds:   # seed 0 = authors; more seeds show how much the result depends on the clustering
            d = cluster_diversity(feats, combos=combos, seed=ks)
            for cond in list(levels) + ["overall"]:
                a, b = np.array(d[f"fp_{cond}"]), np.array(d[f"dp_{cond}"])
                row = dict(D_fp=float(a.mean()), D_dp=float(b.mean()), per_k_fp=a.tolist(), per_k_dp=b.tolist(),
                           **stats.paired_t(b, a))
                if ks == kmeans_seeds[0]:
                    r[cond] = row
                if len(kmeans_seeds) > 1:
                    r.setdefault(f"{cond}_seeds", []).append(dict(seed=ks, D_fp=row["D_fp"], D_dp=row["D_dp"], p=row["p"]))
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", type=Path, required=True)
    ap.add_argument("--dp", type=Path, required=True)
    ap.add_argument("--diversity", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=30, help="diversity subsamples per condition")
    ap.add_argument("--m", type=int, default=20, help="images per subsample")
    ap.add_argument("--table3", action="store_true", help="authors' cluster diversity (Table 3), paired over K")
    ap.add_argument("--view", default="none", choices=["none", "cifar10", "imagenet", "celeba"],
                    help="image view for --table3 (what the authors saved: the classifier input)")
    ap.add_argument("--kmeans-seeds", type=int, default=1, help="--table3: repeat with KMeans seeds 0..n-1")
    a = ap.parse_args()
    print(json.dumps(compare(a.fp, a.dp, diversity=a.diversity, device=a.device, k=a.k, m=a.m,
                             table3=a.table3, view=a.view, kmeans_seeds=tuple(range(a.kmeans_seeds))), indent=1))


if __name__ == "__main__":
    main()
