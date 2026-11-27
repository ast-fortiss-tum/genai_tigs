"""Pre-registered RQ6 diversity test (WORKLOG (23), fixed 2026-11-27 before looking at any result it produces).

PRIMARY:   Eq. 6 (mean pairwise cosine distance), ResNet-50 IMAGENET1K_V2 penultimate features of the classifier
           input (paper_view), failure images of valid seeds, per level (low / high), each arm vs FP.
           Two-sided permutation test on D_arm - D_fp (image labels shuffled, group sizes kept, 10,000 permutations)
           + 95% percentile bootstrap CI (images resampled within each arm, 2,000 resamples).
           Holm correction per arm over all task x level tests available.
SECONDARY: the same with CLIP ViT-B/32 (L2-normalised).
Seeds: first n = min(N, #records FP, #records arm) of both arms (paired latents). No other metric is tested.
Replaces the paired t over overlapping subsamples / over the 6 K values, whose samples are not independent (issue 1).

Usage (repo root, CPU): python scripts/div_perm.py <task> <view> <N> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from dptig.metrics import ClipExtractor, ResNetExtractor, load_images, paper_view

task, view, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ARMS, LEVELS, NPERM, NBOOT = ["dp", "dp-paper", "dp-paper-nostyle"], ("low", "high"), 10_000, 2_000
M = Path("runs/main")


def recs(arm):
    f = M / f"{task}_{arm}" / "records.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def files(arm, n, lv):
    d = M / f"{task}_{arm}"
    return [d / r["runs"][lv]["file"] for r in recs(arm)[:n] if r["seed_ok"] and lv in r["runs"] and r["runs"][lv]["failed"]]


def unit(z):
    z = np.asarray(z, dtype=np.float64)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def D_from_sim(S, idx):
    s = S[np.ix_(idx, idx)]
    n = len(idx)
    return (n * n - s.sum()) / (n * (n - 1))   # mean over i != j of (1 - cos); diagonal of s is 1


def test(za, zf, rng):
    """za, zf: unit features of arm and FP. Returns D_fp, D_arm, diff, permutation p, bootstrap CI."""
    Z = np.vstack([za, zf])
    S = Z @ Z.T
    na, nf = len(za), len(zf)
    ia, i_f = np.arange(na), np.arange(na, na + nf)
    diff = D_from_sim(S, ia) - D_from_sim(S, i_f)
    perm = np.empty(NPERM)
    for k in range(NPERM):
        p = rng.permutation(na + nf)
        perm[k] = D_from_sim(S, p[:na]) - D_from_sim(S, p[na:])
    pval = (1 + np.sum(np.abs(perm) >= abs(diff) - 1e-12)) / (NPERM + 1)
    boot = np.empty(NBOOT)
    def D_boot(idx):   # resampled ids: pairs of the same original image (incl. duplicates) are excluded
        s, same = S[np.ix_(idx, idx)], idx[:, None] == idx[None, :]
        return float(np.mean(1.0 - s[~same]))
    for k in range(NBOOT):
        boot[k] = D_boot(rng.choice(ia, na)) - D_boot(rng.choice(i_f, nf))
    return dict(D_fp=D_from_sim(S, i_f), D_arm=D_from_sim(S, ia), diff=float(diff), p=float(pval),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], n_arm=na, n_fp=nf)


nfp = len(recs("fp"))
res = dict(task=task, N=N, view=view, prereg="WORKLOG (23)", results={})
exts = (("resnet50", ResNetExtractor("cpu", "IMAGENET1K_V2")), ("clip", ClipExtractor("cpu")))
cache = {}
for arm in ARMS:
    n = min(N, nfp, len(recs(arm)))
    if n < 10:
        continue
    for lv in LEVELS:
        fa, ff = files(arm, n, lv), files("fp", n, lv)
        if min(len(fa), len(ff)) < 3:
            continue
        for en, ext in exts:
            def feats(arm_, fl):
                key = (en, arm_, lv, len(fl))
                if key not in cache:
                    cache[key] = unit(ext([paper_view(i, view) for i in load_images(fl)]))
                return cache[key]
            r = test(feats(arm, fa), feats("fp", ff), np.random.default_rng(0))
            r["n_seeds"] = n
            res["results"][f"{arm}|{lv}|{en}"] = r
            print(f"{task:15s} {arm:17s} {lv:4s} {en:8s} n={n} FP {r['D_fp']:.4f} arm {r['D_arm']:.4f} "
                  f"diff {r['diff']:+.4f} CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p']:.4f}", flush=True)
json.dump(res, open(out, "w"), indent=1)
