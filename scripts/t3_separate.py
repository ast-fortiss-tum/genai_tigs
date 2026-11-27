"""Table-3 cluster diversity with each arm clustered ON ITS OWN (KMeans fit per arm), to check whether the joint
clustering (FP + arm pooled, as in the authors' script) inflates differences between visually distinct arms.
Usage (repo root, CPU only, safe while the GPU queue runs): python scripts/t3_separate.py <task> <view> <N> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from dptig import stats
from dptig.metrics import ClipExtractor, ResNetExtractor, cluster_diversity, load_images, paper_view

task, view, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
levels, seeds = ("low", "high"), range(10)
M = Path("runs/main")
_n = lambda a: len((M / f"{task}_{a}" / "records.jsonl").read_text().splitlines()) if (M / f"{task}_{a}" / "records.jsonl").exists() else 0
arms = ["fp"] + [a for a in ("dp", "dp-paper", "dp-paper-nostyle") if _n(a) >= N]   # only arms with all N paired seeds
files = {}
for a in arms:
    d = M / f"{task}_{a}"
    recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if l.strip()][:N]
    for lv in levels:
        files[(a, lv)] = [d / r["runs"][lv]["file"] for r in recs if r["seed_ok"] and r["runs"][lv]["failed"]]
imgs = {g: [paper_view(i, view) for i in load_images(v)] for g, v in files.items()}
res = dict(task=task, N=N, n={f"{a}_{lv}": len(v) for (a, lv), v in files.items()})
for name, ext, l2 in (("clip", ClipExtractor("cpu"), True), ("resnet50", ResNetExtractor("cpu", "IMAGENET1K_V2"), False)):
    feats = {}
    for g, im in imgs.items():
        z = ext(im).astype(np.float32)
        feats[g] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9) if l2 else z
    D = {a: {c: [] for c in list(levels) + ["overall"]} for a in arms}   # per arm, per cond: list over seeds of per-K lists
    for s in seeds:
        for a in arms:
            d = cluster_diversity({lv: feats[(a, lv)] for lv in levels}, combos={"overall": list(levels)}, seed=s)
            for c in D[a]:
                D[a][c].append(d[c])
    r = {}
    for arm in arms[1:]:
        for c in list(levels) + ["overall"]:
            rows = []
            for i in seeds:
                a, b = np.array(D["fp"][c][i]), np.array(D[arm][c][i])
                rows.append(dict(seed=i, D_fp=float(a.mean()), D_arm=float(b.mean()), **stats.paired_t(b, a)))
            r[f"{arm}_{c}"] = dict(D_fp=float(np.mean([x["D_fp"] for x in rows])),
                                   D_arm=float(np.mean([x["D_arm"] for x in rows])),
                                   arm_higher=sum(x["D_arm"] > x["D_fp"] for x in rows),
                                   sig=sum(x["p"] < .05 for x in rows),
                                   sig_higher=sum(x["p"] < .05 and x["D_arm"] > x["D_fp"] for x in rows), seeds=rows)
    res[name] = r
json.dump(res, open(out, "w"), indent=1)
for name in ("resnet50", "clip"):
    for k, v in res[name].items():
        print(f"{task:15s} {name:8s} {k:17s} FP {v['D_fp']:.2f} arm {v['D_arm']:.2f} higher {v['arm_higher']}/10 "
              f"sig {v['sig']}/10 sig&higher {v['sig_higher']}/10", flush=True)
