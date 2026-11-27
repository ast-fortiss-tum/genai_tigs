"""RQ4 / RQ5 recomputed from the raw MTurk answers in Maryam's repo (WORKLOG (23)). CPU, seconds.
Procedure as in the paper (Sec. 5): 2 assessors per image; valid = both pick an in-domain answer, invalid = both pick
"not a real-world object" / "Not a human face", disagreements excluded; preserved = both give the target label, over
the agreed-valid images. Pizza: workers failing the attention check (ACQ = cat image) are dropped.
Only pizza DP and CelebA DP+FP have raw answers in the repo (CIFAR-10 / teddy: one extra assignment each; pizza FP none).
Usage: python scripts/mturk_reanalysis.py [path/to/mturk-evaluation]"""
import collections, csv, sys
from pathlib import Path
from scipy.stats import fisher_exact

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "repos/maryam_src/genai_tigs_diversity/evaluation_results/mturk-evaluation")


def summarise(name, items):   # items: list of (answer1, answer2, is_invalid_fn, is_preserved_fn)
    valid = [a for a in items if not a[2](a[0]) and not a[2](a[1])]
    invalid = [a for a in items if a[2](a[0]) and a[2](a[1])]
    pres = sum(a[3](a[0]) and a[3](a[1]) for a in valid)
    n = len(valid) + len(invalid)
    print(f"{name:22s} images {len(items):3d}  valid {len(valid)}/{n} = {100*len(valid)/max(n,1):6.2f}%  "
          f"preserved {pres}/{len(valid)} = {100*pres/max(len(valid),1):6.2f}%")
    return pres, len(valid)


# pizza DP
ans = collections.defaultdict(list)
for x in csv.DictReader(open(D / "answer/Batch_inpi.csv")):
    imgs = {k.split("url")[1]: v.split("/")[-1] for k, v in x.items() if k.startswith("Input.image_url")}
    a = {i: (x.get(f"Answer.category_{i}", "") or x.get(f"Answer.category_{i}.label", "")).strip().lower() for i in imgs}
    if all(a[i] == "cat" for i, f in imgs.items() if f.startswith("vr_")):
        for i, f in imgs.items():
            if not f.startswith("vr_"):
                ans[f].append(a[i])
for lv in ("low", "high"):
    items = [(v[0], v[1], lambda s: s.startswith("not a real"), lambda s: s == "pizza")
             for f, v in ans.items() if f"_{lv}_" in f and len(v) == 2]
    summarise(f"pizza DP {lv}", items)
print("  (paper Table 2: DP low 100 / 85.33, high 96.47 / 85.37; FP raw answers not in the repo)")

# CelebA DP (sdllm) and FP (icst)
rows = list(csv.DictReader(open(D / "celeb_final_mturk.csv")))
for lv in ("low", "high"):
    res = {}
    for tool, arm in (("sdllm", "DP"), ("icst", "FP")):
        items = [(r["Worker_1_true_options"], r["Worker_2_true_options"], lambda s: "Not a human face" in s,
                  (lambda lab: (lambda s: lab in s))(r["Expected_Label"]))
                 for r in rows if r["TOOL"] == tool and r["Mut_extent"] == lv]
        res[arm] = summarise(f"CelebA {arm} {lv}", items)
    (pd, nd), (pf, nf) = res["DP"], res["FP"]
    print(f"  preserved DP vs FP {lv}: Fisher p = {fisher_exact([[pd, nd - pd], [pf, nf - pf]])[1]:.3f}")
print("  (paper Table 2: DP low 100 / 41.18, FP low 91.30 / 66.67; DP high 100 / 55.06, FP high 100 / 57.47)")
