"""Machine-written results snapshot for docs/auto_results.md (tracked; runs/ is gitignored). Called by
scripts/queue_weekend.sh after each phase. RQ1-RQ3 are computed directly from records.jsonl (paired: first n seeds
common to FP and the arm); RQ6 = the pre-registered permutation test (scripts/div_perm.py, WORKLOG (23)), Holm-corrected
per arm over all task x level tests present. Usage: python scripts/auto_summary.py > docs/auto_results.md"""
import glob, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats as st

M = Path("runs/main")
TASKS = ("cifar10", "imagenet_pizza", "imagenet_teddy", "celebahq")
ARMS = ("fp", "dp", "dp-paper", "dp-paper-nostyle")


def recs(t, a):
    f = M / f"{t}_{a}" / "records.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def holm(ps):
    order, out, run = np.argsort(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


print(f"# Auto-generated results ({datetime.now():%Y-%m-%d %H:%M})\n")
print("Written by `scripts/auto_summary.py` from `runs/main` (machine snapshot; interpretation is in WORKLOG.md).\n")
print("## Queue status (last lines of runs/main/status.log)\n```")
print("\n".join((M / "status.log").read_text().splitlines()[-15:]))
print("```\n\n## Seeds per arm\n")
print("| Task | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
for t in TASKS:
    print(f"| {t} | " + " | ".join(str(len(recs(t, a))) for a in ARMS) + " |")

print("\n## RQ1-RQ3 (paired: first n seeds common to FP and arm)\n")
print("validity = seed_ok %, Fisher p; eff = % of valid seeds with a failure (low level), Fisher p; "
      "iters = mean iterations to first failure (low / high), Mann-Whitney p.\n")
print("| Task | Arm | n | validity FP / arm (p) | eff low FP / arm (p) | iters low FP / arm (p) | iters high FP / arm (p) |")
print("|---|---|---|---|---|---|---|")
for t in TASKS:
    F = recs(t, "fp")
    for a in ARMS[1:]:
        A = recs(t, a)
        n = min(len(F), len(A))
        if n < 5:
            continue
        f, x = F[:n], A[:n]
        vf, va = sum(r["seed_ok"] for r in f), sum(r["seed_ok"] for r in x)
        row = [t, a, str(n), f"{100*vf/n:.0f} / {100*va/n:.0f} ({st.fisher_exact([[va, n-va], [vf, n-vf]])[1]:.2g})"]
        for lv, what in (("low", "eff"), ("low", "it"), ("high", "it")):
            gf = [r["runs"][lv] for r in f if r["seed_ok"] and lv in r["runs"]]
            ga = [r["runs"][lv] for r in x if r["seed_ok"] and lv in r["runs"]]
            if what == "eff":
                kf, ka = sum(g["failed"] for g in gf), sum(g["failed"] for g in ga)
                p = st.fisher_exact([[ka, len(ga)-ka], [kf, len(gf)-kf]])[1]
                row.append(f"{100*kf/max(1,len(gf)):.0f} / {100*ka/max(1,len(ga)):.0f} ({p:.2g})")
            else:
                i_f = [g["iterations"] for g in gf if g["failed"]]
                i_a = [g["iterations"] for g in ga if g["failed"]]
                p = st.mannwhitneyu(i_a, i_f).pvalue if len(i_f) > 1 and len(i_a) > 1 else float("nan")
                row.append(f"{np.mean(i_f):.0f} / {np.mean(i_a):.0f} ({p:.2g})" if i_f and i_a else "-")
        print("| " + " | ".join(row) + " |")

print("\n## RQ6 diversity, pre-registered test (WORKLOG (23))\n")
print("Eq. 6 on the classifier input, failure images; diff = D_arm - D_FP; permutation p (10k), 95% bootstrap CI, "
      "Holm per arm and feature space over the task x level tests listed. **Primary = ResNet-50.**\n")
rows = []
for t in TASKS:
    fs = sorted(glob.glob(str(M / "analysis" / f"{t}_divperm_n*.json")), key=lambda s: int(s.split("_n")[-1].split(".")[0]))
    if fs:
        for k, r in json.load(open(fs[-1]))["results"].items():
            a, lv, en = k.split("|")
            rows.append(dict(task=t, arm=a, lv=lv, en=en, **r))
for a in ARMS[1:]:
    for en in ("resnet50", "clip"):
        R = [r for r in rows if r["arm"] == a and r["en"] == en]
        for r, ph in zip(R, holm([r["p"] for r in R])):
            r["p_holm"] = ph
print("| Arm | Feature | Task | Level | n seeds | D FP | D arm | diff [95% CI] | p | p Holm |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in ARMS[1:]:
    for en in ("resnet50", "clip"):
        for r in [r for r in rows if r["arm"] == a and r["en"] == en]:
            sig = " **" if r["p_holm"] < .05 else ""
            print(f"| {a} | {en} | {r['task']} | {r['lv']} | {r['n_seeds']} | {r['D_fp']:.3f} | {r['D_arm']:.3f} | "
                  f"{r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.3g} | {r['p_holm']:.3g}{sig} |")
