"""Markdown summary of run directories next to the paper's Table 2 (FP/DP rows).

    python -m dptig.summarize runs/main            # all <task>_<arm> dirs under runs/main
Per task x arm x level: seeds (valid/total), failures (found/searched), mean and median iterations over failing
searches, and how many failures happened at iteration 1 (guidance-switch effect, WORKLOG 2026-11-17 (2)).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# paper Table 2: (% seeds, low: % misclass, low iters, high: % misclass, high iters)
PAPER = {
    ("cifar10", "fp"): (87, 89.66, 85.63, 100, 14.18), ("cifar10", "dp"): (84, 92.86, 74.73, 100, 9.89),
    ("imagenet_pizza", "fp"): (73, 97.26, 83.60, 100, 12.53), ("imagenet_pizza", "dp"): (85, 88.24, 123.28, 100, 20.14),
    ("imagenet_teddy", "fp"): (87, 98.85, 48.45, 100, 6.63), ("imagenet_teddy", "dp"): (92, 98.91, 40.79, 100, 4.88),
    ("celebahq", "fp"): (94, 24.73, 18.22, 91.47, 56.08), ("celebahq", "dp"): (93, 18.28, 8.82, 95.70, 45.26),
}
ARMS = ("fp", "dp", "dp-template")


def summarize_dir(d: Path) -> dict:
    rs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if l.strip()]
    out = dict(n=len(rs), valid=sum(r["seed_ok"] for r in rs))
    for lv in ("low", "high"):
        runs = [r["runs"][lv] for r in rs if r["seed_ok"] and lv in r["runs"]]
        it = [x["iterations"] for x in runs if x["failed"]]
        out[lv] = dict(searched=len(runs), failed=len(it), mean=st.mean(it) if it else None,
                       median=st.median(it) if it else None, it1=sum(i == 1 for i in it))
    return out


def table(root: Path) -> str:
    rows = ["| task | arm | seeds valid | low: fail | low: mean (median) it | low: it=1 | high: fail | "
            "high: mean (median) it | high: it=1 | paper %seeds / low fail%, it / high fail%, it |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    tasks = sorted({p.name.rsplit("_", 1)[0] if not p.name.endswith("dp-template") else p.name[:-len("_dp-template")]
                    for p in root.iterdir() if (p / "records.jsonl").exists()})
    for t in tasks:
        for a in ARMS:
            d = root / f"{t}_{a}"
            if not (d / "records.jsonl").exists():
                continue
            s = summarize_dir(d)
            cell = lambda lv: (f"{s[lv]['failed']}/{s[lv]['searched']}",
                               "-" if s[lv]["mean"] is None else f"{s[lv]['mean']:.1f} ({s[lv]['median']:.0f})",
                               str(s[lv]["it1"]))
            p = PAPER.get((t, a))
            ptxt = f"{p[0]} / {p[1]}, {p[2]} / {p[3]}, {p[4]}" if p else ""
            rows.append(f"| {t} | {a} | {s['valid']}/{s['n']} | " + " | ".join(cell("low")) + " | "
                        + " | ".join(cell("high")) + f" | {ptxt} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    print(table(ap.parse_args(argv).root))


if __name__ == "__main__":
    main()
