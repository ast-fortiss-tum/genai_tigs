"""Statistical tests used in the paper (Sec. 5.x): Fisher exact (binary), Mann-Whitney U + Cohen's d
(iterations), paired t-test + Cohen's d (diversity), Pearson/Spearman (CLIP vs ResNet agreement)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d(a, b) -> float:
    """Independent-samples Cohen's d (pooled SD), a minus b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def cohens_d_paired(a, b) -> float:
    """Paired Cohen's d_z: mean difference / SD of differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def fisher(k1: int, n1: int, k2: int, n2: int) -> dict:
    """2x2 Fisher exact test: k successes out of n for group 1 (DP) vs group 2 (FP)."""
    odds, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return dict(odds_ratio=float(odds), p=float(p))


def mann_whitney(a, b) -> dict:
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(U=float(u), p=float(p), d=cohens_d(a, b))


def paired_t(a, b) -> dict:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size < 2 or np.ptp(diff) < 1e-12:      # identical differences: t undefined (was ~1e15 from rounding)
        return dict(t=float("nan"), p=float("nan"), d=float("nan"), mean_a=float(np.mean(a)),
                    mean_b=float(np.mean(b)), degenerate=True)
    t, p = stats.ttest_rel(a, b)
    return dict(t=float(t), p=float(p), d=cohens_d_paired(a, b),
                mean_a=float(np.mean(a)), mean_b=float(np.mean(b)))


def correlations(x, y) -> dict:
    return dict(pearson=float(stats.pearsonr(x, y)[0]), spearman=float(stats.spearmanr(x, y)[0]))
