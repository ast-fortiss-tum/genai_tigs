"""Fitness functions. All are minimised by the search; a negative value means a failure."""
from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def margin_fitness(scores: np.ndarray, expected: int, softmax: bool = False) -> np.ndarray:
    """score[expected] - max_{i != expected} score[i], per row of ``scores`` (N, K).

    softmax=False reproduces the legacy scripts (raw logits).
    softmax=True follows Eq. (3) of the paper literally.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if softmax:
        scores = _softmax(scores)
    exp = scores[:, expected].copy()
    other = scores.copy()
    other[:, expected] = -np.inf
    return exp - other.max(axis=1)


def target_logit_fitness(scores: np.ndarray, attribute: int) -> np.ndarray:
    """Eq. (4) (CelebA-HQ multi-label): the logit of the target attribute. <0 => predicted absent."""
    return np.asarray(scores, dtype=np.float64)[:, attribute].copy()


def make_fitness(kind: str, expected: int):
    """Return f(scores)->(N,) for the configured fitness kind."""
    if kind == "logit_margin":
        return lambda s: margin_fitness(s, expected, softmax=False)
    if kind == "softmax_margin":
        return lambda s: margin_fitness(s, expected, softmax=True)
    if kind == "target_logit":
        return lambda s: target_logit_fitness(s, expected)
    raise ValueError(f"unknown fitness kind: {kind}")
