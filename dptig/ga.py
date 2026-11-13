"""Evolutionary latent-space search (paper Algorithm 1, lines 8-33).

Pure torch/numpy: it never touches the diffusion model or the classifier. The caller
supplies ``evaluate(latents) -> (fitness, payload)``, so the same search drives FP-TIG,
DP-TIG, and (later) any other generator. Behaviour follows the legacy
``tig_sd_*.py`` scripts, with the deviations documented inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class GAParams:
    delta: float                       # delta_init, the perturbation step
    bounds: Tuple[float, float]        # (minBound, maxBound) the latents are clamped to
    pop_size: int = 25
    best_left: int = 10                # tshdBest
    max_iter: int = 250                # N


@dataclass
class SearchResult:
    failed: bool                       # True iff a fitness < 0 individual was found
    iterations: int                    # iterations used (max_iter if no failure)
    best_fitness: float
    best_latent: torch.Tensor
    best_payload: Any                  # whatever ``evaluate`` returned for the best individual


Evaluate = Callable[[torch.Tensor], Tuple[Sequence[float], Sequence[Any]]]


def _noise(shape, dtype, device, gen):
    # sampled in fp32: fp16 randn is unsupported on CPU, and this keeps CPU/GPU streams aligned
    return torch.randn(shape, generator=gen, device=device).to(dtype)


def run_search(seed_latent: torch.Tensor, evaluate: Evaluate, params: GAParams,
               seed: int = 0) -> SearchResult:
    """seed_latent: (1, C, H, W). evaluate: (N, C, H, W) -> (fitness[N], payload[N])."""
    z0 = seed_latent
    dev, dt = z0.device, z0.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    lo, hi = params.bounds
    n_ch = z0.shape[1]

    # initial population: slight variations of the seed latent (Alg.1 lines 8-9)
    pop: List[torch.Tensor] = [z0 + params.delta * _noise(z0.shape, dt, dev, gen)
                               for _ in range(params.pop_size)]

    delta = params.delta
    prev_best = np.inf
    best: Optional[Tuple[float, torch.Tensor, Any]] = None

    for it in range(params.max_iter):
        fit, payload = evaluate(torch.cat(pop, dim=0))
        fit = np.asarray(fit, dtype=np.float64)
        i_min = int(np.argmin(fit))
        now_best = float(fit[i_min])
        if best is None or now_best < best[0]:
            best = (now_best, pop[i_min], payload[i_min])

        if now_best < 0:                                   # misclassification found
            return SearchResult(True, it + 1, best[0], best[1], best[2])

        # adaptive mutation extent: widen when the best fitness stalled, else reset
        delta = delta * 2 if now_best == prev_best else params.delta
        prev_best = now_best

        parents = [pop[i] for i in np.argsort(fit)[: params.best_left]]
        offspring: List[torch.Tensor] = []
        for _ in range(params.pop_size - len(parents)):
            a, b = rng.choice(len(parents), size=2, replace=False)
            split = int(rng.integers(0, n_ch))             # single-point crossover along channels
            child = torch.cat([parents[a][:, :split], parents[b][:, split:]], dim=1)
            # mutate only the positions that differ from the seed latent (legacy behaviour)
            diffs = (child != z0).to(dt)
            child = child + delta * _noise(child.shape, dt, dev, gen) * diffs
            offspring.append(child)

        pop = [torch.clamp(z, min=lo, max=hi) for z in parents + offspring]

    assert best is not None
    return SearchResult(False, params.max_iter, best[0], best[1], best[2])
