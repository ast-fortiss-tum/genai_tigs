"""Replacement for ``tgate.tgate_utils.register_tgate_forward`` (tgate 0.1.1) without per-step closures.

Upstream re-wraps every ``Attention.forward`` in a fresh closure over ``self`` at *every* denoising step. That creates
thousands of module<->closure reference cycles per image, and on CUDA runs we intermittently got
``NameError: free variable 'self' referenced before assignment`` from inside that closure (an emptied closure cell),
which killed the run (WORKLOG 2026-11-17 (3)).

Here each Attention module gets one persistent bound-method forward, installed once; the per-step values
(gate_step, inference_num_per_image, lcm, cur_step) are stored as plain attributes. The attention computation itself
is upstream's ``tgate_processor``, and the step/cache bookkeeping mirrors upstream line by line, so outputs are
unchanged (checked bit-identical, see tests/test_tgate_patch_gpu.py).
"""
from __future__ import annotations

import inspect
import types

_ATTR = "_dptig_tgate"


def _forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, gate_step=None,
             inference_num_per_image=None, lcm=None, cur_step=None, **cross_attention_kwargs):
    from tgate.tgate_utils import tgate_processor

    st = getattr(self, _ATTR)
    gate_step = st["gate_step"] if gate_step is None else gate_step
    inference_num_per_image = st["inference_num_per_image"] if inference_num_per_image is None else inference_num_per_image
    lcm = st["lcm"] if lcm is None else lcm
    cur_step = st["cur_step"] if cur_step is None else cur_step

    # upstream bookkeeping (tgate_utils.py, forward)
    if not hasattr(self, "cur_step"):
        self.cur_step = 1
    else:
        if cur_step is not None:
            self.cur_step = cur_step
        else:
            self.cur_step = (self.cur_step + 1) % inference_num_per_image
    if not hasattr(self, "cache"):
        self.cache = None
    attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

    hidden_states, cache = tgate_processor(
        self, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
        gate_step=gate_step, cur_step=self.cur_step, cache=self.cache, lcm=lcm, **cross_attention_kwargs)
    if gate_step == self.cur_step:
        self.cache = cache
    return hidden_states


def register_tgate_forward(model, filter_name: str = "Attention", gate_step: int = 10,
                           inference_num_per_image: int = 25, lcm: bool = False, cur_step: int = None, **kwargs):
    """Drop-in for the upstream function: same arguments, same return value (number of layers)."""
    count = 0
    for m in model.modules():
        if m.__class__.__name__ != filter_name:
            continue
        m.__dict__[_ATTR] = dict(gate_step=gate_step, inference_num_per_image=inference_num_per_image,
                                 lcm=lcm, cur_step=cur_step)
        fwd = m.__dict__.get("forward")
        if not (isinstance(fwd, types.MethodType) and fwd.__func__ is _forward):
            m.forward = types.MethodType(_forward, m)
        count += 1
    return count


def apply() -> None:
    """Make tgate's SD+DeepCache pipeline use the closure-free registration."""
    import tgate.SD_DeepCache as sd_dc
    sd_dc.register_tgate_forward = register_tgate_forward
