"""Stable Diffusion v1.5 + LoRA (+ TGate/DeepCache) wrapper. Heavy imports are lazy."""
from __future__ import annotations

from typing import List

import torch
from PIL import Image

from .config import resolve_weight


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SDGenerator:
    """G(latent, prompt) -> image. Latents are (N, 4, H/8, W/8) and are the search space."""

    def __init__(self, sd_cfg: dict, device: torch.device):
        from diffusers import StableDiffusionPipeline
        from diffusers.schedulers import DPMSolverMultistepScheduler

        self.cfg, self.device = sd_cfg, device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.size = int(sd_cfg.get("size", 512))
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_cfg["base_model"], torch_dtype=self.dtype, safety_checker=None,
            **({"variant": "fp16"} if self.dtype == torch.float16 else {})).to(device)
        pipe.load_lora_weights(str(resolve_weight(sd_cfg["lora"])))
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.use_tgate = bool(sd_cfg.get("tgate", True))
        if self.use_tgate:
            from tgate import TgateSDDeepCacheLoader
            pipe = TgateSDDeepCacheLoader(pipe, cache_interval=sd_cfg.get("cache_interval", 3),
                                          cache_branch_id=0).to(device)
        self.pipe = pipe
        self.gen = torch.Generator(device=device)

    def latent(self, seed: int) -> torch.Tensor:
        g = self.gen.manual_seed(seed)
        shape = (1, self.pipe.unet.config.in_channels, self.size // 8, self.size // 8)
        return torch.randn(shape, generator=g, device=self.device).to(self.dtype)

    @torch.inference_mode()
    def _call(self, prompt, latents, guidance) -> List[Image.Image]:
        kw = dict(guidance_scale=guidance, num_inference_steps=self.cfg.get("steps", 25), latents=latents.to(self.dtype))
        if self.use_tgate:
            return self.pipe.tgate(prompt, gate_step=self.cfg.get("gate_step", 10), **kw)["images"]
        return self.pipe(prompt, **kw).images

    def seed_image(self, prompt: str, latent: torch.Tensor) -> Image.Image:
        return self._call(prompt, latent, self.cfg.get("seed_guidance", 3.5))[0]

    def batch(self, prompt: str, latents: torch.Tensor) -> List[Image.Image]:
        return self._call([prompt] * latents.shape[0], latents, self.cfg.get("mutation_guidance", 1.4))
