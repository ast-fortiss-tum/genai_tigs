"""Image-diversity metrics (paper Sec. 5.4, Eq. 6): mean pairwise cosine distance of CLIP ViT-B/32 and
ResNet-50 (penultimate layer) features. Feature extractors load lazily; ``diversity`` itself is pure numpy."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


def diversity(feats: np.ndarray) -> float:
    """Eq. 6: D = 2/(N(N-1)) * sum_{i<j} (1 - cos(z_i, z_j)). Needs N >= 2."""
    z = np.asarray(feats, dtype=np.float64)
    n = len(z)
    if n < 2:
        raise ValueError("diversity needs at least 2 images")
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    sim = z @ z.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def subsample_scores(feats: np.ndarray, k: int, m: int, seed: int = 0) -> np.ndarray:
    """k diversity scores, each over m images drawn without replacement (RNG seeded -> the same index
    sets are reused across methods, so scores are pairable for the paired t-test).
    ASSUMPTION: the paper does not say what one 'score' unit is; this gives a distribution per condition."""
    rng = np.random.default_rng(seed)
    m = min(m, len(feats))
    return np.array([diversity(feats[rng.choice(len(feats), m, replace=False)]) for _ in range(k)])


class _Extractor:
    name = ""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, images: Sequence, batch: int = 32) -> np.ndarray:
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch):
            out.append(self._encode(images[i:i + batch]))
        return np.concatenate(out)


class ClipExtractor(_Extractor):
    name = "clip"

    def __init__(self, device="cpu", model_id="openai/clip-vit-base-patch32"):
        super().__init__(device)
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def _encode(self, imgs):
        import torch
        x = self.proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=x).cpu().numpy()


class ResNetExtractor(_Extractor):
    name = "resnet50"

    def __init__(self, device="cpu"):
        super().__init__(device)
        import torch
        import torchvision
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet50(weights=w)
        m.fc = torch.nn.Identity()           # penultimate (2048-d) features
        self.model = m.to(device).eval()
        self.tf = w.transforms()

    def _encode(self, imgs):
        import torch
        x = torch.stack([self.tf(i.convert("RGB")) for i in imgs]).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()


def load_images(paths: Sequence[Path]):
    from PIL import Image
    return [Image.open(p) for p in paths]
