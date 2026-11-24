"""Image-diversity metrics: Eq. 6 (paper Sec. 5.4) and the cluster score behind Table 3 (``cluster_diversity``): mean pairwise cosine distance of CLIP ViT-B/32 and
ResNet-50 (penultimate layer) features. Feature extractors load lazily; ``diversity`` itself is pure numpy."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


K_LIST = (2, 4, 6, 8, 10, 12)


def _cov_ent(labels: np.ndarray, k: int) -> float:
    c = np.bincount(labels, minlength=k).astype(float)
    p = c[c > 0] / c.sum()
    return float((c > 0).sum() / k * (-(p * np.log2(p)).sum() / np.log2(k)))


def cluster_diversity(groups: Dict[str, np.ndarray], combos: Optional[Dict[str, Sequence[str]]] = None,
                      k_list: Sequence[int] = K_LIST, seed: int = 0) -> Dict[str, List[float]]:
    """Table-3 diversity as computed by the authors (Maryam's diversity/compute_diversity_*.py): KMeans(K, n_init=10,
    random_state=seed) on the union of all groups' features; per group (and per union of groups in ``combos``)
    D_K = coverage (occupied clusters / K) * Shannon entropy of the cluster histogram / log2 K. One value per K.
    Features are used as given (the authors L2-normalise CLIP, not ResNet)."""
    from sklearn.cluster import KMeans
    names = list(groups)
    z = np.vstack([groups[g] for g in names])
    idx = dict(zip(names, np.split(np.arange(len(z)), np.cumsum([len(groups[g]) for g in names])[:-1])))
    combos = combos or {}
    out: Dict[str, List[float]] = {g: [] for g in list(names) + list(combos)}
    for k in k_list:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z).labels_
        for g in names:
            out[g].append(_cov_ent(lab[idx[g]], k))
        for c, members in combos.items():
            out[c].append(_cov_ent(np.concatenate([lab[idx[m]] for m in members]), k))
    return out


def paper_view(img, view: str):
    """The image the authors embedded for Table 3 = what their scripts saved: the classifier input.
    cifar10: 32x32 nearest; imagenet: resize 256x256 + centre crop 224 (they also min-max rescaled, not reproduced);
    celeba: 224x224."""
    from PIL import Image
    img = img.convert("RGB")
    if view == "cifar10":
        return img.resize((32, 32), Image.NEAREST)
    if view == "imagenet":
        img = img.resize((256, 256), Image.BILINEAR)
        return img.crop((16, 16, 240, 240))
    if view == "celeba":
        return img.resize((224, 224), Image.BILINEAR)
    return img


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

    def __init__(self, device="cpu", weights: str = "IMAGENET1K_V1"):
        """weights: IMAGENET1K_V1 (our Eq. 6 runs) or IMAGENET1K_V2 + Resize 256/CenterCrop 224 (authors' Table 3)."""
        super().__init__(device)
        import torch
        import torchvision
        w = getattr(torchvision.models.ResNet50_Weights, weights)
        m = torchvision.models.resnet50(weights=w)
        m.fc = torch.nn.Identity()           # penultimate (2048-d) features
        self.model = m.to(device).eval()
        self.tf = w.transforms()
        if weights == "IMAGENET1K_V2":       # authors' transform (V2's default resizes to 232)
            T = torchvision.transforms
            self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _encode(self, imgs):
        import torch
        x = torch.stack([self.tf(i.convert("RGB")) for i in imgs]).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()


def load_images(paths: Sequence[Path]):
    from PIL import Image
    return [Image.open(p) for p in paths]
