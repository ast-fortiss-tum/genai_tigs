"""Classifiers under test: weights + the exact preprocessing used by the legacy scripts."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import resolve_weight


# --- CIFAR-10 VGGNet (copied from experiments/cifar10_experiments/cifar10_classifier/model_cifar10.py
#     so the new package does not depend on the legacy, non-package tree) ------------------------------
class _VGGBlock(nn.Module):
    def __init__(self, inp, outp, lnum):
        super().__init__()
        self.first_lyr = nn.Sequential(nn.Conv2d(inp, outp, 3, padding=1), nn.BatchNorm2d(outp), nn.ReLU(inplace=True))
        rest = []
        for _ in range(1, lnum):
            rest += [nn.Conv2d(outp, outp, 3, padding=1), nn.BatchNorm2d(outp), nn.ReLU(inplace=True)]
        self.next_lyrs = nn.Sequential(*rest)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.maxpool(self.next_lyrs(self.first_lyr(x)))


class _VGGClassifier(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(inp, 4096), nn.ReLU(True), nn.Dropout(),
                                        nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 10))

    def forward(self, x):
        return self.classifier(x.view(x.size(0), -1))


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = _VGGBlock(3, 64, 2)
        self.layer2 = _VGGBlock(64, 128, 2)
        self.layer3 = _VGGBlock(128, 256, 4)
        self.layer4 = _VGGBlock(256, 512, 4)
        self.layer5 = _VGGBlock(512, 512, 4)
        self.vggc = _VGGClassifier(512)

    def forward(self, x):
        for l in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5):
            x = l(x)
        return self.vggc(x)


class Classifier:
    """Wraps a model with its preprocessing: ``scores(images)`` -> (N, K) raw outputs (logits)."""

    def __init__(self, model: nn.Module, preprocess, device):
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.device = device

    @torch.no_grad()
    def scores(self, images: List[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        return self.model(x).float().cpu().numpy()


def _nearest_resize(a: np.ndarray, size: int) -> np.ndarray:
    """cv2.resize(..., INTER_NEAREST) without cv2: src = floor(dst * src_size / dst_size) (no half-pixel
    centre, unlike PIL's NEAREST), so results match the legacy scripts exactly."""
    h, w = a.shape[:2]
    ys = np.minimum((np.arange(size) * (h / size)).astype(int), h - 1)
    xs = np.minimum((np.arange(size) * (w / size)).astype(int), w - 1)
    return a[ys][:, xs]


def _cifar10_preprocess():
    to_tensor = transforms.ToTensor()

    def f(img: Image.Image) -> torch.Tensor:
        a = np.array(img)
        if a.shape[0] != 32 or a.shape[1] != 32:
            a = _nearest_resize(a, 32)
        return to_tensor(Image.fromarray(a))
    return f


def _imagenet_preprocess():
    return transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_classifier(cfg: dict, device) -> Classifier:
    kind = cfg["kind"]
    if kind == "cifar10_vgg":
        model = VGGNet()
        model.load_state_dict(torch.load(resolve_weight(cfg["weights"]), map_location=device))
        return Classifier(model, _cifar10_preprocess(), device)
    if kind == "imagenet_vgg19bn":
        import torchvision
        try:
            model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)
        except AttributeError:                      # older torchvision
            model = torchvision.models.vgg19_bn(pretrained=True)
        return Classifier(model, _imagenet_preprocess(), device)
    if kind == "celeba_resnet50":                   # our replication, see dptig/celeba_classifier.py
        from .celeba_classifier import build_model, preprocess
        model = build_model()
        model.load_state_dict(torch.load(resolve_weight(cfg["weights"]), map_location=device))
        return Classifier(model, preprocess(), device)
    raise ValueError(f"unknown classifier kind: {kind}")
