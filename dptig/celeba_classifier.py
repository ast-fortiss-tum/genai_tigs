"""CelebA 40-attribute classifier (paper Sec. 5.2): ResNet-50 pretrained on ImageNet, fc -> 40 logits, fine-tuned on
CelebA for multi-label attribute classification. Our replication; the authors' weights were not available.

Prepare: python -m dptig.celeba_classifier prepare --src ~/data/celeba_flwr --dst ~/data/celeba_packed
Train:   python -m dptig.celeba_classifier train --celeba ~/data/celeba_packed --hq ~/data/celebamask_hq/CelebAMask-HQ \
             --out weights/classifiers/celeba
Data:    flwrlabs/celeba (aligned 178x218 PNGs + the 40 attributes, split as specified by the CelebA authors:
         162770/19867/19962) and the official CelebAMask-HQ zip (CelebA-HQ images + attribute annotations), the latter
         only for evaluation on the HQ domain. (huggan/CelebA-faces-with-attributes was tried first: 11 of its 132
         parquet shards are truncated on the hub.)
Split:   official CelebA partition. CelebA-HQ images are evaluated only if their original CelebA image (by file number)
         is in the test partition (> 182637), so there is no train/test leakage.

Choices (paper silent): torchvision IMAGENET1K_V1 weights; Adam lr 1e-4, cosine, 5 epochs, batch 128, AMP;
train aug = random resized crop (scale 0.6-1) of the centred 178x178 square + horizontal flip.
At test time (and on generated 512x512 images) the whole square image is resized to 224 (``preprocess``).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ATTRS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
         'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
         'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
         'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
         'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
         'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']   # official CelebA order
TARGETS = {"Smiling": 31, "Bangs": 5, "Eyeglasses": 15}
TRAIN_MAX, VAL_MAX = 162770, 182637
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(pretrained: bool = False) -> nn.Module:
    import torchvision
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet50(weights=w)
    m.fc = nn.Linear(m.fc.in_features, len(ATTRS))
    return m


def preprocess():
    """Inference preprocessing for square face images (generated 512x512, CelebA-HQ)."""
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])


def _square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


# --- data -----------------------------------------------------------------------------------------------------
class _Packed(torch.utils.data.Dataset):
    """JPEG bytes packed in one uint8 buffer (fork-friendly: no per-item Python objects shared with workers)."""

    def __init__(self, buf, offs, labels, tf):
        self.buf, self.offs, self.labels, self.tf = buf, offs, labels, tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.buf[self.offs[i]:self.offs[i + 1]].tobytes())).convert("RGB")
        return self.tf(img), self.labels[i]


def _pack(blobs, labels):
    offs = np.zeros(len(blobs) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([len(b) for b in blobs])
    buf = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    return buf, offs, torch.tensor(np.asarray(labels), dtype=torch.float32)


_SPLITS = {"train": "train", "valid": "val", "test": "test"}     # flwrlabs file prefix -> our name
_EXPECTED = {"train": 162770, "val": 19867, "test": 19962}


def prepare(src: Path, dst: Path):
    """flwrlabs/celeba parquet -> per split: <split>.bin (concatenated image bytes), <split>.npz (offsets, labels).
    Streams shard by shard, so RAM stays small; the .bin files are read back with np.memmap."""
    import pyarrow.parquet as pq
    dst.mkdir(parents=True, exist_ok=True)
    for prefix, name in _SPLITS.items():
        files = sorted(glob.glob(str(src / "img_align+identity+attr" / f"{prefix}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no {prefix} shards under {src}")
        offs, labels, pos = [0], [], 0
        with open(dst / f"{name}.bin", "wb") as fb:
            for f in files:
                pf = pq.ParquetFile(f)                       # raises on a truncated shard
                for batch in pf.iter_batches(batch_size=2048, columns=["image"] + ATTRS):
                    d = batch.to_pydict()
                    for k, im in enumerate(d["image"]):
                        fb.write(im["bytes"])
                        pos += len(im["bytes"])
                        offs.append(pos)
                        labels.append([1.0 if d[a][k] else 0.0 for a in ATTRS])
        n = len(labels)
        assert n == _EXPECTED[name], f"{name}: {n} images, expected {_EXPECTED[name]}"
        np.savez(dst / f"{name}.npz", offs=np.asarray(offs, dtype=np.int64), labels=np.asarray(labels, np.float32))
        print(f"{name}: {n} images, {pos / 1e9:.2f} GB", flush=True)


def load_celeba(root: Path):
    out = {}
    for name in _EXPECTED:
        z = np.load(Path(root) / f"{name}.npz")
        buf = np.memmap(Path(root) / f"{name}.bin", dtype=np.uint8, mode="r")
        out[name] = (buf, z["offs"], torch.tensor(z["labels"]))
    return out


def load_hq_test(hq_root: Path):
    """CelebA-HQ images whose source CelebA image is in the official test partition."""
    hq_root = Path(hq_root)
    lines = (hq_root / "CelebAMask-HQ-attribute-anno.txt").read_text().splitlines()
    names = lines[1].split()
    assert names == ATTRS, "attribute order differs from CelebA"
    mapping = {}
    for l in (hq_root / "CelebA-HQ-to-CelebA-mapping.txt").read_text().splitlines()[1:]:
        idx, orig_idx, orig_file = l.split()
        mapping[f"{idx}.jpg"] = int(orig_file.split(".")[0])
    blobs, labels = [], []
    for l in lines[2:]:
        p = l.split()
        if mapping[p[0]] <= VAL_MAX:
            continue
        img = Image.open(hq_root / "CelebA-HQ-img" / p[0]).convert("RGB").resize((224, 224), Image.BICUBIC)
        b = io.BytesIO()
        img.save(b, format="PNG")
        blobs.append(b.getvalue())
        labels.append([1.0 if v == "1" else 0.0 for v in p[1:]])
    return _pack(blobs, labels)


# --- train / eval ---------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = torch.zeros(len(ATTRS), dtype=torch.float64)
    n = 0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True))
        correct += ((z.float().cpu() > 0).float() == y).sum(0).double()
        n += len(y)
    acc = (correct / max(n, 1)).numpy()
    return dict(mean_acc=float(acc.mean()), n=n, per_attr={a: float(acc[i]) for i, a in enumerate(ATTRS)},
                targets={a: float(acc[i]) for a, i in TARGETS.items()})


def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_celeba(Path(a.celeba))
    if a.limit:
        data = {s: _pack([v[0][v[1][i]:v[1][i + 1]].tobytes() for i in range(min(a.limit, len(v[2])))],
                         v[2][:a.limit].numpy()) for s, v in data.items()}
    print({s: len(v[2]) for s, v in data.items()}, f"loaded in {time.time() - t0:.0f}s", flush=True)

    norm = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    tf_train = transforms.Compose([transforms.Lambda(_square),
                                   transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
                                   transforms.RandomHorizontalFlip()] + norm)
    tf_eval = transforms.Compose([transforms.Lambda(_square), transforms.Resize((224, 224))] + norm)
    mk = lambda d, tf, sh: torch.utils.data.DataLoader(_Packed(*d, tf), batch_size=a.batch, shuffle=sh,
                                                        num_workers=a.workers, pin_memory=True, drop_last=sh,
                                                        persistent_workers=a.workers > 0)
    train_dl, val_dl, test_dl = mk(data["train"], tf_train, True), mk(data["val"], tf_eval, False), \
        mk(data["test"], tf_eval, False)

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    total = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    ckpt, best = out / "last.pt", -1.0
    start = 0
    if ckpt.exists():                               # resume (this box can crash, see WORKLOG)
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); start, best = st["epoch"] + 1, st["best"]
        print(f"resumed at epoch {start}", flush=True)
    for ep in range(start, a.epochs):
        model.train()
        t1, run = time.time(), 0.0
        for it, (x, y) in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = lossf(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if it % 200 == 0:
                print(f"ep {ep} it {it}/{len(train_dl)} loss {run / (it + 1):.4f} {time.time() - t1:.0f}s", flush=True)
        v = evaluate(model, val_dl, device)
        print(f"ep {ep} val mean_acc {v['mean_acc']:.4f} targets {v['targets']} ({time.time() - t1:.0f}s)", flush=True)
        if v["mean_acc"] > best:
            best = v["mean_acc"]
            torch.save(model.state_dict(), out / "resnet50_celeba40.pth")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep, best=best), ckpt)

    model.load_state_dict(torch.load(out / "resnet50_celeba40.pth", map_location=device))
    report = dict(val=evaluate(model, val_dl, device), test=evaluate(model, test_dl, device), args=vars(a))
    if a.hq:
        hq = load_hq_test(Path(a.hq))
        hq_dl = torch.utils.data.DataLoader(_Packed(*hq, transforms.Compose(norm)), batch_size=a.batch,
                                            num_workers=a.workers)
        report["hq_test"] = evaluate(model, hq_dl, device)
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    print("test", report["test"]["mean_acc"], report["test"]["targets"],
          "hq_test", report.get("hq_test", {}).get("mean_acc"), report.get("hq_test", {}).get("targets"), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True, help="flwrlabs/celeba snapshot dir")
    p.add_argument("--dst", required=True)
    t = sub.add_parser("train")
    t.add_argument("--celeba", required=True)
    t.add_argument("--hq", help="CelebAMask-HQ dir (attribute-anno + mapping + CelebA-HQ-img) for HQ-domain eval")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--limit", type=int, default=0, help="debug: use only the first N images per split")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(Path(a.src).expanduser(), Path(a.dst).expanduser())
    elif a.cmd == "train":
        train(a)


if __name__ == "__main__":
    main()
