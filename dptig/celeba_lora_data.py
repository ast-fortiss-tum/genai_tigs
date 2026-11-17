"""Build the kohya_ss training folder for the CelebA-HQ concept LoRA (our replication; authors' LoRA unavailable).

    python -m dptig.celeba_lora_data --hq ~/data/celebamask_hq/CelebAMask-HQ --out ~/data/celeba_lora_train

Recipe copied from the metadata of the authors' ImageNet LoRAs (weights/lora/pizza_imgnet-000005.safetensors,
kohya_ss v21.8.9): 38 images per concept, 25 repeats, no regularisation images, caption = folder class token
(i.e. the trigger word only), 512x512. One LoRA for the three concepts, like the CIFAR-10 LoRA (10 concepts).

Our choices (not documented): images carry the target attribute and NOT the other two target attributes (so each
trigger word maps to one attribute); Blurry images excluded; only CelebA-HQ images whose source CelebA image is
outside the CelebA test partition (keeps the classifier's HQ test set independent); with --clf, CelebA label noise is
filtered by requiring our classifier to agree confidently (target p>0.9, other two targets p<0.1) -- a reproducible
stand-in for the manual curation the authors likely did (an unfiltered pick had an "Eyeglasses" face without glasses);
random pick with a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from .celeba_classifier import ATTRS, VAL_MAX

CONCEPTS = {"Smiling": "smiling-face", "Bangs": "curtain-bangs", "Eyeglasses": "eye-glasses"}   # paper Sec. 3
N_IMAGES, REPEATS, SIZE = 38, 25, 512


def _clf_probs(hq_root: Path, files, weights: Path):
    """Sigmoid outputs of our CelebA classifier for the given CelebA-HQ files -> {file: (40,) array}."""
    import torch
    from .celeba_classifier import build_model, preprocess
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model()
    m.load_state_dict(torch.load(weights, map_location=dev))
    m = m.to(dev).eval()
    tf, out = preprocess(), {}
    with torch.no_grad():
        for i in range(0, len(files), 128):
            chunk = files[i:i + 128]
            x = torch.stack([tf(Image.open(hq_root / "CelebA-HQ-img" / f).convert("RGB")) for f in chunk]).to(dev)
            for f, p in zip(chunk, torch.sigmoid(m(x)).cpu().numpy()):
                out[f] = p
    return out


def select(hq_root: Path, seed: int = 0, clf_weights: Path | None = None, hi: float = 0.9, lo: float = 0.1):
    lines = (hq_root / "CelebAMask-HQ-attribute-anno.txt").read_text().splitlines()
    assert lines[1].split() == ATTRS
    orig = {}
    for l in (hq_root / "CelebA-HQ-to-CelebA-mapping.txt").read_text().splitlines()[1:]:
        idx, _, orig_file = l.split()
        orig[f"{idx}.jpg"] = int(orig_file.split(".")[0])
    rows = []
    for l in lines[2:]:
        p = l.split()
        rows.append((p[0], {a: v == "1" for a, v in zip(ATTRS, p[1:])}))
    picks, pools = {}, {}
    for attr in CONCEPTS:
        others = [o for o in CONCEPTS if o != attr]
        pools[attr] = sorted(f for f, at in rows if at[attr] and not any(at[o] for o in others) and not at["Blurry"]
                             and orig[f] <= VAL_MAX)
    probs = _clf_probs(hq_root, sorted({f for p in pools.values() for f in p}), clf_weights) if clf_weights else None
    for attr, token in CONCEPTS.items():
        pool = pools[attr]
        if probs is not None:     # label-noise filter: our classifier must agree confidently with the annotation
            ia, io = ATTRS.index(attr), [ATTRS.index(o) for o in CONCEPTS if o != attr]
            pool = [f for f in pool if probs[f][ia] > hi and all(probs[f][j] < lo for j in io)]
        picks[token] = sorted(random.Random(f"{seed}-{attr}").sample(pool, N_IMAGES), key=lambda s: int(s[:-4]))
        print(f"{token}: pool {len(pools[attr])}, after classifier filter {len(pool)}, picked {N_IMAGES}")
    return picks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clf", help="our CelebA classifier weights; enables the label-noise filter (recommended)")
    a = ap.parse_args(argv)
    hq, out = Path(a.hq).expanduser(), Path(a.out).expanduser()
    if (out / "img").exists():
        raise SystemExit(f"{out}/img exists; remove it first")
    picks = select(hq, a.seed, Path(a.clf).expanduser() if a.clf else None)
    for token, files in picks.items():
        d = out / "img" / f"{REPEATS}_{token}"
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            Image.open(hq / "CelebA-HQ-img" / f).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS) \
                .save(d / f.replace(".jpg", ".png"))
    (out / "selection.json").write_text(json.dumps(dict(seed=a.seed, clf=a.clf, concepts=CONCEPTS, picks=picks),
                                                    indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
