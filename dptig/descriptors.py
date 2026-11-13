"""Data-derived conditioning descriptors (paper Sec. 3.1.2): caption reference images with BLIP, count
terms, keep frequent ones. The final shortlist is curated by hand (as in the paper: 'terms that represent
relevant visual characteristics'), so this tool only produces the ranked candidates."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

STOPWORDS = set("""a an the of on in at with and or to is are was be being been it its this that these those
there their his her he she they them from by for as into onto up down over under near next front back behind
very some one two three other another around while has have having also can could would sitting standing
laying lying looking holding picture photo image close view top side""".split())


def extract_terms(captions: Iterable[str], exclude: Sequence[str] = (), top_k: int = 20,
                  min_count: int = 2) -> List[Tuple[str, int]]:
    """Ranked (term, count). Terms are counted once per caption. ``exclude`` removes the class name words."""
    ex = {w.lower() for e in exclude for w in re.findall(r"[a-z]+", e.lower())}
    counts: Counter = Counter()
    for cap in captions:
        words = {w for w in re.findall(r"[a-z]+", cap.lower()) if len(w) > 2 and w not in STOPWORDS and w not in ex}
        counts.update(words)
    return [(w, c) for w, c in counts.most_common(top_k) if c >= min_count]


def caption_images(paths: Sequence[Path], model_id: str = "Salesforce/blip-image-captioning-base",
                   device: str = "cpu") -> List[str]:
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    caps = []
    for p in paths:
        inputs = proc(Image.open(p).convert("RGB"), return_tensors="pt").to(device)
        caps.append(proc.decode(model.generate(**inputs, max_new_tokens=30)[0], skip_special_tokens=True))
    return caps


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rank candidate conditioning descriptors for one class.")
    ap.add_argument("--images", required=True, help="folder of reference images of ONE class")
    ap.add_argument("--category", required=True, help="class name, excluded from the terms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    caps = caption_images(paths, device=a.device)
    ranked = extract_terms(caps, exclude=[a.category], top_k=a.top_k)
    Path(a.out).write_text(json.dumps(dict(category=a.category, n_images=len(paths), captions=caps,
                                           candidates=ranked), indent=1))
    print(ranked)


if __name__ == "__main__":
    main()
