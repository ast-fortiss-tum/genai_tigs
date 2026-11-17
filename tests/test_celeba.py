"""CelebA replication code: packing/prepare round-trip, classifier kind, LoRA image selection (CPU, synthetic data)."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from dptig import celeba_classifier as cc
from dptig import celeba_lora_data as cl


def _png(color):
    b = io.BytesIO()
    Image.new("RGB", (178, 218), color).save(b, format="PNG")
    return b.getvalue()


class TestCeleba(unittest.TestCase):
    def test_attrs_and_targets(self):
        self.assertEqual(len(cc.ATTRS), 40)
        for name, i in cc.TARGETS.items():
            self.assertEqual(cc.ATTRS[i], name)

    def test_pack_roundtrip(self):
        blobs = [_png((i * 40, 0, 0)) for i in range(3)]
        buf, offs, lab = cc._pack(blobs, [[1.0] * 40, [0.0] * 40, [1.0] * 40])
        ds = cc._Packed(buf, offs, lab, lambda im: torch.tensor(np.asarray(im)[0, 0, 0]))
        self.assertEqual(len(ds), 3)
        self.assertEqual([int(ds[i][0]) for i in range(3)], [0, 40, 80])
        self.assertEqual(float(ds[1][1].sum()), 0.0)

    def test_prepare_and_load(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t) / "src", Path(t) / "dst"
            (src / "img_align+identity+attr").mkdir(parents=True)
            sizes = {"train": 3, "valid": 2, "test": 2}
            for prefix, n in sizes.items():
                cols = {"image": [{"bytes": _png((k, 0, 0)), "path": None} for k in range(n)],
                        "celeb_id": list(range(n))}
                cols.update({a: [bool(k % 2) for k in range(n)] for a in cc.ATTRS})
                pq.write_table(pa.table(cols), src / "img_align+identity+attr" / f"{prefix}-00000-of-00001.parquet")
            with mock.patch.dict(cc._EXPECTED, {"train": 3, "val": 2, "test": 2}):
                cc.prepare(src, dst)
                data = cc.load_celeba(dst)
            self.assertEqual({s: len(v[2]) for s, v in data.items()}, {"train": 3, "val": 2, "test": 2})
            buf, offs, lab = data["train"]
            img = Image.open(io.BytesIO(buf[offs[1]:offs[2]].tobytes()))
            self.assertEqual(img.getpixel((0, 0)), (1, 0, 0))
            self.assertEqual(lab[1].tolist(), [1.0] * 40)

    def test_classifier_kind(self):
        from dptig.classifiers import load_classifier
        with tempfile.TemporaryDirectory() as t:
            w = Path(t) / "w.pth"
            torch.save(cc.build_model().state_dict(), w)
            clf = load_classifier({"kind": "celeba_resnet50", "weights": str(w)}, torch.device("cpu"))
            s = clf.scores([Image.new("RGB", (512, 512), (120, 100, 90))] * 2)
        self.assertEqual(s.shape, (2, 40))

    def test_lora_selection(self):
        with tempfile.TemporaryDirectory() as t:
            hq = Path(t)
            rows, mapping = [], []
            for i in range(300):
                at = {a: -1 for a in cc.ATTRS}
                at[["Smiling", "Bangs", "Eyeglasses"][i % 3]] = 1
                if i % 30 == 0:
                    at["Bangs"] = 1                     # two targets -> must be excluded
                rows.append(f"{i}.jpg " + " ".join(str(at[a]) for a in cc.ATTRS))
                orig = 200000 if i % 7 == 0 else 1000 + i   # test-partition source -> excluded
                mapping.append(f"{i} {orig - 1} {orig:06d}.jpg")
            (hq / "CelebAMask-HQ-attribute-anno.txt").write_text("300\n" + " ".join(cc.ATTRS) + "\n" + "\n".join(rows))
            (hq / "CelebA-HQ-to-CelebA-mapping.txt").write_text("idx orig_idx orig_file\n" + "\n".join(mapping))
            with mock.patch.object(cl, "N_IMAGES", 10):
                picks = cl.select(hq, seed=0)
                again = cl.select(hq, seed=0)
        self.assertEqual(picks, again)
        for token, attr in [("smiling-face", 0), ("curtain-bangs", 1), ("eye-glasses", 2)]:
            self.assertEqual(len(picks[token]), 10)
            for f in picks[token]:
                i = int(f[:-4])
                self.assertEqual(i % 3, attr)
                self.assertNotEqual(i % 30, 0)
                self.assertNotEqual(i % 7, 0)


if __name__ == "__main__":
    unittest.main()
