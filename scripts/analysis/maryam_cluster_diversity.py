"""Apply Maryam's Table-3 diversity (diversity/compute_diversity_*.py + evaluation_results/paired-ttest.py)
to our runs/main images. Faithful copy of her algorithm: KMeans(K, n_init=10, rs=0) on the union of
FP_LOW, FP_HIGH, DP_LOW, DP_HIGH embeddings; per group coverage * normalised entropy; paired t-test over K=2..12."""
import sys, glob
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sklearn.cluster import KMeans
from scipy.stats import ttest_rel

runs = Path(sys.argv[1]); tasks = sys.argv[2:]
K_LIST = [2, 4, 6, 8, 10, 12]
dev = "cuda" if torch.cuda.is_available() else "cpu"

def clip_model():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def enc(paths):
        x = p(images=[Image.open(q).convert("RGB") for q in paths], return_tensors="pt")["pixel_values"].to(dev)
        with torch.no_grad(): z = m.get_image_features(pixel_values=x).float()
        return (z / (z.norm(dim=-1, keepdim=True) + 1e-9)).cpu().numpy()   # EMB_L2=True
    return enc

def resnet_model():
    import torchvision as tv
    m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity(); m = m.to(dev).eval()
    tf = tv.transforms.Compose([tv.transforms.Resize(256), tv.transforms.CenterCrop(224), tv.transforms.ToTensor(),
                                tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def enc(paths):
        x = torch.stack([tf(Image.open(q).convert("RGB")) for q in paths]).to(dev)
        with torch.no_grad(): return m(x).float().cpu().numpy()          # EMB_L2=False
    return enc

def cov_ent(labels, K):
    c = np.bincount(labels, minlength=K).astype(float); cov = (c > 0).sum() / K
    p = c[c > 0] / c.sum(); H = -(p * np.log2(p)).sum() / np.log2(K)
    return cov * H

def pairwise_cos(Z):  # our Eq. 6 for reference
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True); S = Zn @ Zn.T; n = len(Z)
    return float((1 - S)[np.triu_indices(n, 1)].mean())

encs = {"clip": clip_model(), "resnet50": resnet_model()}
for task in tasks:
    groups = {}
    for arm in ("fp", "dp"):
        for lv in ("low", "high"):
            groups[f"{arm}_{lv}"] = sorted(glob.glob(str(runs / f"{task}_{arm}" / lv / "*.png")))
    order = ["fp_low", "fp_high", "dp_low", "dp_high"]
    for ename, enc in encs.items():
        Zs = {g: enc(groups[g]) for g in order}
        Z = np.vstack([Zs[g] for g in order]); sizes = [len(Zs[g]) for g in order]
        idx = np.split(np.arange(len(Z)), np.cumsum(sizes)[:-1])
        div = {g: [] for g in order + ["fp_all", "dp_all"]}
        for K in K_LIST:
            lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Z).labels_
            for g, ix in zip(order, idx): div[g].append(cov_ent(lab[ix], K))
            div["fp_all"].append(cov_ent(np.concatenate([lab[idx[0]], lab[idx[1]]]), K))
            div["dp_all"].append(cov_ent(np.concatenate([lab[idx[2]], lab[idx[3]]]), K))
        print(f"\n{task} {ename}  n={dict(zip(order, sizes))}")
        for cond, a, b in [("LOW", "fp_low", "dp_low"), ("HIGH", "fp_high", "dp_high"), ("OVERALL", "fp_all", "dp_all")]:
            A, B = np.array(div[a]), np.array(div[b]); t, p = ttest_rel(A, B); d = (A - B).mean() / (A - B).std(ddof=1)
            print(f"  {cond:7s} FP {A.mean():.3f}  DP {B.mean():.3f}  p={p:.3f} d={d:+.2f}  "
                  f"| Eq6 FP {pairwise_cos(Zs.get(a, np.vstack([Zs['fp_low'], Zs['fp_high']]))):.3f}"
                  f" DP {pairwise_cos(Zs.get(b, np.vstack([Zs['dp_low'], Zs['dp_high']]))):.3f}")
