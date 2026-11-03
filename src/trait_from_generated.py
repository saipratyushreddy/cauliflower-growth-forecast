"""
Direction 3: do the generated next photos carry trait information? (design in docs/06-multi-trait-and-trait-link.md, written before any run)

1. Trait readers: ridge on the frozen ResNet18 (ImageNet) embedding of ONE photo -> that day's measured trait (diameter, head diameter), fitted on train-plant photos,
   alpha on validation-plant photos. Two readers per trait: 'sharp' (fitted on the real photos) and 'blur' (fitted on Gaussian-blurred real photos, sigma chosen
   on train photos so the blurred/real gradient-magnitude ratio matches the image models' test ratio, --target-grad-ratio), which removes the blur mismatch
   between real and generated photos. All photos go through the same path: the 256x256 cache, centre crop 224, ImageNet normalisation.
2. Test image-track pairs whose target date has a measurement of the trait. The target-day trait is read from:
     real target photo          ceiling: how well one photo can show that day's trait at all
     last real photo            the photo-space analogue of persistence
     last real photo + growth   last-photo reading + mean train growth of the trait for that (field, last day -> target day) step
     blurred last photo         last photo blurred like the reader's blur
     single-photo / ConvLSTM K=4 / Transformer outputs (frozen checkpoints)
3. MAE against the measured target-day value; paired differences against 'last real photo + growth' with plant-bootstrap 95% intervals; overall and by field.
   A generated photo carries forward-looking trait information only if it beats 'last real photo + growth' with the same reader.
Outputs: outputs/trait_link_results.csv, outputs/trait_link_paired.csv, outputs/trait_link_per_pair.csv.

Usage:
    python src/trait_from_generated.py --image-pairs data/image_pairs.parquet --traits data/traits.parquet --cache-dir data --out-dir outputs \
        --sf-ckpt checkpoints/img_single_frame_best.pt --cl-ckpt checkpoints/img_convlstm_k4_best.pt --tf-ckpt checkpoints/img_transformer_best.pt --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torchvision.models import ResNet18_Weights, resnet18

from img_convlstm import TemporalFiLMUNet
from img_film_unet import FiLMUNet
from img_sharp import sharpness_table
from img_transformer import TemporalTransformerUNet
from train_img_convlstm import build_sequences, predict_uint8 as predict_cl
from train_img_single_frame import load_rows, predict_uint8 as predict_sf
from train_img_transformer import build_histories, predict as predict_tf

TRAITS = ["diameter", "head_diameter"]
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
SIGMAS = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def blur_u8(imgs, sigma, device, bs=64):
    """Gaussian blur of uint8 [N,H,W,3] images on the GPU; returns uint8."""
    if sigma <= 0:
        return imgs
    r = int(np.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).to(device)
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.asarray(imgs[i:i + bs])).to(device).permute(0, 3, 1, 2).float()
        x = F.pad(x, (r, r, r, r), mode="reflect")
        x = F.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        x = F.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
        out.append(x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


class Embedder:
    def __init__(self, device):
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.m, self.dev = m.eval().to(device), device

    @torch.no_grad()
    def __call__(self, imgs, bs=128):
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(np.asarray(imgs[i:i + bs])).to(self.dev).permute(0, 3, 1, 2).float() / 255
            x = x[:, :, 16:240, 16:240]                                   # centre crop 224 of the 256 cache
            x = (x - MEAN.to(self.dev)) / STD.to(self.dev)
            out.append(self.m(x).cpu().numpy())
        return np.concatenate(out)


def boot(diff, plants, n=2000, seed=0):
    g = pd.DataFrame({"d": diff, "p": plants}).groupby("p")["d"].agg(["sum", "count"])
    i = np.random.RandomState(seed).randint(0, len(g), (n, len(g)))
    m = g["sum"].values[i].sum(1) / g["count"].values[i].sum(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--traits", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sf-ckpt", required=True)
    ap.add_argument("--cl-ckpt", required=True)
    ap.add_argument("--tf-ckpt", required=True)
    ap.add_argument("--target-grad-ratio", type=float, default=0.0, help="0 = use the mean gradient ratio of the three models' test outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="small subsets, never reads the test split")
    a = ap.parse_args()
    dev = torch.device(a.device)
    arr = np.load(os.path.join(a.cache_dir, "images256.npy"), mmap_mode="r")
    cidx = pd.read_parquet(os.path.join(a.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    pairs = load_rows(a.image_pairs, a.cache_dir)
    tr_ = pd.read_parquet(a.traits)
    split_of = pairs[["plant_id", "split"]].drop_duplicates().set_index("plant_id")["split"]
    photos = tr_[tr_.filepath.isin(set(cidx)) & tr_.plant_id.isin(split_of.index)].copy()
    photos["split"] = photos.plant_id.map(split_of)
    photos = photos[photos.filepath.isin(set(pairs.target_filepath) | set(pairs.input_last_filepath))]   # cleaned photos only (no black / blurry)
    lab = tr_.set_index("filepath")
    emb = Embedder(dev)

    ev = pairs[pairs.split == ("val" if a.smoke else "test")].sort_values("pair_id").reset_index(drop=True)
    if a.smoke:
        ev = ev.sample(min(64, len(ev)), random_state=0).reset_index(drop=True)
        photos = pd.concat([g.sample(min(400, len(g)), random_state=0) for _, g in photos.groupby("split")])

    # --- generated next photos (frozen checkpoints)
    sd = torch.load(a.sf_ckpt, map_location="cpu")
    sf = FiLMUNet()
    sf.load_state_dict(sd["model"])
    gen = {"single-photo model": predict_sf(sf.eval().to(dev), arr, ev, sd["norm"], dev)}
    sd = torch.load(a.cl_ckpt, map_location="cpu")
    assert sd["args"]["k"] == 4
    cl = TemporalFiLMUNet()
    cl.load_state_dict(sd["model"])
    S = build_sequences(ev, cidx, 4, sd["norm"])
    gen["ConvLSTM K=4"] = predict_cl(cl.eval().to(dev), arr, ev, {"rows": S[0], "gaps": S[1], "mask": S[2]}, dev)
    sd = torch.load(a.tf_ckpt, map_location="cpu")
    ar = sd["args"]
    assert ar.get("max_history", 0) == 0
    tf = TemporalTransformerUNet(d_model=ar["d_model"], nhead=ar["nhead"], num_layers=ar["num_layers"], dim_ff=ar["dim_ff"], dropout=ar["dropout"],
                                 grad_checkpoint=False)
    tf.load_state_dict(sd["model"])
    gen["Transformer"], _ = predict_tf(tf.eval().to(dev), arr, ev, build_histories(ev, cidx, 0), dev)
    real_tg = np.stack([np.asarray(arr[int(r)]) for r in ev.row_tg])
    real_last = np.stack([np.asarray(arr[int(r)]) for r in ev.row_in])
    ratios = {k: sharpness_table(v, real_tg)["grad_ratio"] for k, v in gen.items()}
    target = a.target_grad_ratio or float(np.mean(list(ratios.values())))
    print("gradient ratio of model outputs:", {k: round(v, 3) for k, v in ratios.items()}, "-> blur target", round(target, 3), flush=True)

    # --- blur level matching the models' sharpness (chosen on TRAIN photos)
    trp = photos[photos.split == "train"].sample(min(300, (photos.split == "train").sum()), random_state=0)
    trimg = np.stack([np.asarray(arr[cidx[f]]) for f in trp.filepath])
    sig_r = {s: sharpness_table(blur_u8(trimg, s, dev), trimg)["grad_ratio"] for s in SIGMAS}
    sigma = min(SIGMAS, key=lambda s: abs(sig_r[s] - target))
    print("blur sigma -> gradient ratio:", {s: round(v, 3) for s, v in sig_r.items()}, "chosen sigma", sigma, flush=True)

    # --- embeddings of every photo the readers need
    imgs = np.stack([np.asarray(arr[cidx[f]]) for f in photos.filepath])
    E_sharp, E_blur = emb(imgs), emb(blur_u8(imgs, sigma, dev))
    conds = {"real target photo": real_tg, "last real photo": real_last, "blurred last photo": blur_u8(real_last, sigma, dev), **gen}
    E_cond = {k: emb(v) for k, v in conds.items()}

    rows, paired, per_pair = [], [], []
    for trait in TRAITS:
        ok = photos[trait].notna().values
        P = photos[ok]
        yv = P[trait].values
        m_tr, m_va = (P.split == "train").values, (P.split == "val").values
        ev_y = ev.target_filepath.map(lab[trait]).values.astype(float)
        last_y_day = ev.input_last_filepath.map(lab["day_after_planting"]).values
        keep = ~np.isnan(ev_y)
        if keep.sum() < 5:
            print(f"{trait}: only {keep.sum()} evaluation pairs, skipped", flush=True)
            continue
        # mean train growth of the trait per (field, last day -> target day), from consecutive labelled photos of train plants
        tt = tr_[tr_.plant_id.map(split_of).eq("train") & tr_[trait].notna()].sort_values(["plant_id", "day_after_planting"])
        tt = tt.assign(prev=tt.groupby("plant_id")[trait].shift(), prev_day=tt.groupby("plant_id").day_after_planting.shift()).dropna(subset=["prev"])
        growth = (tt[trait] - tt.prev).groupby([tt.field, tt.prev_day, tt.day_after_planting]).mean()
        step = np.array([growth.get((f, l, t), 0.0) for f, l, t in zip(ev.field, last_y_day, ev.target_day)])
        for reader, E in (("sharp", E_sharp[ok]), ("blur", E_blur[ok])):
            best = min(ALPHAS, key=lambda al: np.abs(Ridge(alpha=al).fit(E[m_tr], yv[m_tr]).predict(E[m_va]) - yv[m_va]).mean())
            R = Ridge(alpha=best).fit(E[m_tr], yv[m_tr])
            same_day = np.abs(R.predict(E[m_va]) - yv[m_va]).mean()
            preds = {k: R.predict(v) for k, v in E_cond.items()}
            preds["last real photo + growth"] = preds["last real photo"] + step
            errs = {k: np.abs(p - ev_y)[keep] for k, p in preds.items()}
            sub = ev[keep]
            for k, e in errs.items():
                lo, hi = boot(e, sub.plant_id.values)
                r = {"trait": trait, "reader": reader, "alpha": best, "reader_val_mae_same_day": same_day, "condition": k, "n": int(keep.sum()),
                     "n_plants": sub.plant_id.nunique(), "mae": e.mean(), "ci_lo": lo, "ci_hi": hi}
                for f in sorted(sub.field.unique()):
                    r[f"mae_{f}"] = e[sub.field.values == f].mean()
                rows.append(r)
                per_pair.append(pd.DataFrame({"trait": trait, "reader": reader, "condition": k, "pair_id": sub.pair_id, "plant_id": sub.plant_id,
                                              "field": sub.field, "y_true": ev_y[keep], "y_pred": preds[k][keep]}))
            ref = errs["last real photo + growth"]
            for k in ("single-photo model", "ConvLSTM K=4", "Transformer", "last real photo", "blurred last photo"):
                for sname, msk in [("all", np.ones(len(sub), bool))] + [(f, sub.field.values == f) for f in sorted(sub.field.unique())]:
                    if msk.sum() < 2:
                        continue
                    lo, hi = boot((errs[k] - ref)[msk], sub.plant_id.values[msk])
                    paired.append({"trait": trait, "reader": reader, "condition": k, "vs": "last real photo + growth", "subset": sname, "n": int(msk.sum()),
                                   "diff": float((errs[k] - ref)[msk].mean()), "ci_lo": lo, "ci_hi": hi})
    pre = "smoke_" if a.smoke else ""
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"{pre}trait_link_results.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(a.out_dir, f"{pre}trait_link_paired.csv"), index=False)
    pd.concat(per_pair).to_csv(os.path.join(a.out_dir, f"{pre}trait_link_per_pair.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
