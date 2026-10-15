"""
Memorisation check: how do the frozen models score on plants they were TRAINED on versus plants they never saw?

Runs the frozen Transformer (full history) and ConvLSTM K=4 checkpoints on the train, validation and test pairs (no augmentation, same scoring as every README
table), plus copy-forward as a DIFFICULTY control (copy-forward uses no training, so its train-vs-test difference shows how much harder/easier the splits are
by themselves). Reports per split, per field, with/without Field1 day-28 pairs: mean (median) of structure SSIM, RGB SSIM, PSNR; then for each model the
"gain over copy-forward" per split and the train-minus-test difference of that gain with a plant-level bootstrap CI (plants differ between splits, so groups
are resampled independently). Inference only; no model selection.

Usage:
    python src/memorization_check.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs \
        --tf-ckpt checkpoints/img_transformer_best.pt --cl-ckpt checkpoints/img_convlstm_k4_best.pt --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from img_convlstm import TemporalFiLMUNet
from img_transformer import TemporalTransformerUNet
from train_img_convlstm import build_sequences, predict_uint8
from train_img_single_frame import load_rows, score_pairs
from train_img_transformer import build_histories, predict

METRICS = [("structure_ssim", 4), ("ssim", 4), ("psnr", 2)]


def boot_diff(x_tr, p_tr, x_te, p_te, n=2000, seed=0):
    """CI of mean(x_tr) - mean(x_te), resampling plants independently in each group."""
    def agg(x, p):
        g = pd.DataFrame({"x": x, "p": p}).groupby("p")["x"].agg(["sum", "count"])
        return g["sum"].values, g["count"].values
    s1, c1 = agg(x_tr, p_tr)
    s2, c2 = agg(x_te, p_te)
    rng = np.random.RandomState(seed)
    i1 = rng.randint(0, len(s1), (n, len(s1)))
    i2 = rng.randint(0, len(s2), (n, len(s2)))
    d = s1[i1].sum(1) / c1[i1].sum(1) - s2[i2].sum(1) / c2[i2].sum(1)
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tf-ckpt", required=True)
    ap.add_argument("--cl-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", type=int, default=0, help="subsample this many train pairs (0 = all); smoke tests only")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = torch.device(a.device)
    pairs = load_rows(a.image_pairs, a.cache_dir)
    arr = np.load(os.path.join(a.cache_dir, "images256.npy"), mmap_mode="r")
    cache_idx = pd.read_parquet(os.path.join(a.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()

    sd = torch.load(a.tf_ckpt, map_location="cpu")
    ar = sd["args"]
    assert ar.get("max_history", 0) == 0
    tf = TemporalTransformerUNet(d_model=ar["d_model"], nhead=ar["nhead"], num_layers=ar["num_layers"], dim_ff=ar["dim_ff"],
                                 dropout=ar["dropout"], grad_checkpoint=False)
    tf.load_state_dict(sd["model"])
    tf.eval().to(dev)
    sd = torch.load(a.cl_ckpt, map_location="cpu")
    assert sd["args"]["k"] == 4
    cl_norm = sd["norm"]
    cl = TemporalFiLMUNet()
    cl.load_state_dict(sd["model"])
    cl.eval().to(dev)

    frames = []
    for split in ("train", "val", "test"):
        d = pairs[pairs.split == split].sort_values("pair_id").reset_index(drop=True)
        if split == "train" and a.max_train:
            d = d.sample(min(a.max_train, len(d)), random_state=0).sort_values("pair_id").reset_index(drop=True)
        print(f"{split}: {len(d)} pairs, {d.plant_id.nunique()} plants", flush=True)
        if len(d) == 0:
            continue
        H = build_histories(d, cache_idx, 0)
        p_tf, _ = predict(tf, arr, d, H, dev)
        S = build_sequences(d, cache_idx, 4, cl_norm)
        p_cl = predict_uint8(cl, arr, d, {"rows": S[0], "gaps": S[1], "mask": S[2]}, dev)
        p_cf = np.stack([np.asarray(arr[int(r)]) for r in d["row_in"]])
        for name, p in (("Transformer", p_tf), ("ConvLSTM K=4", p_cl), ("Copy-forward", p_cf)):
            s = score_pairs(p, arr, d)
            s["model"], s["split"] = name, split
            s["target_day"], s["input_last_day"] = d["target_day"].values, d["input_last_day"].values
            frames.append(s)
    res = pd.concat(frames, ignore_index=True)
    res["d28"] = (res.field == "Field1") & ((res.target_day == 28) | (res.input_last_day == 28))
    res.to_csv(os.path.join(a.out_dir, f"{a.tag}img_memorization_per_pair.csv"), index=False)

    subsets = {"all pairs": lambda d: d, "excluding Field1 day-28 pairs": lambda d: d[~d.d28],
               "Field1 (excl. day 28)": lambda d: d[(d.field == "Field1") & ~d.d28], "Field2": lambda d: d[d.field == "Field2"]}
    for sname, f in subsets.items():
        print(f"\n== {sname}: mean (median) by split")
        print("| Model | Split | Pairs | Structure SSIM | RGB SSIM | PSNR dB |\n|---|---|---|---|---|---|")
        for m in ("Copy-forward", "ConvLSTM K=4", "Transformer"):
            for sp in ("train", "val", "test"):
                g = f(res[(res.model == m) & (res.split == sp)])
                print(f"| {m} | {sp} | {len(g):,} | " + " | ".join(f"{g[k].mean():.{nd}f} ({g[k].median():.{nd}f})" for k, nd in METRICS) + " |")
        print(f"\n   Gain over copy-forward (model minus copy-forward on the same pairs), and train-minus-test of that gain [95% plant-bootstrap CI]")
        print("   | Model | Metric | gain on train | gain on test | train minus test |\n   |---|---|---|---|---|")
        for m in ("ConvLSTM K=4", "Transformer"):
            gains = {}
            for sp in ("train", "test"):
                a_ = f(res[(res.model == m) & (res.split == sp)]).set_index("pair_id")
                b_ = f(res[(res.model == "Copy-forward") & (res.split == sp)]).set_index("pair_id")
                c = a_.index.intersection(b_.index)
                gains[sp] = {k: (a_.loc[c, k].values - b_.loc[c, k].values, a_.loc[c, "plant_id"].values) for k, _ in METRICS}
            for k, nd in METRICS:
                (g_tr, p_tr), (g_te, p_te) = gains["train"][k], gains["test"][k]
                lo, hi = boot_diff(g_tr, p_tr, g_te, p_te)
                print(f"   | {m} | {k} | {g_tr.mean():+.{nd}f} | {g_te.mean():+.{nd}f} | {g_tr.mean() - g_te.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] |")


if __name__ == "__main__":
    main()
