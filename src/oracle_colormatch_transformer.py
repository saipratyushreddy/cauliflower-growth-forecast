"""
ORACLE colour-match DIAGNOSTIC on the full-history Transformer and the single-frame Transformer (K=1): is the history gap on
the early-Field2 subset (targets days 16, 22; n=146) an exposure effect?

DIAGNOSTIC ONLY, NOT achievable at inference: each prediction is shifted/scaled per channel to its OWN TARGET's mean and std (the same
oracle transform as the Step B oracle, image_copy_forward_baseline.color_match), then structure SSIM / RGB SSIM / PSNR are recomputed.
No training and no model selection: both checkpoints are the already-selected ones.

Integrity check: the recomputed "before colour-matching" per-pair scores are compared with the per-pair CSVs saved when each model was
first evaluated (they should agree to rounding; a GPU run can differ in the last digits).

Usage:
    python src/oracle_colormatch_transformer.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs \
        --tf-ckpt checkpoints/img_transformer_best.pt --tf-k1-ckpt checkpoints/img_transformer_k1_best.pt \
        --tf-csv outputs/img_transformer_per_pair.csv --tf-k1-csv outputs/img_transformer_k1_per_pair.csv --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from compare_generic_date import ratio_ci
from compare_temporal import boot_ci
from image_copy_forward_baseline import color_match
from img_transformer import TemporalTransformerUNet
from train_img_single_frame import load_rows, score_pairs
from train_img_transformer import build_histories, predict


def mean_match(pred_u8, target_u8):
    """SUPPLEMENTARY oracle: per-channel MEAN offset only (no std rescaling): removes brightness/colour offsets without
    stretching the prediction's contrast."""
    d = target_u8.astype(np.float64).mean((0, 1)) - pred_u8.astype(np.float64).mean((0, 1))
    return np.clip(np.rint(pred_u8.astype(np.float64) + d), 0, 255).astype(np.uint8)


def run_model(ckpt_path, pairs_ev, arr, cache_idx, device):
    sd = torch.load(ckpt_path, map_location=device)
    a = sd["args"]
    model = TemporalTransformerUNet(d_model=a["d_model"], nhead=a["nhead"], num_layers=a["num_layers"], dim_ff=a["dim_ff"],
                                    dropout=a["dropout"], grad_checkpoint=False).to(device)
    model.load_state_dict(sd["model"])
    H = build_histories(pairs_ev, cache_idx, a.get("max_history", 0))
    preds, _ = predict(model, arr, pairs_ev, H, device)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tf-ckpt", required=True)
    ap.add_argument("--tf-k1-ckpt", required=True)
    ap.add_argument("--tf-csv", default=None, help="saved per-pair CSV of the full Transformer (integrity check)")
    ap.add_argument("--tf-k1-csv", default=None, help="saved per-pair CSV of the K=1 Transformer (integrity check)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--early-days", type=int, nargs="+", default=[16, 22])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = torch.device(args.device)

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    cache_idx = pd.read_parquet(os.path.join(args.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    te = pairs[pairs.split == "test"].sort_values("pair_id").reset_index(drop=True)
    early = ((te.field == "Field2") & te.target_day.isin(args.early_days)).values
    print(f"{len(te)} test pairs; early Field2 subset (targets {args.early_days}): {early.sum()}", flush=True)

    out = {}
    for name, ck, csv in [("TF", args.tf_ckpt, args.tf_csv), ("TFK1", args.tf_k1_ckpt, args.tf_k1_csv)]:
        preds = run_model(ck, te, arr, cache_idx, device)
        before = score_pairs(preds, arr, te)
        matched = np.stack([color_match(p, arr[int(r)]) for p, r in zip(preds, te.row_tg)])   # ORACLE: uses the target's own stats
        after = score_pairs(matched, arr, te)
        matched_m = np.stack([mean_match(p, arr[int(r)]) for p, r in zip(preds, te.row_tg)])   # supplementary: mean offset only
        after_mean = score_pairs(matched_m, arr, te)
        out[name] = (before, after, after_mean)
        if csv:
            ref = pd.read_csv(csv).set_index("pair_id").loc[te.pair_id]
            d = np.abs(before.structure_ssim.values - ref.structure_ssim.values)
            print(f"[{name}] integrity: recomputed vs saved structure SSIM: max |diff| = {d.max():.2e}, mean {d.mean():.2e}; "
                  f"pooled mean {before.structure_ssim.mean():.4f} vs saved {ref.structure_ssim.mean():.4f}", flush=True)
        for lab, df in [("before", before), ("after", after), ("aftermean", after_mean)]:
            df.to_csv(os.path.join(args.out_dir, f"{args.tag}img_oracle_cm_{name}_{lab}_per_pair.csv"), index=False)

    pl = te.plant_id.values
    print("\n(ORACLE diagnostic: predictions colour-matched to their own targets; not achievable at inference)\n")
    for mname, mask in [("early Field2 subset (n=%d)" % early.sum(), early), ("all OTHER test pairs (n=%d)" % (~early).sum(), ~early),
                        ("all test pairs (n=%d)" % len(te), np.ones(len(te), bool))]:
        print(f"== {mname}")
        print("| Metric | Model | before colour-matching | after colour-matching (mean+std; PRIMARY) | after mean-only match (supplementary) |\n|---|---|---|---|---|")
        for m, nd in [("structure_ssim", 4), ("ssim", 3), ("psnr", 2)]:
            for nm, lab in [("TFK1", "Transformer K=1"), ("TF", "Transformer (full)")]:
                b, a, c = (out[nm][i][m].values[mask] for i in range(3))
                print(f"| {m} | {lab} | {b.mean():.{nd}f} ({np.median(b):.{nd}f}) | {a.mean():.{nd}f} ({np.median(a):.{nd}f}) | {c.mean():.{nd}f} ({np.median(c):.{nd}f}) |")
        print("\n| Gap (Transformer full minus Transformer K=1) | before | after mean+std (PRIMARY) | share remaining | after mean-only (supplementary) | share remaining |\n|---|---|---|---|---|---|")
        for m, nd in [("structure_ssim", 4), ("ssim", 4), ("psnr", 3)]:
            g = [out["TF"][i][m].values - out["TFK1"][i][m].values for i in range(3)]
            cells = []
            for i in range(3):
                lo, hi = boot_ci(g[i][mask], pl[mask])
                cells.append(f"{g[i][mask].mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[i][mask] > 0).mean():.0%}")
            shares = []
            for i in (1, 2):
                if abs(g[0][mask].mean()) > 1e-9:
                    rl, rh = ratio_ci(g[i][mask], g[0][mask], pl[mask])
                    shares.append(f"{g[i][mask].mean() / g[0][mask].mean():.2f} [{rl:.2f}, {rh:.2f}]")
                else:
                    shares.append("n/a")
            print(f"| {m} | {cells[0]} | {cells[1]} | {shares[0]} | {cells[2]} | {shares[1]} |")
        print()


if __name__ == "__main__":
    main()
