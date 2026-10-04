"""
Quick check (no training): are day-28 TARGET images lower-contrast than typical targets?
Tests the candidate "SSIM / structure SSIM trivially reward smooth output on a low-contrast target".

Per test pair we compute, for the TARGET image and (separately) the INPUT image:
  luma_std      std of grayscale pixels (0-255 scale)
  rgb_std       mean over channels of per-channel pixel std
  luma_mean     mean grayscale
  local_std     median Gaussian-weighted local std of luma (sigma = LCN_SIGMA, [0,1] scale): what the
                local contrast normalization in structure SSIM actually sees
  frac_below_eps  fraction of pixels with local std < LCN_EPS (0.05), where LCN does not amplify structure
Groups (mean / median, n):  target-is-day-28 vs ALL other test targets, vs other FIELD1 targets (field control);
and input-is-day-28 vs other / other Field1 inputs. Mann-Whitney U p-values are reported.
With --per-pair-csv NAME=path ..., also reports the Spearman correlation between target contrast and
that method's per-pair SSIM / structure SSIM over ALL test pairs (is smooth output rewarded on low-contrast
targets in general, not only on day 28?).

Usage:
    python src/target_variance_check.py --image-pairs data/image_pairs.parquet --cache-dir data \
        --per-pair-csv copy-fwd=outputs/img_stepB_copy_forward_per_pair.csv C1=outputs/img_ctrl1_per_pair.csv \
        model=outputs/img_single_frame_per_pair.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu, spearmanr

from image_copy_forward_baseline import LCN_EPS, LCN_SIGMA
from train_img_single_frame import load_rows


def contrast_stats(img_u8):
    g = np.asarray(Image.fromarray(img_u8).convert("L"), dtype=np.float64)
    gl = g / 255.0
    mu = gaussian_filter(gl, LCN_SIGMA)
    ls = np.sqrt(np.maximum(gaussian_filter((gl - mu) ** 2, LCN_SIGMA), 0))
    return {"luma_std": g.std(), "rgb_std": img_u8.astype(np.float64).std((0, 1)).mean(), "luma_mean": g.mean(),
            "local_std": float(np.median(ls)), "frac_below_eps": float((ls < LCN_EPS).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--per-pair-csv", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    te = pairs[pairs.split == "test"].reset_index(drop=True)
    tg = pd.DataFrame([contrast_stats(arr[int(r)]) for r in te.row_tg]).add_prefix("tg_")
    ip = pd.DataFrame([contrast_stats(arr[int(r)]) for r in te.row_in]).add_prefix("in_")
    df = pd.concat([te[["pair_id", "field", "input_last_day", "target_day"]], tg, ip], axis=1)
    f1 = df.field == "Field1"
    out = {}
    for side, flag, groups in [
        ("target", df.target_day == 28, "tg_"), ("input", df.input_last_day == 28, "in_")]:
        d28 = f1 & flag
        comps = {"day-28 " + side: d28, "all other test": ~d28, "other Field1": f1 & ~d28, "Field2": ~f1}
        print(f"\n=== {side.upper()} image contrast, test pairs ({d28.sum()} day-28-{side} pairs) ===")
        rows = []
        for name, m in comps.items():
            r = {"group": name, "n": int(m.sum())}
            for k in ["luma_std", "rgb_std", "luma_mean", "local_std", "frac_below_eps"]:
                v = df.loc[m, groups + k]
                r[k] = f"{v.mean():.3f} ({v.median():.3f})" if k in ("local_std", "frac_below_eps") else f"{v.mean():.1f} ({v.median():.1f})"
            rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))
        print("  (cells are mean (median); luma/rgb std and luma_mean on 0-255; local_std on 0-1; eps = %.2f)" % LCN_EPS)
        for k in ["luma_std", "local_std"]:
            a = df.loc[d28, groups + k]
            for cname in ["all other test", "other Field1"]:
                b = df.loc[comps[cname], groups + k]
                u = mannwhitneyu(a, b, alternative="two-sided")
                print(f"  {k}: day-28 {side} median {a.median():.3f} vs {cname} median {b.median():.3f} "
                      f"(ratio {a.median() / b.median():.2f}), Mann-Whitney p = {u.pvalue:.2g}")
                out[f"{side}_{k}_vs_{cname}"] = {"ratio_of_medians": float(a.median() / b.median()), "p": float(u.pvalue)}

    if args.per_pair_csv:
        print("\n=== Spearman: TARGET contrast vs per-pair score, ALL test pairs (negative = smooth output scores worse on high-contrast targets) ===")
        for spec in args.per_pair_csv:
            name, path = spec.split("=", 1)
            m = pd.read_csv(path)
            m = df.merge(m[["pair_id", "ssim", "structure_ssim"]], on="pair_id")
            for metric in ["ssim", "structure_ssim"]:
                for c in ["tg_luma_std", "tg_local_std"]:
                    rho, p = spearmanr(m[c], m[metric])
                    print(f"  {name:9s} {metric:15s} vs {c:13s}: rho = {rho:+.3f} (n={len(m)})")
                    out[f"spearman_{name}_{metric}_{c}"] = float(rho)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
