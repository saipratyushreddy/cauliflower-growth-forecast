"""
Final comparison for the image CNN-Transformer on the cleaned TEST pairs, per the project protocol: structure SSIM against the
best prior model stage (the K=4 ConvLSTM) and RGB SSIM / PSNR against Control 3, ALWAYS in the same table.

  Table 1  headline: per subset, structure SSIM [K=4 ConvLSTM | Transformer] and RGB SSIM / PSNR [Control 3 | Transformer].
  Table 2  paired differences (Transformer minus X; mean (median) [95% plant-bootstrap CI], % pairs Transformer better):
           (1) vs K=4 ConvLSTM seed 42 [headline], (2) vs K=4 ConvLSTM seed 43 [same comparison against the other seed:
           reference for training noise], (3) vs Step C, (4) vs Control 3.
  Table 3  Transformer minus K=4 ConvLSTM by history length (1 frame: same information; the Transformer should not differ
           from the ConvLSTM there except by training noise).
  Table 4  attention: weight on the LAST frame vs the uniform weight, and effective number of frames used, by history length.

Usage:
    python src/compare_transformer.py --image-pairs data/image_pairs.parquet --transformer outputs/img_transformer_per_pair.csv \
        --attn outputs/img_transformer_attn.csv --k4 outputs/img_convlstm_k4_per_pair.csv \
        --k4-seed2 outputs/img_convlstm_k4_seed43_per_pair.csv --step-c outputs/img_single_frame_per_pair.csv \
        --control3 outputs/img_ctrl3_per_pair.csv
"""
import argparse

import numpy as np
import pandas as pd

from compare_temporal import METRICS, boot_ci, load
from summarize_baseline import strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--k4", required=True)
    ap.add_argument("--k4-seed2", required=True)
    ap.add_argument("--step-c", required=True)
    ap.add_argument("--control3", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    te = pairs[pairs.split == "test"][["pair_id", "plant_id", "field", "input_last_day", "target_day", "n_input_frames"]]
    te = te.sort_values("pair_id").reset_index(drop=True)
    ids = set(te.pair_id)
    te["d28"] = (te.field == "Field1") & ((te.input_last_day == 28) | (te.target_day == 28))
    M = {"TF": load(args.transformer, ids), "K4": load(args.k4, ids), "K4b": load(args.k4_seed2, ids),
         "SC": load(args.step_c, ids), "C3": load(args.control3, ids)}
    for k, v in M.items():
        assert list(v.index) == list(te.pair_id)
    fr = lambda n, m: pd.Series(M[n][m].values, index=te.index)  # noqa: E731
    print(f"{len(te)} test pairs, identical across all models\n")

    cols = [("structure_ssim", ["K4", "TF"], ["K=4 ConvLSTM", "Transformer"]),
            ("ssim", ["C3", "TF"], ["Control 3", "Transformer"]), ("psnr", ["C3", "TF"], ["Control 3", "Transformer"])]
    base = te.copy()
    for m, ns, _ in cols:
        for n in ns:
            base[f"{m}|{n}"] = fr(n, m)
    S = strata(base)
    names = dict((a, b) for a, b, _ in METRICS)
    print("TABLE 1 -- headline (test, mean (median)): structure SSIM vs best prior model (K=4 ConvLSTM); RGB SSIM / PSNR vs Control 3")
    print("| Subset | Pairs | " + " | ".join(f"{names[m]}: {l}" for m, _, ls in cols for l in ls) + " |")
    print("|---|---|" + "---|" * 6)
    for name, g in S.items():
        cells = []
        for m, ns, _ in cols:
            nd = 2 if m == "psnr" else 3
            cells += [f"{g[f'{m}|{n}'].mean():.{nd}f} ({g[f'{m}|{n}'].median():.{nd}f})" for n in ns]
        print(f"| {name.strip()} | {len(g):,} | " + " | ".join(cells) + " |")

    for title, ref in [("(1) Transformer vs K=4 ConvLSTM, seed 42  [headline]", "K4"),
                       ("(2) Transformer vs K=4 ConvLSTM, seed 43  [other seed: training-noise reference]", "K4b"),
                       ("(3) Transformer vs Step C", "SC"), ("(4) Transformer vs Control 3", "C3")]:
        print(f"\nTABLE 2{title[:3]} {title[4:]}: Transformer minus reference, paired; mean (median) [95% plant-bootstrap CI], % pairs Transformer better")
        print("| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |\n|---|---|---|---|---|")
        d = te.copy()
        for m, _, _ in METRICS:
            d[m] = fr("TF", m) - fr(ref, m)
        for name, g in strata(d).items():
            cells = []
            for m, _, nd in METRICS:
                lo, hi = boot_ci(g[m].values, g["plant_id"].values)
                cells.append(f"{g[m].mean():+.{nd}f} ({g[m].median():+.{nd}f}) [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[m] > 0).mean():.0%}")
            print(f"| {name.strip()} | {len(g):,} | " + " | ".join(cells) + " |")

    print("\nTABLE 3 -- Transformer minus K=4 ConvLSTM (seed 42) by number of history frames")
    print("| History frames | Pairs | d Structure SSIM | d RGB SSIM | d PSNR dB |\n|---|---|---|---|---|")
    d = te.copy()
    for m, _, _ in METRICS:
        d[m] = fr("TF", m) - fr("K4", m)
    bucket = pd.cut(d.n_input_frames, [0, 1, 2, 3, 5, 9, 14], labels=["1", "2", "3", "4-5", "6-9", "10-14"])
    for lab, g in d.groupby(bucket, observed=True):
        cells = []
        for m, _, nd in METRICS:
            lo, hi = boot_ci(g[m].values, g["plant_id"].values)
            cells.append(f"{g[m].mean():+.{nd}f} ({g[m].median():+.{nd}f}) [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[m] > 0).mean():.0%}")
        print(f"| {lab} | {len(g):,} | " + " | ".join(cells) + " |")

    if args.attn:
        a = pd.read_csv(args.attn)
        a = a[a.pair_id.isin(ids)]
        print("\nTABLE 4 -- attention over history (test): weight on the LAST frame vs the uniform weight 1/n; effective number of frames used")
        print("| History frames | Pairs | Weight on last frame | Uniform 1/n | Effective frames used | Mean max weight |\n|---|---|---|---|---|---|")
        b = pd.cut(a.n_frames, [0, 1, 2, 3, 5, 9, 14], labels=["1", "2", "3", "4-5", "6-9", "10-14"])
        for lab, g in a.groupby(b, observed=True):
            print(f"| {lab} | {len(g)} | {g.w_last.mean():.3f} | {g.uniform_w.mean():.3f} | {g.eff_frames.mean():.2f} | {g.w_max.mean():.3f} |")


if __name__ == "__main__":
    main()
