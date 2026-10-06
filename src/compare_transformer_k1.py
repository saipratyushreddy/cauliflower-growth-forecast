"""
Transformer-stage decomposition and full comparison on the cleaned TEST pairs.

  Table A  all models, structure SSIM mean (median), by subset (primary metric) -- Step C, ConvLSTM K=1, K=4 (2 seeds),
           Transformer K=1 (single frame), Transformer (full history).
  Table B  pooled, all three metrics for every model, with differences against Step C (structure SSIM) and Control 3
           (RGB SSIM / PSNR) in ONE table, per the protocol.
  Table C  three-way decomposition, paired (first minus second; mean (median) [95% plant-bootstrap CI], % pairs first better):
           (1) Transformer K=1 vs Step C [architecture alone]; (2) Transformer vs Transformer K=1 [history alone, same
           architecture]; (3) Transformer vs Step C [combined].
  Table D  (2) by number of history frames. The 1-frame row has identical information AND identical architecture, so it
           measures pure training-run noise.
  Table E  early-season Field2 targets (days 16, 22) vs all other test pairs: Transformer minus Transformer K=1,
           Transformer minus ConvLSTM K=4, Transformer K=1 minus Step C.

Usage:
    python src/compare_transformer_k1.py --image-pairs data/image_pairs.parquet --step-c ... --control3 ... --k1 ... \
        --k4 ... --k4-seed2 ... --tf ... --tf-k1 ...
"""
import argparse

import numpy as np
import pandas as pd

from compare_temporal import METRICS, boot_ci, load
from summarize_baseline import strata


def fmt(g, m, nd, ci=True):
    lo, hi = boot_ci(g[m].values, g["plant_id"].values)
    s = f"{g[m].mean():+.{nd}f} ({g[m].median():+.{nd}f})"
    return s + (f" [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[m] > 0).mean():.0%}" if ci else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    for a in ["step-c", "control3", "k1", "k4", "k4-seed2", "tf", "tf-k1"]:
        ap.add_argument(f"--{a}", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    te = pairs[pairs.split == "test"][["pair_id", "plant_id", "field", "input_last_day", "target_day", "n_input_frames"]]
    te = te.sort_values("pair_id").reset_index(drop=True)
    ids = set(te.pair_id)
    te["d28"] = (te.field == "Field1") & ((te.input_last_day == 28) | (te.target_day == 28))
    P = {"SC": args.step_c, "C3": args.control3, "K1": args.k1, "K4": args.k4, "K4b": args.k4_seed2, "TF": args.tf, "TFK1": args.tf_k1}
    M = {k: load(v, ids) for k, v in P.items()}
    for v in M.values():
        assert list(v.index) == list(te.pair_id)
    fr = lambda n, m: pd.Series(M[n][m].values, index=te.index)  # noqa: E731
    lab = {"SC": "Step C", "C3": "Control 3", "K1": "ConvLSTM K=1", "K4": "ConvLSTM K=4 (s42)", "K4b": "ConvLSTM K=4 (s43)",
           "TFK1": "Transformer K=1", "TF": "Transformer (full)"}
    print(f"{len(te)} test pairs, identical across all models\n")

    order = ["SC", "K1", "K4", "K4b", "TFK1", "TF"]
    base = te.copy()
    for n in order:
        base[f"s|{n}"] = fr(n, "structure_ssim")
    S = strata(base)
    print("TABLE A -- structure SSIM, test, mean (median)")
    print("| Subset | Pairs | " + " | ".join(lab[n] for n in order) + " |\n|---|---|" + "---|" * len(order))
    for name, g in S.items():
        print(f"| {name.strip()} | {len(g):,} | " + " | ".join(f"{g[f's|{n}'].mean():.3f} ({g[f's|{n}'].median():.3f})" for n in order) + " |")

    print("\nTABLE B -- pooled test, one table: structure SSIM (diff vs Step C), RGB SSIM and PSNR (diff vs Control 3)")
    print("| Model | Structure SSIM | d vs Step C | RGB SSIM | d vs Control 3 | PSNR dB | d vs Control 3 |\n|---|---|---|---|---|---|---|")
    for n in ["C3"] + order:
        s, r, p = (M[n][m].mean() for m in ("structure_ssim", "ssim", "psnr"))
        ds = "" if n in ("SC", "C3") else f"{s - M['SC']['structure_ssim'].mean():+.4f}"
        dr = "" if n == "C3" else f"{r - M['C3']['ssim'].mean():+.4f}"
        dp = "" if n == "C3" else f"{p - M['C3']['psnr'].mean():+.2f}"
        print(f"| {lab[n]} | {s:.4f} | {ds} | {r:.4f} | {dr} | {p:.2f} | {dp} |")

    def table(title, a, b):
        print(f"\n{title}")
        print("| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |\n|---|---|---|---|---|")
        d = te.copy()
        for m, _, _ in METRICS:
            d[m] = fr(a, m) - fr(b, m)
        for name, g in strata(d).items():
            print(f"| {name.strip()} | {len(g):,} | " + " | ".join(fmt(g, m, nd) for m, _, nd in METRICS) + " |")

    table("TABLE C(1) Transformer K=1 vs Step C  [architecture alone]", "TFK1", "SC")
    table("TABLE C(2) Transformer (full) vs Transformer K=1  [history alone, same architecture]", "TF", "TFK1")
    table("TABLE C(3) Transformer (full) vs Step C  [combined]", "TF", "SC")

    print("\nTABLE D -- Transformer (full) minus Transformer K=1 by history frames (1 frame = same info, same architecture: training-noise floor)")
    print("| History frames | Pairs | d Structure SSIM | d RGB SSIM | d PSNR dB |\n|---|---|---|---|---|")
    d = te.copy()
    for m, _, _ in METRICS:
        d[m] = fr("TF", m) - fr("TFK1", m)
    b = pd.cut(d.n_input_frames, [0, 1, 2, 3, 5, 9, 14], labels=["1", "2", "3", "4-5", "6-9", "10-14"])
    for l, g in d.groupby(b, observed=True):
        print(f"| {l} | {len(g):,} | " + " | ".join(fmt(g, m, nd) for m, _, nd in METRICS) + " |")

    print("\nTABLE E -- early-season Field2 targets (days 16, 22) vs all other test pairs, structure SSIM difference, mean [95% CI]")
    early = ((te.field == "Field2") & te.target_day.isin([16, 22])).values
    print("| Comparison | Early Field2 (n=%d) | All other (n=%d) |\n|---|---|---|" % (early.sum(), (~early).sum()))
    for title, a, bb in [("Transformer - Transformer K=1", "TF", "TFK1"), ("Transformer - ConvLSTM K=4 (s42)", "TF", "K4"),
                         ("Transformer - ConvLSTM K=4 (s43)", "TF", "K4b"), ("ConvLSTM K=4 (s42) - ConvLSTM K=1", "K4", "K1"),
                         ("Transformer K=1 - Step C", "TFK1", "SC")]:
        dd = (fr(a, "structure_ssim") - fr(bb, "structure_ssim")).values
        cells = []
        for mask in (early, ~early):
            lo, hi = boot_ci(dd[mask], te.plant_id.values[mask])
            cells.append(f"{dd[mask].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"| {title} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
