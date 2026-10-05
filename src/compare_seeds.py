"""
Seed-to-seed check for the K=4 ConvLSTM (single confirmatory run, not a sweep): is the difference between two
K=4 runs that differ ONLY in random seed within the ~0.002-0.004 structure-SSIM band seen between K=4 and K=1 on
identical-input (1-history-frame) pairs?

Reports, on the same cleaned test pairs and protocol as compare_temporal.py:
  * seed-to-seed deltas (K=4 seed B minus seed A) pooled / per field / with-without day 28 / the 1-frame pairs,
    mean (median), plant-bootstrap CI, on structure SSIM, RGB SSIM and PSNR;
  * an explicit yes/no for the pooled structure-SSIM delta against the stated band (|delta| <= 0.004);
  * the history effect re-expressed against the seed-noise reference: K=4 (each seed, and their average) minus K=1.

Usage:
    python src/compare_seeds.py --image-pairs data/image_pairs.parquet --k1 outputs/img_convlstm_k1_per_pair.csv \
        --k4-a outputs/img_convlstm_k4_per_pair.csv --k4-b outputs/img_convlstm_k4_seed43_per_pair.csv \
        --step-c outputs/img_single_frame_per_pair.csv --control3 outputs/img_ctrl3_per_pair.csv
"""
import argparse

import numpy as np
import pandas as pd

from compare_temporal import METRICS, boot_ci, load
from summarize_baseline import strata

BAND = 0.004


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--k4-a", required=True, help="K=4, seed 42 (the original run)")
    ap.add_argument("--k4-b", required=True, help="K=4, second seed")
    ap.add_argument("--step-c", required=True)
    ap.add_argument("--control3", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    te = pairs[pairs.split == "test"][["pair_id", "plant_id", "field", "input_last_day", "target_day", "n_input_frames"]]
    te = te.sort_values("pair_id").reset_index(drop=True)
    ids = set(te.pair_id)
    te["d28"] = (te.field == "Field1") & ((te.input_last_day == 28) | (te.target_day == 28))
    M = {"K1": load(args.k1, ids), "A": load(args.k4_a, ids), "B": load(args.k4_b, ids),
         "StepC": load(args.step_c, ids), "C3": load(args.control3, ids)}
    for k, v in M.items():
        assert list(v.index) == list(te.pair_id)
    val = lambda n, m: pd.Series(M[n][m].values, index=te.index)  # noqa: E731
    mean_ab = {m: (val("A", m) + val("B", m)) / 2 for m, _, _ in METRICS}

    print(f"{len(te)} test pairs. Test means (structure SSIM / RGB SSIM / PSNR):")
    for n, lab in [("StepC", "Step C"), ("K1", "K=1"), ("A", "K=4 seed 42"), ("B", "K=4 seed 2")]:
        print(f"  {lab:12s} {M[n]['structure_ssim'].mean():.4f} / {M[n]['ssim'].mean():.4f} / {M[n]['psnr'].mean():.2f}")

    print("\nTABLE S1 -- seed-to-seed: K=4 seed 2 minus K=4 seed 42 (paired), mean (median) [95% plant-bootstrap CI], % pairs seed 2 better")
    print("| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |\n|---|---|---|---|---|")
    d = te.copy()
    for m, _, _ in METRICS:
        d[m] = val("B", m) - val("A", m)
    pooled_delta = None
    for name, g in strata(d).items():
        cells = []
        for m, _, nd in METRICS:
            lo, hi = boot_ci(g[m].values, g["plant_id"].values)
            cells.append(f"{g[m].mean():+.{nd}f} ({g[m].median():+.{nd}f}) [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[m] > 0).mean():.0%}")
        if name == "pooled":
            pooled_delta = g["structure_ssim"].mean()
        print(f"| {name.strip()} | {len(g):,} | " + " | ".join(cells) + " |")
    one = d[d.n_input_frames == 1]
    cells = []
    for m, _, nd in METRICS:
        lo, hi = boot_ci(one[m].values, one["plant_id"].values)
        cells.append(f"{one[m].mean():+.{nd}f} ({one[m].median():+.{nd}f}) [{lo:+.{nd}f}, {hi:+.{nd}f}]")
    print(f"| (pairs with 1 history frame) | {len(one)} | " + " | ".join(cells) + " |")

    print(f"\nWITHIN BAND? pooled seed-to-seed structure-SSIM delta = {pooled_delta:+.4f}; stated band |delta| <= {BAND}: "
          f"{'YES' if abs(pooled_delta) <= BAND else 'NO'}")

    print("\nTABLE S2 -- the history effect against seed noise: K=4 minus K=1, pooled test, mean [95% plant-bootstrap CI]")
    print("| K=4 run | Structure SSIM | RGB SSIM | RGB PSNR dB |\n|---|---|---|---|")
    for lab, getter in [("seed 42", lambda m: val("A", m)), ("seed 2", lambda m: val("B", m)),
                        ("average of the two seeds", lambda m: mean_ab[m])]:
        cells = []
        for m, _, nd in METRICS:
            dd = (getter(m) - val("K1", m)).values
            lo, hi = boot_ci(dd, te["plant_id"].values)
            cells.append(f"{dd.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]")
        print(f"| {lab} | " + " | ".join(cells) + " |")

    print("\nTABLE S3 -- protocol: structure SSIM vs Step C and RGB SSIM / PSNR vs Control 3, pooled test, mean [95% CI]")
    print("| K=4 run | d Structure SSIM vs Step C | d RGB SSIM vs Control 3 | d RGB PSNR dB vs Control 3 |\n|---|---|---|---|")
    for lab, getter in [("seed 42", lambda m: val("A", m)), ("seed 2", lambda m: val("B", m))]:
        cells = []
        for m, ref, nd in [("structure_ssim", "StepC", 3), ("ssim", "C3", 3), ("psnr", "C3", 2)]:
            dd = (getter(m) - val(ref, m)).values
            lo, hi = boot_ci(dd, te["plant_id"].values)
            cells.append(f"{dd.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]")
        print(f"| {lab} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
