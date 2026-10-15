"""
Comparison of the sharpness ablation (train_img_sharp.py) with the existing models and controls on the cleaned TEST pairs (1,233).
Table 1: pooled, per method: structure SSIM, RGB SSIM, PSNR, VGG feature distance (lower = closer to the real photo), sharpness ratios (prediction/real; 1 = as sharp as the
         real photo; pooled = sum over pairs of prediction / sum of target), and the protocol deltas (structure SSIM vs Step C; RGB SSIM / PSNR vs Control 3).
Table 2: paired differences vs Step C, mean [95% plant-bootstrap CI], % pairs better, for every variant and metric.
Table 3: the same pooled table by subset (excluding Field1 day-28 pairs, Field1, Field2).
Table 4: any-gap variants only: test any-gap pairs (every earlier photo -> every later photo of the test plants) by gap length.
Usage: python src/compare_sharp.py --image-pairs data/image_pairs.parquet --outputs outputs
"""
import argparse
import os

import numpy as np
import pandas as pd

from compare_temporal import boot_ci

METHODS = [("Copy-forward", "img_sharp_existing_copy_forward_per_pair.csv"), ("Control 3 (colour+blur)", "img_ctrl3_per_pair.csv"),
           ("Step C (baseline)", "img_sharp_existing_step_c_per_pair.csv"), ("ConvLSTM K=4", "img_sharp_existing_convlstm_k4_per_pair.csv"),
           ("Transformer", "img_sharp_existing_transformer_per_pair.csv"), ("+ perceptual", "img_sharp_perc_per_pair.csv"),
           ("+ residual", "img_sharp_residual_per_pair.csv"), ("+ residual + perceptual", "img_sharp_residual_perc_per_pair.csv"),
           ("+ any-gap", "img_sharp_anygap_per_pair.csv"), ("+ residual + perceptual + any-gap", "img_sharp_full_per_pair.csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--outputs", required=True)
    a = ap.parse_args()
    p = pd.read_parquet(a.image_pairs)
    te = p[p.split == "test"][["pair_id", "plant_id", "field", "target_day", "input_last_day"]].sort_values("pair_id").reset_index(drop=True)
    te["d28"] = (te.field == "Field1") & ((te.target_day == 28) | (te.input_last_day == 28))
    D = {}
    for name, f in METHODS:
        path = os.path.join(a.outputs, f)
        if not os.path.exists(path):
            print(f"(missing {f}: {name} skipped)")
            continue
        d = pd.read_csv(path).set_index("pair_id")
        assert set(d.index) == set(te.pair_id), f"{f}: not the same 1,233 test pairs ({len(d)})"
        D[name] = d.loc[te.pair_id]
    sc, c3 = D["Step C (baseline)"], D["Control 3 (colour+blur)"]
    has_sharp = {n: "grad_pred" in d.columns for n, d in D.items()}

    def pooled(d, mask):
        r = {"structure SSIM": d.structure_ssim.values[mask].mean(), "RGB SSIM": d.ssim.values[mask].mean(), "PSNR": d.psnr.values[mask].mean()}
        if "vgg_dist" in d.columns:
            r["VGG dist"] = d.vgg_dist.values[mask].mean()
            r["grad ratio"] = d.grad_pred.values[mask].sum() / d.grad_target.values[mask].sum()
            r["lap ratio"] = d.lap_pred.values[mask].sum() / d.lap_target.values[mask].sum()
        return r

    def table(mask, title):
        print(f"\n{title}")
        print("| Method | Structure SSIM (d vs Step C) | RGB SSIM (d vs Control 3) | PSNR dB (d vs Control 3) | VGG distance (lower better) | Sharpness: gradient ratio | Sharpness: Laplacian ratio |")
        print("|---|---|---|---|---|---|---|")
        ref_sc, ref_c3 = pooled(sc, mask), pooled(c3, mask)
        for n, d in D.items():
            r = pooled(d, mask)
            f = lambda k, nd: f"{r[k]:.{nd}f}" if k in r else "n/a"  # noqa: E731
            dss = "" if n in ("Step C (baseline)", "Control 3 (colour+blur)") else f" ({r['structure SSIM'] - ref_sc['structure SSIM']:+.4f})"
            drg = "" if n == "Control 3 (colour+blur)" else f" ({r['RGB SSIM'] - ref_c3['RGB SSIM']:+.3f})"
            dps = "" if n == "Control 3 (colour+blur)" else f" ({r['PSNR'] - ref_c3['PSNR']:+.2f})"
            print(f"| {n} | {r['structure SSIM']:.4f}{dss} | {r['RGB SSIM']:.4f}{drg} | {r['PSNR']:.2f}{dps} | {f('VGG dist', 3)} | {f('grad ratio', 3)} | {f('lap ratio', 4)} |")

    allm = np.ones(len(te), bool)
    table(allm, "TABLE 1 -- pooled test (1,233 pairs); sharpness ratio: 1 = as sharp as the real photo, <1 blurrier")
    print("\nTABLE 2 -- paired difference vs Step C (variant minus Step C), mean [95% plant-bootstrap CI], % of pairs where the variant is better")
    print("| Variant | Structure SSIM | RGB SSIM | PSNR dB | VGG distance (negative = closer) |\n|---|---|---|---|---|")
    pl = te.plant_id.values
    for n, d in D.items():
        if n in ("Step C (baseline)", "Control 3 (colour+blur)", "Copy-forward") or "vgg_dist" not in d.columns:
            continue
        cells = []
        for k, nd, better in (("structure_ssim", 4, 1), ("ssim", 4, 1), ("psnr", 3, 1), ("vgg_dist", 3, -1)):
            x = d[k].values - sc[k].values if k in sc.columns else None
            if x is None or np.isnan(x).all():
                cells.append("n/a")
                continue
            lo, hi = boot_ci(x, pl)
            cells.append(f"{x.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}], {((x * better) > 0).mean():.0%}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    table(~te.d28.values, "TABLE 3a -- excluding Field1 day-28 pairs")
    table((te.field == "Field1").values & ~te.d28.values, "TABLE 3b -- Field1 excluding day 28")
    table((te.field == "Field2").values, "TABLE 3c -- Field2")
    print("\nTABLE 4 -- any-gap variants on the test plants' any-gap pairs (every earlier photo -> every later photo), by gap length")
    for n, f in (("+ any-gap", "img_sharp_anygap_anygap_per_pair.csv"), ("+ residual + perceptual + any-gap", "img_sharp_full_anygap_per_pair.csv")):
        path = os.path.join(a.outputs, f)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        b = pd.cut(d.gap_days, [0, 7, 14, 28, 56, 100], labels=["<=7", "8-14", "15-28", "29-56", "57+"])
        g = d.groupby(b, observed=True).agg(pairs=("pair_id", "size"), structure_ssim=("structure_ssim", "mean"), ssim=("ssim", "mean"), psnr=("psnr", "mean"),
                                            vgg=("vgg_dist", "mean"), gsum_p=("grad_pred", "sum"), gsum_t=("grad_target", "sum"))
        g["grad_ratio"] = g.gsum_p / g.gsum_t
        print(f"\n{n}: {len(d)} pairs; gap (days) bucket:")
        print(g[["pairs", "structure_ssim", "ssim", "psnr", "vgg", "grad_ratio"]].round(4).to_string())


if __name__ == "__main__":
    main()
