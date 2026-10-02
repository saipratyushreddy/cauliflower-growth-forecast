"""
Blur/corruption scan: variance-of-Laplacian (VoL) for every image in
metadata.parquet (9,377 images; the 8,638 figure elsewhere is PAIRS, not images).

VoL is computed on the 256x256 grayscale (same resize as the image track) so
the two fields are on a comparable scale. Raw VoL depends on content (bare-soil
seedling frames are legitimately smoother than dense canopy), so outliers are
flagged WITHIN (field, day_after_planting) groups: an image is flagged if its VoL
is in the bottom --pct percent of its group. Known degenerate black frames
(--exclude-list) are reported separately and left out of the percentile
computation so they don't distort it. Also reports file size and the
within-group z-score of log(VoL) to help rank.

Usage:
    python src/blur_scan.py --metadata data/metadata.parquet \
        --images-root $WORK/cauliflower-growth-forecast/data/images \
        --exclude-list outputs/tiny_images_md5.txt --out-dir outputs
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import laplace

SIZE = 256


def vol(images_root, fp):
    with Image.open(os.path.join(images_root, fp)) as im:
        im = im.convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.LANCZOS)
        return float(laplace(np.asarray(im, dtype=np.float64)).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--exclude-list", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pct", type=float, default=1.0)
    ap.add_argument("--min-group", type=int, default=20, help="skip percentile flagging for groups smaller than this")
    args = ap.parse_args()

    meta = pd.read_parquet(args.metadata)[["plant_id", "field", "acquisition_date",
                                           "day_after_planting", "filepath"]].copy()
    known_bad = set()
    if args.exclude_list:
        known_bad = set(pd.read_csv(args.exclude_list, sep=r"\s+", header=None, names=["md5", "fp"])["fp"])
    meta["known_degenerate"] = meta["filepath"].isin(known_bad)

    vols, sizes = [], []
    for i, fp in enumerate(meta["filepath"]):
        try:
            vols.append(vol(args.images_root, fp))
        except Exception as e:
            print(f"FAILED {fp}: {e}", flush=True)
            vols.append(np.nan)
        sizes.append(os.path.getsize(os.path.join(args.images_root, fp)))
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(meta)}", flush=True)
    meta["vol"] = vols
    meta["file_bytes"] = sizes
    print(f"Scanned {meta['vol'].notna().sum()}/{len(meta)} images; "
          f"{meta['known_degenerate'].sum()} known degenerate (reported separately)")

    ok = meta[~meta["known_degenerate"] & meta["vol"].notna()].copy()
    ok["logvol"] = np.log(ok["vol"].clip(lower=1e-6))
    g = ok.groupby(["field", "day_after_planting"])["logvol"]
    ok["z_in_group"] = (ok["logvol"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    ok["group_pct_rank"] = g.rank(pct=True) * 100
    ok["group_n"] = g.transform("size")
    ok["flagged"] = (ok["group_pct_rank"] <= args.pct) & (ok["group_n"] >= args.min_group)

    print("\nVoL distribution by field (non-degenerate images):")
    print(ok.groupby("field")["vol"].describe(percentiles=[.01, .05, .5, .95]).round(1).to_string())
    print("\nVoL median by field x day (shows content dependence):")
    print(ok.groupby(["field", "day_after_planting"])["vol"].median().round(0).unstack(0).to_string())
    print("\nKnown degenerate frames, VoL summary:")
    print(meta[meta.known_degenerate]["vol"].describe().round(2).to_string())

    fl = ok[ok["flagged"]].sort_values("z_in_group")
    print(f"\nFLAGGED: bottom {args.pct}% within (field, day) -> {len(fl)} images "
          f"({fl['plant_id'].nunique()} plants). Lowest 40 by z-score:")
    cols = ["plant_id", "field", "acquisition_date", "day_after_planting", "vol", "z_in_group", "file_bytes", "filepath"]
    print(fl[cols].head(40).round(2).to_string(index=False))
    print("\nFlagged count by field/day:")
    print(fl.groupby(["field", "day_after_planting"]).size().to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    meta.merge(ok[["filepath", "z_in_group", "group_pct_rank", "flagged"]], on="filepath", how="left") \
        .to_csv(os.path.join(args.out_dir, "img_blur_scan_all.csv"), index=False)
    fl[cols].to_csv(os.path.join(args.out_dir, "img_blur_scan_flagged.csv"), index=False)
    print(f"\nWrote {args.out_dir}/img_blur_scan_all.csv and img_blur_scan_flagged.csv")


if __name__ == "__main__":
    main()
