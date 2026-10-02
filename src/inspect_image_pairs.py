"""
Diagnostic for the copy-forward baseline: are the very low SSIM/PSNR values
real, or caused by pairing/alignment/duplicate-file issues?

  1. Counts pairs with pixel-identical input and target (PSNR=inf), per field.
  2. Saves figures of test pairs (input | target | |diff|) for each field:
     random pairs plus highest- and lowest-SSIM pair.
  3. Prints the (day, filepath, size, md5-prefix) sequence for a few plants so
     date ordering and duplicate files are visible.

Usage:
    python src/inspect_image_pairs.py --image-pairs data/image_pairs.parquet \
        --per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv \
        --images-root $WORK/cauliflower-growth-forecast/data/images --out-dir outputs
"""
import argparse
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from image_copy_forward_baseline import SIZE


def load(root, fp):
    im = Image.open(os.path.join(root, fp)).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im)


def md5(root, fp):
    with open(os.path.join(root, fp), "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--per-pair-csv", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    res = pd.read_csv(args.per_pair_csv)
    df = pairs.merge(res[["pair_id", "ssim", "psnr"]], on="pair_id")

    ident = df[df["psnr"].isna()]
    print(f"Pixel-identical (PSNR=inf) pairs: {len(ident)} / {len(df)}")
    print(df.assign(identical=df["psnr"].isna()).groupby(["split", "field"])["identical"]
          .agg(["sum", "count"]).to_string())
    print("Identical pairs by plant (top 10):")
    print(ident["plant_id"].value_counts().head(10).to_string())

    test = df[df["split"] == "test"]
    rng = np.random.RandomState(args.seed)
    for field, g in test.groupby("field"):
        picks = [("random", r) for _, r in g.sample(args.n_random, random_state=rng).iterrows()]
        picks.append(("best_ssim", g.loc[g["ssim"].idxmax()]))
        picks.append(("worst_ssim", g.loc[g["ssim"].idxmin()]))
        fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
        for row, (tag, r) in zip(axes, picks):
            a = load(args.images_root, r["input_last_filepath"])
            b = load(args.images_root, r["target_filepath"])
            d = np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)
            for ax, im, t in zip(row, [a, b, d], [f"input day {r['input_last_day']}",
                                                  f"target day {r['target_day']}", "|diff|"]):
                ax.imshow(im); ax.set_title(t, fontsize=8); ax.axis("off")
            row[0].set_ylabel(tag)
            row[1].set_title(f"target day {r['target_day']}  {tag}\n{r['plant_id']}  "
                             f"SSIM={r['ssim']:.3f} PSNR={r['psnr']:.1f}", fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"img_diag_pairs_{field}.png")
        fig.savefig(out, dpi=90); plt.close(fig)
        print(f"Wrote {out}")

    meta_pairs = pairs.sort_values("target_day").drop_duplicates("plant_id", keep="last")  # last pair = full sequence
    for _, r in pd.concat([meta_pairs[meta_pairs.field == f].sample(min(2, (meta_pairs.field == f).sum()), random_state=1)
                           for f in meta_pairs.field.unique()]).iterrows():
        print(f"\nSequence for {r['plant_id']} ({r['field']}):")
        seq = list(zip(r["input_days"], r["input_filepaths"])) + [(r["target_day"], r["target_filepath"])]
        for day, fp in seq:
            print(f"  day {day:3d}  {fp}  {os.path.getsize(os.path.join(args.images_root, fp)):>8d}B  "
                  f"md5={md5(args.images_root, fp)}")


if __name__ == "__main__":
    main()
