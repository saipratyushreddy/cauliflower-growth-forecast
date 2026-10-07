"""
Field+date structural control ("generic plant at this date"): tests whether the early-season Field2 gain could come from
identifying the target DATE rather than from per-plant information.

For every test pair, the prediction is the pixel-wise MEAN, over all TRAINING plants, of those plants' actual images at the pair's
(field, target_day) (one image per plant and date; rounded to uint8). It uses no information about the test pair at all except its
field and target date, and nothing from the test split: the structural analog of Control 2's colour-only transform. Computed for
ALL test pairs (the 146-pair early Field2 subset, target days 16/22, and everything else), scored ONCE with the standard skimage
RGB SSIM / PSNR + structure SSIM (train_img_single_frame.score_pairs).

Writes <out-dir>/img_ctrl_genericdate_per_pair.csv (+ n_train_images per pair) and a figure of the mean image per early date next
to two real test targets (illustration only).

Usage:
    python src/generic_date_control.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_img_single_frame import load_rows, score_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-train", type=int, default=10, help="minimum training images required for a group's mean")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    tr = pairs[pairs.split == "train"]
    te = pairs[pairs.split == "test"].reset_index(drop=True)

    means, counts = {}, {}
    for (field, day), g in tr.groupby(["field", "target_day"]):
        rows = g.drop_duplicates("plant_id")["row_tg"].astype(int).values   # one image per training plant at this date
        assert len(rows) == g["plant_id"].nunique()
        means[(field, day)] = np.rint(arr[rows].astype(np.float64).mean(0)).astype(np.uint8)
        counts[(field, day)] = len(rows)

    preds, n_img = [], []
    for r in te.itertuples(index=False):
        key = (r.field, r.target_day)
        assert key in means and counts[key] >= args.min_train, f"no usable training mean for {key} ({counts.get(key, 0)} images)"
        preds.append(means[key])
        n_img.append(counts[key])
    print(f"Generic-date means built from TRAIN plants only: {len(means)} (field, target_day) groups; training images per group "
          f"min {min(counts.values())}, median {int(np.median(list(counts.values())))}, max {max(counts.values())}", flush=True)
    for d in (16, 22):
        if ("Field2", d) in counts:
            print(f"  Field2 target_day {d}: {counts[('Field2', d)]} training images", flush=True)

    print(f"=== TEST EVALUATION (generic-date control): {len(te)} pairs ===", flush=True)
    res = score_pairs(np.stack(preds), arr, te)
    res["n_train_images"] = n_img
    res.to_csv(os.path.join(args.out_dir, f"{args.tag}img_ctrl_genericdate_per_pair.csv"), index=False)
    print(res[["ssim", "psnr", "structure_ssim"]].agg(["mean", "median"]).round(4).to_string(), flush=True)
    early = ((te.field == "Field2") & te.target_day.isin([16, 22])).values
    if early.any():
        print("Early Field2 subset (targets days 16, 22), n=%d:" % early.sum())
        print(res[early][["ssim", "psnr", "structure_ssim"]].agg(["mean", "median"]).round(4).to_string(), flush=True)

    # illustration only
    days = [d for d in (16, 22) if ("Field2", d) in means]
    if days:
        fig, axes = plt.subplots(len(days), 3, figsize=(8, 2.7 * len(days)), squeeze=False)
        for row in axes:
            for ax in row:
                ax.axis('off')
        rng = np.random.RandomState(0)
        for i, d in enumerate(days):
            idx = np.where(((te.field == "Field2") & (te.target_day == d)).values)[0]
            pick = rng.choice(idx, min(2, len(idx)), replace=False)
            axes[i][0].imshow(means[("Field2", d)]); axes[i][0].set_title(f"mean of {counts[('Field2', d)]} training images, day {d}", fontsize=7)
            for j, k in enumerate(pick):  # one or two real test targets
                axes[i][1 + j].imshow(arr[int(te.row_tg.iloc[k])])
                axes[i][1 + j].set_title(f"real test target {te.plant_id.iloc[k]} d{d}\nstructure SSIM of the mean = {res.structure_ssim.iloc[k]:.3f}", fontsize=6.5)
            for ax in axes[i]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{args.tag}img_ctrl_genericdate_examples.png"), dpi=85)


if __name__ == "__main__":
    main()
