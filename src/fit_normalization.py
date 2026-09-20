"""
Step 3d: Fit normalization parameters on TRAINING PLANTS ONLY, then apply
those fixed parameters to val/test unchanged.

What gets normalized in this baseline:
  - target_diameter: z-score normalized using train-split mean/std, so the
    LSTM regresses a well-scaled target. Predictions are de-normalized
    before computing MAE/RMSE (those metrics are reported in raw mm units).
  - day_after_planting (used as a continuous per-timestep feature in Step 6):
    z-score normalized the same way.

CNN embeddings themselves (Step 4) use a frozen, off-the-shelf ImageNet-
pretrained ResNet18 with the standard ImageNet mean/std for input
normalization -- that normalization is a fixed, dataset-independent
constant from the pretraining source, not something fit on this dataset,
so it does not leak any GrowliFlower information and is applied identically
to train/val/test images.

This script only handles the trait-value and day-count normalization.
It must be run AFTER split_data.py so train/val/test assignment already
exists in the pairs table.

Usage:
    python src/fit_normalization.py --pairs-split data/pairs_split.parquet --out data/norm_stats.json
"""
import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.pairs_split)
    train_df = df[df["split"] == "train"]

    # --- Target diameter stats (fit on train targets only) ---
    target_mean = float(train_df["target_diameter"].mean())
    target_std = float(train_df["target_diameter"].std())

    # --- day_after_planting stats (fit on train INPUT days only, i.e. every
    # day value that appears inside any training pair's input sequence --
    # never target-date days, and never any val/test plant's days) ---
    all_train_input_days = np.concatenate(
        [np.array(d, dtype=float) for d in train_df["input_days_after_planting"]]
    )
    day_mean = float(all_train_input_days.mean())
    day_std = float(all_train_input_days.std())

    stats = {
        "target_diameter_mean": target_mean,
        "target_diameter_std": target_std,
        "day_after_planting_mean": day_mean,
        "day_after_planting_std": day_std,
        "fit_on_n_train_plants": int(train_df["plant_id"].nunique()),
        "fit_on_n_train_pairs": int(len(train_df)),
        "image_normalization": {
            "note": "Frozen ImageNet-pretrained ResNet18 uses fixed ImageNet "
                    "mean/std (not fit on GrowliFlower data).",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }

    print("Normalization stats (fit on TRAIN plants only):")
    print(json.dumps(stats, indent=2))

    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote normalization stats to {args.out}")


if __name__ == "__main__":
    main()
