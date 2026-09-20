"""
Step 3b: Construct (input_sequence, target) forecasting pairs from the
metadata table.

Definition (per-plant, per the approved t -> t+1 spec):
  For a plant with acquisition dates d_1 < d_2 < ... < d_n (day_after_planting
  order), and valid (non-missing, cleaned) diameter measurements at some
  subset of those dates, each valid measurement date d_k (k >= 2, i.e. not
  the plant's very first valid measurement) becomes one target. Its
  input sequence is every image from planting through the last acquisition
  date STRICTLY BEFORE d_k (regardless of whether that date itself has a
  valid measurement) -- so images are never restricted to just the dates
  with valid labels; only the input images are used, no label information
  from d_k or later is used anywhere in the sequence.

This intentionally allows the resulting sample's input sequence to include
image dates that have no diameter measurement at all -- the trait can be
sparser than the image acquisition calendar. It also means the true
"forecast gap" (label date minus the last input image's date) can differ
from the "date-only" gap between two consecutive labeled dates reported in
Step 2, since unlabeled image dates in between still count as valid inputs.

Usage:
    python src/build_pairs.py --metadata data/metadata.parquet --out data/pairs.parquet
"""
import argparse

import pandas as pd


def build_pairs(metadata_path):
    df = pd.read_parquet(metadata_path)
    df = df.sort_values(["plant_id", "day_after_planting"]).reset_index(drop=True)

    pair_rows = []
    for plant_id, g in df.groupby("plant_id", sort=False):
        g = g.sort_values("day_after_planting").reset_index(drop=True)
        valid_idx = g.index[g["diameter"].notna()].tolist()

        # Need at least 2 valid measurements to form one forecasting pair.
        if len(valid_idx) < 2:
            continue

        # Every valid measurement after the plant's first valid one is a
        # candidate target. Input = all images at row indices < that row's
        # position in the full (sorted) per-plant sequence, i.e. every image
        # strictly before the target's acquisition date, valid-label or not.
        for target_pos in valid_idx[1:]:
            input_rows = g.iloc[:target_pos]  # strictly before target row
            if len(input_rows) == 0:
                continue  # should not happen since target_pos > 0 always here
            target_row = g.iloc[target_pos]

            pair_rows.append({
                "plant_id": plant_id,
                "field": target_row["field"],
                "input_filepaths": list(input_rows["filepath"]),
                "input_days_after_planting": list(input_rows["day_after_planting"].astype(int)),
                "input_acquisition_dates": list(input_rows["acquisition_date"]),
                "n_input_frames": len(input_rows),
                "last_input_day": int(input_rows["day_after_planting"].iloc[-1]),
                "last_input_date": input_rows["acquisition_date"].iloc[-1],
                "target_day": int(target_row["day_after_planting"]),
                "target_date": target_row["acquisition_date"],
                "target_diameter": float(target_row["diameter"]),
                "forecast_gap_days": int(target_row["day_after_planting"]) - int(input_rows["day_after_planting"].iloc[-1]),
            })

    pairs_df = pd.DataFrame(pair_rows)
    return pairs_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs_df = build_pairs(args.metadata)

    n_pairs = len(pairs_df)
    n_plants = pairs_df["plant_id"].nunique()
    print(f"Constructed {n_pairs} (input_sequence, target) pairs from {n_plants} distinct plants")
    print(f"\nInput sequence length distribution:")
    print(pairs_df["n_input_frames"].describe())
    print(f"\nForecast gap (days) distribution:")
    print(pairs_df["forecast_gap_days"].describe())
    print(f"\nPairs per plant distribution:")
    print(pairs_df.groupby("plant_id").size().describe())

    pairs_df.to_parquet(args.out, index=False)
    print(f"\nWrote pairs table to {args.out}")


if __name__ == "__main__":
    main()
