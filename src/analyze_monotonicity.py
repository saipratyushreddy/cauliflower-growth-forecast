"""
Check whether test-set diameter trajectories are mostly monotonically
increasing, and whether the CNN-LSTM's error concentrates on the
non-monotonic (plateau/decline) segments specifically.

A "declining segment" for a given pair is defined as: target_diameter <
y_t (the diameter at the pair's own last input date), i.e. the true value
actually went down (or, for "non-monotonic", stayed flat/down) relative to
the most recent known measurement. This uses the same y_t lookup logic as
Step 5's persistence baseline (metadata.parquet), so it's directly
comparable to that baseline's own eligible set.

Usage:
    python src/analyze_monotonicity.py \
        --pairs-split data/pairs_split.parquet \
        --metadata data/metadata.parquet \
        --cnn-lstm-predictions outputs/step6_cnn_lstm_test_predictions.csv
"""
import argparse

import pandas as pd


def make_pair_id(df):
    return df["plant_id"].astype(str) + "::" + df["target_day"].astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--cnn-lstm-predictions", required=True)
    args = ap.parse_args()

    pairs_df = pd.read_parquet(args.pairs_split)
    pairs_df["pair_id"] = make_pair_id(pairs_df)
    meta_df = pd.read_parquet(args.metadata)
    lookup = meta_df.set_index(["plant_id", "day_after_planting"])["diameter"]

    test_df = pairs_df[pairs_df["split"] == "test"].copy()

    y_t_values = []
    for _, row in test_df.iterrows():
        key = (row["plant_id"], row["last_input_day"])
        v = lookup.loc[key] if key in lookup.index else None
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        y_t_values.append(v)
    test_df["y_t"] = y_t_values
    test_df = test_df[test_df["y_t"].notna()].copy()

    test_df["is_decline_or_flat"] = test_df["target_diameter"] <= test_df["y_t"]
    test_df["is_decline_strict"] = test_df["target_diameter"] < test_df["y_t"]
    test_df["delta"] = test_df["target_diameter"] - test_df["y_t"]

    print(f"Test pairs with a valid y_t lookup: {len(test_df)}")
    print(f"Pairs where target <= y_t (flat or declining): "
          f"{test_df['is_decline_or_flat'].sum()} / {len(test_df)} "
          f"({100*test_df['is_decline_or_flat'].mean():.1f}%)")
    print(f"Pairs where target < y_t (strictly declining): "
          f"{test_df['is_decline_strict'].sum()} / {len(test_df)} "
          f"({100*test_df['is_decline_strict'].mean():.1f}%)")

    # Per-plant: does EVERY test plant have at least one such segment, or is it a subset?
    plant_level = test_df.groupby("plant_id")["is_decline_or_flat"].any()
    print(f"\nDistinct test plants with >=1 flat-or-declining segment: "
          f"{plant_level.sum()} / {plant_level.shape[0]} "
          f"({100*plant_level.mean():.1f}%)")

    # Same check but over the FULL trajectory (metadata), not just test pairs --
    # i.e. does this plant's true diameter ever go down anywhere in its whole
    # measured history, train+val+test plants combined, to see if this is a
    # trait-wide property, not just a test-set artifact.
    print("\n--- Same check over ALL plants' full measured trajectories (train+val+test) ---")
    all_declines = 0
    all_plants_with_decline = 0
    all_plants_total = 0
    for plant_id, g in meta_df.groupby("plant_id"):
        g = g.sort_values("day_after_planting")
        vals = g["diameter"].dropna().values
        if len(vals) < 2:
            continue
        all_plants_total += 1
        diffs = vals[1:] - vals[:-1]
        n_decline = (diffs < 0).sum()
        all_declines += n_decline
        if n_decline > 0:
            all_plants_with_decline += 1
    print(f"Plants with >=1 consecutive-measurement decline anywhere in their full history: "
          f"{all_plants_with_decline} / {all_plants_total} "
          f"({100*all_plants_with_decline/all_plants_total:.1f}%)")

    # ---- Merge with CNN-LSTM predictions to compare error on decline vs growth segments ----
    preds_df = pd.read_csv(args.cnn_lstm_predictions)
    merged = test_df.merge(preds_df[["pair_id", "y_true", "y_pred", "abs_error"]], on="pair_id", how="inner")
    print(f"\nMerged with CNN-LSTM predictions: {len(merged)} / {len(test_df)} test pairs matched")

    growth_mae = merged[~merged["is_decline_or_flat"]]["abs_error"].mean()
    decline_mae = merged[merged["is_decline_or_flat"]]["abs_error"].mean()
    strict_decline_mae = merged[merged["is_decline_strict"]]["abs_error"].mean()
    n_growth = (~merged["is_decline_or_flat"]).sum()
    n_decline = merged["is_decline_or_flat"].sum()
    n_strict_decline = merged["is_decline_strict"].sum()

    print(f"\nCNN-LSTM MAE on GROWTH segments (target > y_t):     N={n_growth}, MAE={growth_mae:.3f}")
    print(f"CNN-LSTM MAE on FLAT-OR-DECLINE segments (target <= y_t): N={n_decline}, MAE={decline_mae:.3f}")
    print(f"CNN-LSTM MAE on STRICTLY DECLINING segments (target < y_t): N={n_strict_decline}, MAE={strict_decline_mae:.3f}")
    if n_growth > 0 and n_decline > 0:
        ratio = decline_mae / growth_mae
        print(f"\nFlat-or-decline segments have {ratio:.2f}x the MAE of growth segments.")


if __name__ == "__main__":
    main()
