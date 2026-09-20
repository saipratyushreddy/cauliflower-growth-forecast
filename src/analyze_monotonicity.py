"""
Check whether test-set diameter trajectories are mostly monotonically
increasing, and whether the CNN-LSTM's error concentrates on the
non-monotonic (plateau/decline) segments specifically.

A "flat-or-decline segment" for a given pair is defined EXACTLY as:
    target_diameter <= y_t
(delta = target_diameter - y_t <= 0), where y_t is the diameter at the
pair's own last input date (looked up from metadata.parquet, same y_t
used by Step 5's persistence baseline). "Strictly declining" is the
stricter target_diameter < y_t (delta < 0). Both thresholds are exact
equality/inequality on the raw measured values -- no noise band or
smoothing is applied, so a delta of e.g. -0.5mm is counted identically to
a delta of -20mm under "flat-or-decline". The magnitude breakdown below
(delta histogram) is reported specifically so a reader can judge how much
of this is small measurement jitter around a plateau vs. large true
declines, rather than treating the count alone as evidence of either.

Also checks whether these segments cluster late-season (by day_after_planting
and by normalized position in each plant's own trajectory) or are spread
evenly throughout -- this distinguishes two very different explanations:
a real late-season biological transition (senescence, head dynamics) the
model saw little training signal for, vs. measurement noise scattered
throughout the season that no model could be expected to track.

Usage:
    python src/analyze_monotonicity.py \
        --pairs-split data/pairs_split.parquet \
        --metadata data/metadata.parquet \
        --cnn-lstm-predictions outputs/step6_cnn_lstm_test_predictions.csv
"""
import argparse

import numpy as np
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

    print(f"Threshold used: 'flat-or-decline' = target_diameter <= y_t (delta <= 0mm), "
          f"'strictly declining' = target_diameter < y_t (delta < 0mm). "
          f"No noise band -- exact inequality on raw measured values.")
    print(f"\nTest pairs with a valid y_t lookup: {len(test_df)}")
    print(f"Pairs where target <= y_t (flat or declining): "
          f"{test_df['is_decline_or_flat'].sum()} / {len(test_df)} "
          f"({100*test_df['is_decline_or_flat'].mean():.1f}%)")
    print(f"Pairs where target < y_t (strictly declining): "
          f"{test_df['is_decline_strict'].sum()} / {len(test_df)} "
          f"({100*test_df['is_decline_strict'].mean():.1f}%)")

    # ---- Magnitude breakdown: how much of "flat-or-decline" is small jitter
    # vs a real, sizeable drop? ----
    decline_df = test_df[test_df["is_decline_or_flat"]]
    print(f"\n--- Magnitude of flat-or-decline deltas (mm) ---")
    print(decline_df["delta"].describe())
    for lo, hi, label in [(0, 0, "delta == 0 (exact tie)"),
                          (-2, 0, "delta in (-2, 0) -- likely measurement jitter"),
                          (-5, -2, "delta in (-5, -2]"),
                          (-np.inf, -5, "delta <= -5mm -- likely a real decline")]:
        if lo == 0 and hi == 0:
            n = (decline_df["delta"] == 0).sum()
        else:
            n = ((decline_df["delta"] > lo) & (decline_df["delta"] <= hi)).sum()
        print(f"  {label}: {n} / {len(decline_df)} ({100*n/len(decline_df):.1f}%)")

    # Per-plant: does EVERY test plant have at least one such segment, or is it a subset?
    plant_level = test_df.groupby("plant_id")["is_decline_or_flat"].any()
    print(f"\nDistinct test plants with >=1 flat-or-declining segment: "
          f"{plant_level.sum()} / {plant_level.shape[0]} "
          f"({100*plant_level.mean():.1f}%)")

    # ---- Seasonal timing: do flat-or-decline segments cluster late-season? ----
    print("\n--- Seasonal timing of flat-or-decline segments ---")
    print("By absolute day_after_planting (target_day):")
    print("  Flat-or-decline segments:")
    print(decline_df["target_day"].describe())
    print("  Growth segments (target > y_t), for comparison:")
    growth_df = test_df[~test_df["is_decline_or_flat"]]
    print(growth_df["target_day"].describe())

    # Normalized position within each plant's OWN trajectory (0=first
    # measurement, 1=last), to control for different plants having
    # different absolute day ranges/season lengths.
    def normalized_position(row, plant_max_day, plant_min_day):
        span = plant_max_day - plant_min_day
        if span == 0:
            return 0.0
        return (row["target_day"] - plant_min_day) / span

    plant_day_range = meta_df.groupby("plant_id")["day_after_planting"].agg(["min", "max"])
    test_df = test_df.merge(plant_day_range, left_on="plant_id", right_index=True)
    test_df["norm_position"] = (test_df["target_day"] - test_df["min"]) / (test_df["max"] - test_df["min"]).replace(0, np.nan)

    decline_df2 = test_df[test_df["is_decline_or_flat"]]
    growth_df2 = test_df[~test_df["is_decline_or_flat"]]
    print(f"\nNormalized position within each plant's own trajectory (0=earliest measurement, 1=latest):")
    print(f"  Flat-or-decline segments: mean={decline_df2['norm_position'].mean():.3f}, "
          f"median={decline_df2['norm_position'].median():.3f}")
    print(f"  Growth segments: mean={growth_df2['norm_position'].mean():.3f}, "
          f"median={growth_df2['norm_position'].median():.3f}")

    # Bucket by normalized position quartile to see concentration directly.
    print(f"\nFlat-or-decline segment count by normalized-position quartile of the season:")
    bins = [0, 0.25, 0.5, 0.75, 1.0001]
    labels = ["0-25% (early)", "25-50%", "50-75%", "75-100% (late)"]
    test_df["season_quartile"] = pd.cut(test_df["norm_position"], bins=bins, labels=labels, include_lowest=True)
    quartile_counts = test_df.groupby("season_quartile", observed=True)["is_decline_or_flat"].agg(["sum", "count"])
    quartile_counts["pct_decline"] = 100 * quartile_counts["sum"] / quartile_counts["count"]
    print(quartile_counts)

    # ---- Same check over ALL plants' full measured trajectories ----
    print("\n--- Same check over ALL plants' full measured trajectories (train+val+test) ---")
    all_plants_with_decline = 0
    all_plants_total = 0
    late_season_declines = 0
    early_season_declines = 0
    for plant_id, g in meta_df.groupby("plant_id"):
        g = g.sort_values("day_after_planting")
        valid = g.dropna(subset=["diameter"])
        if len(valid) < 2:
            continue
        all_plants_total += 1
        days = valid["day_after_planting"].values
        vals = valid["diameter"].values
        diffs = vals[1:] - vals[:-1]
        n_decline = (diffs <= 0).sum()
        if n_decline > 0:
            all_plants_with_decline += 1
            day_span = days[-1] - days[0] if days[-1] != days[0] else 1
            decline_positions = (days[1:][diffs <= 0] - days[0]) / day_span
            late_season_declines += (decline_positions >= 0.5).sum()
            early_season_declines += (decline_positions < 0.5).sum()
    print(f"Plants with >=1 consecutive-measurement flat-or-decline anywhere in their full history: "
          f"{all_plants_with_decline} / {all_plants_total} "
          f"({100*all_plants_with_decline/all_plants_total:.1f}%)")
    total_declines_full = late_season_declines + early_season_declines
    if total_declines_full > 0:
        print(f"Of all such declines (full population, train+val+test): "
              f"{late_season_declines} ({100*late_season_declines/total_declines_full:.1f}%) fall in the "
              f"LATE half of that plant's own trajectory, "
              f"{early_season_declines} ({100*early_season_declines/total_declines_full:.1f}%) in the EARLY half.")

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

    # Error by season timing, regardless of decline/growth, to see if late-season
    # is just harder generally vs specifically harder on declines.
    print(f"\nCNN-LSTM MAE by normalized season position quartile (all test pairs, not just declines):")
    merged["season_quartile"] = pd.cut(merged["norm_position"], bins=bins, labels=labels, include_lowest=True)
    print(merged.groupby("season_quartile", observed=True)["abs_error"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
