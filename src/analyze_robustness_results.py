"""
Follow-up analysis on the missing-observation robustness test results
(src/robustness_missing_obs.py), addressing two questions before any
README conclusion is written:

1. Is the Transformer's advantage at 50-75% drop broadly distributed
   across test plants, or concentrated in a subset (e.g. plants with
   unusually large gaps, or a specific field/year)? Mirrors the same
   check that found 2 plants driving >50% of Step 5's worst misses.

2. Does the LSTM's degradation as a function of REMAINING OBSERVATION
   COUNT (not drop %) look smooth, or does it show a cliff specifically
   at very short remaining lengths (1-2)? A cliff at short lengths
   suggests the LSTM specifically struggles with sequence lengths never
   seen in training (Step 6 trained only on the natural, denser
   sequences, no observation-dropout augmentation) -- a narrower claim
   than "attention-based temporal modeling is inherently more robust to
   missing observations" in general.

Usage:
    python src/analyze_robustness_results.py \
        --per-pair-csv outputs/step_p2_robustness_per_pair.csv \
        --pairs-split data/pairs_split.parquet
"""
import argparse

import numpy as np
import pandas as pd


def make_pair_id(df):
    return df["plant_id"].astype(str) + "::" + df["target_day"].astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair-csv", required=True)
    ap.add_argument("--pairs-split", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.per_pair_csv)
    df["lstm_abs_error"] = (df["lstm_pred"] - df["y_true"]).abs()
    df["transformer_abs_error"] = (df["transformer_pred"] - df["y_true"]).abs()
    df["plant_id"] = df["pair_id"].str.split("::").str[0]
    df["transformer_advantage"] = df["lstm_abs_error"] - df["transformer_abs_error"]  # positive = transformer better

    pairs_df = pd.read_parquet(args.pairs_split)
    pairs_df["pair_id"] = make_pair_id(pairs_df)
    field_lookup = pairs_df.set_index("pair_id")["field"]
    df["field"] = df["pair_id"].map(field_lookup)

    # ============================================================
    # Question 1: is the Transformer's advantage population-wide or
    # concentrated in a subset of plants?
    # ============================================================
    print("=" * 70)
    print("QUESTION 1: Is the Transformer's advantage at 50-75% drop broad or concentrated?")
    print("=" * 70)

    for drop_frac in [0.5, 0.75]:
        sub = df[df["drop_frac"] == drop_frac]
        # Average each plant's advantage across its own pairs and seeds
        per_plant = sub.groupby("plant_id")["transformer_advantage"].mean().sort_values(ascending=False)
        n_plants = len(per_plant)
        n_favor_transformer = (per_plant > 0).sum()
        n_favor_lstm = (per_plant < 0).sum()

        print(f"\n--- drop={drop_frac:.0%} ---")
        print(f"Plants where Transformer has lower mean abs error: {n_favor_transformer} / {n_plants} "
              f"({100*n_favor_transformer/n_plants:.1f}%)")
        print(f"Plants where LSTM has lower mean abs error: {n_favor_lstm} / {n_plants} "
              f"({100*n_favor_lstm/n_plants:.1f}%)")

        # How much of the total advantage comes from the top few plants?
        total_advantage = per_plant[per_plant > 0].sum()
        top5_advantage = per_plant[per_plant > 0].head(5).sum()
        print(f"Sum of positive (Transformer-favoring) advantage across all plants: {total_advantage:.2f}")
        print(f"Same, from just the top 5 plants: {top5_advantage:.2f} "
              f"({100*top5_advantage/total_advantage:.1f}% of the total, if concentrated this would be high)")

        print(f"\nTop 5 plants by Transformer advantage:")
        print(per_plant.head(5).to_string())
        print(f"\nBottom 5 plants (where LSTM does relatively better):")
        print(per_plant.tail(5).to_string())

        # Field/year breakdown
        per_field = sub.groupby("field")["transformer_advantage"].agg(["mean", "count"])
        print(f"\nBy field:")
        print(per_field)

    # ============================================================
    # Question 2: does the LSTM's degradation look smooth vs. n_kept, or
    # is there a cliff specifically at very short remaining lengths?
    # ============================================================
    print("\n" + "=" * 70)
    print("QUESTION 2: LSTM/Transformer error as a function of REMAINING observation count")
    print("=" * 70)

    by_n_kept = df.groupby("n_kept").agg(
        lstm_mae=("lstm_abs_error", "mean"),
        transformer_mae=("transformer_abs_error", "mean"),
        n_trials=("lstm_abs_error", "count"),
    ).reset_index().sort_values("n_kept")
    print(by_n_kept.to_string(index=False))

    print("\nLSTM MAE deltas between consecutive n_kept values (look for a cliff, not a smooth ramp):")
    by_n_kept["lstm_mae_delta"] = by_n_kept["lstm_mae"].diff()
    by_n_kept["transformer_mae_delta"] = by_n_kept["transformer_mae"].diff()
    print(by_n_kept[["n_kept", "n_trials", "lstm_mae", "lstm_mae_delta",
                      "transformer_mae", "transformer_mae_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
