"""
Follow-up on the n_kept=2 spike found in analyze_robustness_results.py:
LSTM MAE jumps from 8.70 (n_kept=1) to 18.52 (n_kept=2), then falls back
to 14.63 (n_kept=3). Is this a genuine broad instability across most
n_kept=2 trials, or a mean skewed by a handful of pathological outlier
predictions?

Usage:
    python src/dig_into_n_kept_2.py --per-pair-csv outputs/step_p2_robustness_per_pair.csv
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.per_pair_csv)
    df["lstm_abs_error"] = (df["lstm_pred"] - df["y_true"]).abs()
    df["transformer_abs_error"] = (df["transformer_pred"] - df["y_true"]).abs()
    df["plant_id"] = df["pair_id"].str.split("::").str[0]

    for n in [1, 2, 3]:
        sub = df[df["n_kept"] == n]
        print(f"\n{'='*70}")
        print(f"n_kept={n}  (N={len(sub)} trials)")
        print(f"{'='*70}")

        print("\nLSTM abs_error distribution:")
        print(sub["lstm_abs_error"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

        print("\nTransformer abs_error distribution:")
        print(sub["transformer_abs_error"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

        # Median (robust to outliers) vs mean, for both models
        print(f"\nMedian LSTM abs_error: {sub['lstm_abs_error'].median():.3f}  "
              f"(vs mean {sub['lstm_abs_error'].mean():.3f})")
        print(f"Median Transformer abs_error: {sub['transformer_abs_error'].median():.3f}  "
              f"(vs mean {sub['transformer_abs_error'].mean():.3f})")

        # How many trials exceed a "large error" threshold, for each model
        for thresh in [20, 30, 50, 100]:
            n_lstm = (sub["lstm_abs_error"] > thresh).sum()
            n_tfmr = (sub["transformer_abs_error"] > thresh).sum()
            print(f"  Trials with abs_error > {thresh}: LSTM={n_lstm} ({100*n_lstm/len(sub):.1f}%), "
                  f"Transformer={n_tfmr} ({100*n_tfmr/len(sub):.1f}%)")

        # Worst 10 LSTM predictions at this n_kept level
        worst = sub.sort_values("lstm_abs_error", ascending=False).head(10)
        print(f"\nWorst 10 LSTM predictions at n_kept={n}:")
        print(worst[["pair_id", "seed", "y_true", "lstm_pred", "lstm_abs_error",
                      "transformer_pred", "transformer_abs_error"]].to_string(index=False))

        # How many DISTINCT plants appear in the worst 10% of LSTM errors
        # here, relative to the total distinct-plant pool at this n_kept
        # level? A low distinct-plant count relative to trial count would
        # indicate concentration in a few repeat-offender plants; a count
        # close to the trial count indicates the worst errors are spread
        # across many different plants (broad instability, not a few
        # pathological plants).
        n_worst_decile = max(1, len(sub) // 10)
        worst_decile = sub.sort_values("lstm_abs_error", ascending=False).head(n_worst_decile)
        n_distinct_plants_in_worst = worst_decile["plant_id"].nunique()
        n_distinct_plants_total = sub["plant_id"].nunique()
        print(f"\nWorst 10% of LSTM trials at n_kept={n} ({n_worst_decile} trials): "
              f"{n_distinct_plants_in_worst} distinct plants involved "
              f"(out of {n_distinct_plants_total} distinct plants total at this n_kept level)")


if __name__ == "__main__":
    main()
