"""
Follow-up check: do the 2 plants that accounted for >50% of Step 5's
worst single-frame misses (2021_Ref_Plot2_A1, 2021_Ref_Plot2_E18) also
show up disproportionately in Step 6's flat-or-decline segment analysis?

If yes: the Step 5 finding is subsumed by the broader late-season decline
pattern (these plants just have more/larger decline segments than
average) and doesn't need separate treatment in limitations.

If no: it's a second, distinct anomaly (e.g. a real data issue specific
to those plants) still worth a sentence in limitations.

Usage:
    python src/check_worst_plants_vs_decline.py \
        --pairs-split data/pairs_split.parquet \
        --metadata data/metadata.parquet \
        --single-frame-predictions outputs/step5_single_frame_test_predictions.csv \
        --samples-json <path to samples.json, for in_situ_comment lookup>
"""
import argparse
import json

import pandas as pd

WORST_PLANTS = ["2021_Ref_Plot2_A1", "2021_Ref_Plot2_E18"]


def make_pair_id(df):
    return df["plant_id"].astype(str) + "::" + df["target_day"].astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--single-frame-predictions", required=True)
    ap.add_argument("--samples-json", default=None,
                     help="Path to samples.json, for in_situ_comment lookup (metadata.parquet doesn't carry it)")
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
    test_df["delta"] = test_df["target_diameter"] - test_df["y_t"]

    print(f"Overall flat-or-decline rate (all test plants): "
          f"{100*test_df['is_decline_or_flat'].mean():.1f}%")

    for plant_id in WORST_PLANTS:
        plant_pairs = test_df[test_df["plant_id"] == plant_id]
        if len(plant_pairs) == 0:
            print(f"\n{plant_id}: NOT FOUND in test_df with a valid y_t lookup "
                  f"(check plant_id spelling / whether it's actually a test plant)")
            continue
        n_decline = plant_pairs["is_decline_or_flat"].sum()
        n_total = len(plant_pairs)
        print(f"\n{plant_id}: {n_decline}/{n_total} pairs are flat-or-decline "
              f"({100*n_decline/n_total:.1f}%)")
        print(f"  Deltas: {sorted(plant_pairs['delta'].tolist())}")
        print(f"  Target days: {sorted(plant_pairs['target_day'].tolist())}")

    # Cross-reference with Step 5's actual worst-miss records for these plants
    print("\n--- Step 5 single-frame worst misses for these 2 plants ---")
    sf_preds = pd.read_csv(args.single_frame_predictions)
    for plant_id in WORST_PLANTS:
        plant_preds = sf_preds[sf_preds["pair_id"].str.startswith(plant_id + "::")]
        plant_preds = plant_preds.merge(
            test_df[["pair_id", "is_decline_or_flat", "delta"]], on="pair_id", how="left"
        )
        print(f"\n{plant_id} ({len(plant_preds)} test pairs):")
        print(plant_preds[["pair_id", "y_true", "y_pred", "abs_error", "is_decline_or_flat", "delta"]]
              .sort_values("abs_error", ascending=False).to_string(index=False))

    # Also check in_situ_comment for these 2 plants across their FULL history
    # (not just test pairs) for any documented stress/anomaly flag.
    # metadata.parquet doesn't carry in_situ_comment (only diameter survived
    # the build_metadata.py cleaning pipeline), so read it fresh from
    # samples.json here instead.
    if args.samples_json:
        print("\n--- in_situ_comment for these 2 plants (full measured history) ---")
        with open(args.samples_json) as f:
            samples = json.load(f)["samples"]
        ref_samples = [s for s in samples if s.get("task") == "reference"]
        for plant_id in WORST_PLANTS:
            plant_samples = sorted(
                [s for s in ref_samples if s.get("plant_id") == plant_id],
                key=lambda s: s.get("day_after_planting"),
            )
            print(f"\n{plant_id}:")
            for s in plant_samples:
                print(f"  day={s.get('day_after_planting'):>3}  "
                      f"diameter_raw={s.get('in_situ_diameter')!r:>10}  "
                      f"comment={s.get('in_situ_comment')!r}")
    else:
        print("\n(--samples-json not provided; skipping in_situ_comment lookup)")


if __name__ == "__main__":
    main()
