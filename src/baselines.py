"""
Step 5: Persistence and single-frame baselines, evaluated on held-out test
plants.

(a) Persistence: y_hat_{t+1} = y_t. Only constructed for pairs where the
    LAST INPUT DATE (t, i.e. `last_input_day`/`last_input_date` in the pairs
    table) itself has a valid diameter measurement -- this is a stricter
    filter than the full pairs table, which allows the last input date to
    be an unlabeled image (see build_pairs.py docstring). We look up y_t
    from metadata.parquet by (plant_id, last_input_day) rather than storing
    it redundantly in pairs.parquet, since pairs.parquet was built without
    assuming any use of the immediately-preceding label.

(b) Single-frame: regress target_diameter from ONLY the cached embedding of
    the single most recent input frame (image at t), no temporal modeling.
    Trained as a simple linear (ridge) regression on the 512-dim ResNet18
    embedding -- adequate as a non-temporal baseline; not itself a deep
    model. Fit on train split, evaluated on val (for the ridge alpha) and
    test.

Every pair has a stable identifier `pair_id = (plant_id, target_day)`
(unique because a plant has at most one measurement per day). This id is
used to align samples ACROSS baselines and, later, across Step 6's
CNN-LSTM, since each method can have a slightly different eligible-sample
set (persistence needs a valid y_t; single-frame needs a cached last-frame
embedding, which in practice is always available; the LSTM will need the
full embedding sequence). The convention going forward:
  - Always report each method's own full-N metrics (honest per-method
    result, what a real deployment would see).
  - ALSO report a "shared eval set" comparison: metrics recomputed for
    every reported method restricted to the INTERSECTION of pair_ids all
    of them can produce a prediction for, so head-to-head deltas (e.g.
    "N% better MAE") are never comparing two different sample sets.
  - Per-pair predictions/residuals are dumped to CSV for later inspection
    (e.g. worst-miss analysis), keyed by pair_id and split.

Usage:
    python src/baselines.py \
        --pairs-split data/pairs_split.parquet \
        --metadata data/metadata.parquet \
        --embeddings-dir data/embeddings \
        --norm-stats data/norm_stats.json \
        --out-dir outputs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def make_pair_id(df):
    return df["plant_id"].astype(str) + "::" + df["target_day"].astype(str)


def build_persistence_frame(pairs_df, meta_df):
    """
    For each pair, look up y_t = diameter at (plant_id, last_input_day).
    Keep only pairs where that lookup finds a valid (non-missing) value.
    """
    lookup = meta_df.set_index(["plant_id", "day_after_planting"])["diameter"]

    y_t_values = []
    has_y_t = []
    for _, row in pairs_df.iterrows():
        key = (row["plant_id"], row["last_input_day"])
        if key in lookup.index:
            v = lookup.loc[key]
            if isinstance(v, pd.Series):  # shouldn't happen (metadata is 1 row per plant/date) but be safe
                v = v.iloc[0]
        else:
            v = None
        y_t_values.append(v)
        has_y_t.append(v is not None and not pd.isna(v))

    out = pairs_df.copy()
    out["y_t"] = y_t_values
    out["has_y_t"] = has_y_t
    return out[out["has_y_t"]].copy()


def load_last_frame_embeddings(pairs_df, embeddings_dir):
    """
    For each pair, load the cached embedding of the LAST image in
    input_filepaths (the most recent frame, image at t) from its shard.
    Returns an (N, 512) array aligned with pairs_df's row order.
    """
    index_df = pd.read_parquet(os.path.join(embeddings_dir, "embedding_index.parquet"))
    fp_to_loc = index_df.set_index("image_id")[["shard_file", "offset"]]

    shard_cache = {}
    embeddings = []
    missing_mask = []
    for _, row in pairs_df.iterrows():
        last_fp = row["input_filepaths"][-1]
        if last_fp not in fp_to_loc.index:
            embeddings.append(None)
            missing_mask.append(True)
            continue
        shard_file, offset = fp_to_loc.loc[last_fp, ["shard_file", "offset"]]
        if shard_file not in shard_cache:
            shard_cache[shard_file] = torch.load(os.path.join(embeddings_dir, shard_file))
        shard = shard_cache[shard_file]
        emb = shard["embeddings"][offset].numpy()
        embeddings.append(emb)
        missing_mask.append(False)

    return embeddings, missing_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pairs_df = pd.read_parquet(args.pairs_split)
    pairs_df["pair_id"] = make_pair_id(pairs_df)
    assert pairs_df["pair_id"].is_unique, "pair_id is not unique -- a plant has >1 row for the same target_day!"

    meta_df = pd.read_parquet(args.metadata)

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    target_mean = norm_stats["target_diameter_mean"]
    target_std = norm_stats["target_diameter_std"]

    results = {}
    test_preds = {}  # method -> DataFrame(pair_id, y_true, y_pred) for TEST split only

    # ============================================================
    # (a) Persistence baseline
    # ============================================================
    print("=" * 60)
    print("PERSISTENCE BASELINE: y_hat_{t+1} = y_t")
    print("=" * 60)

    persistence_df = build_persistence_frame(pairs_df, meta_df)
    print(f"Pairs with a valid y_t (both y_t and y_(t+1) exist): "
          f"{len(persistence_df)} / {len(pairs_df)} total pairs")
    n_plants_persistence = persistence_df["plant_id"].nunique()
    print(f"Distinct plants surviving this filter: {n_plants_persistence} / {pairs_df['plant_id'].nunique()}")

    for split in ["train", "val", "test"]:
        split_df = persistence_df[persistence_df["split"] == split]
        n = len(split_df)
        n_p = split_df["plant_id"].nunique()
        if n == 0:
            print(f"  {split}: N=0 pairs, skipping metric computation")
            continue
        mae, rmse = mae_rmse(split_df["target_diameter"], split_df["y_t"])
        print(f"  {split}: N={n} pairs ({n_p} plants), MAE={mae:.3f}, RMSE={rmse:.3f}")
        if split == "test":
            results["persistence"] = {"n_pairs": n, "n_plants": n_p, "mae": mae, "rmse": rmse}
            test_preds["persistence"] = pd.DataFrame({
                "pair_id": split_df["pair_id"].values,
                "y_true": split_df["target_diameter"].values,
                "y_pred": split_df["y_t"].values,
            })

    # ============================================================
    # (b) Single-frame baseline (ridge regression on last-frame embedding)
    # ============================================================
    print("\n" + "=" * 60)
    print("SINGLE-FRAME BASELINE: ridge regression on embedding at t")
    print("=" * 60)

    embeddings, missing_mask = load_last_frame_embeddings(pairs_df, args.embeddings_dir)
    sf_df = pairs_df.copy()
    sf_df["embedding"] = embeddings
    sf_df["embedding_missing"] = missing_mask
    n_missing = sum(missing_mask)
    if n_missing > 0:
        print(f"WARNING: {n_missing} pairs have no cached embedding for their last input frame "
              f"(unexpected given Step 4's 0-failure run) -- dropping these.")
    sf_df = sf_df[~sf_df["embedding_missing"]].copy()
    print(f"Pairs with a usable last-frame embedding: {len(sf_df)} / {len(pairs_df)} total pairs")

    train_sf = sf_df[sf_df["split"] == "train"]
    val_sf = sf_df[sf_df["split"] == "val"]
    test_sf = sf_df[sf_df["split"] == "test"]

    X_train = np.stack(train_sf["embedding"].values)
    y_train_raw = train_sf["target_diameter"].values
    y_train_norm = (y_train_raw - target_mean) / target_std

    X_val = np.stack(val_sf["embedding"].values)
    y_val_raw = val_sf["target_diameter"].values

    X_test = np.stack(test_sf["embedding"].values)
    y_test_raw = test_sf["target_diameter"].values

    # Small alpha sweep, model-selected on val (never test), fit only on train.
    best_alpha, best_val_mae, best_model = None, np.inf, None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train_norm)
        val_pred_norm = model.predict(X_val)
        val_pred_raw = val_pred_norm * target_std + target_mean
        val_mae, _ = mae_rmse(y_val_raw, val_pred_raw)
        print(f"  alpha={alpha:>8.1f}  val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae, best_alpha, best_model = val_mae, alpha, model

    print(f"Selected alpha={best_alpha} (lowest val MAE={best_val_mae:.3f})")

    test_pred_norm = best_model.predict(X_test)
    test_pred_raw = test_pred_norm * target_std + target_mean
    test_mae, test_rmse = mae_rmse(y_test_raw, test_pred_raw)
    n_test_plants = test_sf["plant_id"].nunique()
    print(f"  test: N={len(test_sf)} pairs ({n_test_plants} plants), "
          f"MAE={test_mae:.3f}, RMSE={test_rmse:.3f}")

    results["single_frame"] = {
        "n_pairs": len(test_sf), "n_plants": n_test_plants,
        "mae": test_mae, "rmse": test_rmse, "selected_alpha": best_alpha,
    }
    test_preds["single_frame"] = pd.DataFrame({
        "pair_id": test_sf["pair_id"].values,
        "y_true": y_test_raw,
        "y_pred": test_pred_raw,
    })

    # ============================================================
    # Shared evaluation set: intersection of test pair_ids that EVERY
    # reported method can produce a prediction for. Metrics recomputed
    # here are the ones that should be used for head-to-head comparisons
    # between methods -- the per-method numbers above remain each
    # method's own honest full-N result, but are not directly comparable
    # to each other since their eligible sample sets differ.
    # ============================================================
    print("\n" + "=" * 60)
    print("SHARED EVALUATION SET (intersection of all methods' test pair_ids)")
    print("=" * 60)

    shared_ids = None
    for method, df in test_preds.items():
        ids = set(df["pair_id"])
        shared_ids = ids if shared_ids is None else (shared_ids & ids)

    n_shared = len(shared_ids)
    n_persistence_only = len(set(test_preds["persistence"]["pair_id"]) - shared_ids)
    n_single_frame_only = len(set(test_preds["single_frame"]["pair_id"]) - shared_ids)
    print(f"Shared test pairs (all methods can predict): {n_shared}")
    print(f"  Dropped from persistence's own set (had y_t, but excluded from shared set for another reason): "
          f"{n_persistence_only}")
    print(f"  Dropped from single-frame's own set: {n_single_frame_only}")

    shared_results = {}
    for method, df in test_preds.items():
        shared_df = df[df["pair_id"].isin(shared_ids)]
        mae, rmse = mae_rmse(shared_df["y_true"], shared_df["y_pred"])
        print(f"  {method}: N={len(shared_df)}, MAE={mae:.3f}, RMSE={rmse:.3f}")
        shared_results[method] = {"n_pairs": len(shared_df), "mae": mae, "rmse": rmse}

    mae_improvement = 100 * (1 - shared_results["single_frame"]["mae"] / shared_results["persistence"]["mae"])
    rmse_improvement = 100 * (1 - shared_results["single_frame"]["rmse"] / shared_results["persistence"]["rmse"])
    print(f"\nOn the SAME {n_shared} test pairs: single-frame improves MAE by {mae_improvement:.1f}% "
          f"and RMSE by {rmse_improvement:.1f}% over persistence.")

    results["shared_eval_set"] = {
        "n_pairs": n_shared,
        "methods": shared_results,
        "single_frame_vs_persistence_mae_improvement_pct": mae_improvement,
        "single_frame_vs_persistence_rmse_improvement_pct": rmse_improvement,
    }

    # ============================================================
    # Persist per-pair predictions for residual/worst-miss analysis
    # ============================================================
    os.makedirs(args.out_dir, exist_ok=True)
    for method, df in test_preds.items():
        df = df.copy()
        df["residual"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = df["residual"].abs()
        df.sort_values("abs_error", ascending=False).to_csv(
            os.path.join(args.out_dir, f"step5_{method}_test_predictions.csv"), index=False
        )

    out_path = os.path.join(args.out_dir, "step5_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")
    print(f"Wrote per-pair predictions to {args.out_dir}/step5_<method>_test_predictions.csv")


if __name__ == "__main__":
    main()
