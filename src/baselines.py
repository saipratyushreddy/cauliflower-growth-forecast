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

Both baselines report MAE/RMSE on test, over ONLY the sample set each one
can actually produce a prediction for, with N stated for each -- per spec,
persistence's usable N may be smaller than the single-frame baseline's,
since persistence additionally requires a valid label at the input's last
date rather than just an image.

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
    meta_df = pd.read_parquet(args.metadata)

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    target_mean = norm_stats["target_diameter_mean"]
    target_std = norm_stats["target_diameter_std"]

    results = {}

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

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "step5_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
