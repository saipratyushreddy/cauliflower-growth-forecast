"""
Step 3c: Plant-wise (grouped) train/val/test split, 70/15/15, fixed seed.

Splits at the PLANT level (never the pair level) so no plant_id appears in
more than one split -- this is required because pairs from the same plant
share overlapping input images and are temporally correlated; a row-level
split would leak information across train/val/test.

Usage:
    python src/split_data.py --pairs data/pairs.parquet --out-dir data/ --seed 42
"""
import argparse

import numpy as np
import pandas as pd

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def split_plants(plant_ids, seed=SEED):
    rng = np.random.RandomState(seed)
    plant_ids = sorted(plant_ids)  # deterministic order before shuffling
    plant_ids = np.array(plant_ids)
    rng.shuffle(plant_ids)

    n = len(plant_ids)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # test gets the remainder so all plants are assigned
    train_ids = set(plant_ids[:n_train])
    val_ids = set(plant_ids[n_train:n_train + n_val])
    test_ids = set(plant_ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pairs_df = pd.read_parquet(args.pairs)
    plant_ids = pairs_df["plant_id"].unique().tolist()

    train_ids, val_ids, test_ids = split_plants(plant_ids, seed=args.seed)

    # --- Hard assertion: zero plant_id overlap across splits ---
    assert train_ids.isdisjoint(val_ids), "train/val plant overlap!"
    assert train_ids.isdisjoint(test_ids), "train/test plant overlap!"
    assert val_ids.isdisjoint(test_ids), "val/test plant overlap!"
    assert train_ids | val_ids | test_ids == set(plant_ids), "not all plants assigned!"

    def assign_split(pid):
        if pid in train_ids:
            return "train"
        elif pid in val_ids:
            return "val"
        else:
            return "test"

    pairs_df = pairs_df.copy()
    pairs_df["split"] = pairs_df["plant_id"].map(assign_split)

    # Re-verify at the row level too (belt and suspenders).
    row_level_check = pairs_df.groupby("plant_id")["split"].nunique()
    assert (row_level_check == 1).all(), "a plant_id spans multiple splits at the row level!"

    print(f"Total plants: {len(plant_ids)}")
    print(f"  train: {len(train_ids)} plants, {(pairs_df['split']=='train').sum()} pairs")
    print(f"  val:   {len(val_ids)} plants, {(pairs_df['split']=='val').sum()} pairs")
    print(f"  test:  {len(test_ids)} plants, {(pairs_df['split']=='test').sum()} pairs")
    print(f"\nAssertion passed: zero plant_id overlap across train/val/test splits.")

    out_path = f"{args.out_dir.rstrip('/')}/pairs_split.parquet"
    pairs_df.to_parquet(out_path, index=False)
    print(f"Wrote split-labeled pairs table to {out_path}")


if __name__ == "__main__":
    main()
