"""
Image-prediction track, Step A: build (history -> next image) pairs and the
plant-wise split over ALL 739 plants.

Differences from the Phase 0 diameter pipeline (build_pairs.py):
  * No diameter-validity constraint: every acquisition is a usable frame, so
    the target for a plant's k-th image (k >= 1) is simply that image, and the
    input is every earlier image of the same plant. (Phase 0's "t+1" was the
    next date with a valid measurement; here it is the next acquired image.)
  * The split is FROZEN to Phase 0 for all 738 plants in pairs_split.parquet, so
    the test set is identical across tasks. (Re-running split_plants on the
    full 739 list reshuffles the whole permutation and moved 125 plants,
    including 29 Phase 0 train plants into test.) The one plant absent from
    Phase 0 (excluded there for <2 diameter measurements) is assigned with
    split_data.split_plants (RandomState(seed)) applied to that single plant.
    NOTE: with n=1, round(0.7*1)=1, so that plant deterministically lands in
    train; this is a documented exception, not a random draw.

Usage:
    python src/build_image_pairs.py --metadata data/metadata.parquet \
        --phase0-split data/pairs_split.parquet --out data/image_pairs.parquet --seed 42
"""
import argparse

import pandas as pd

from split_data import SEED, split_plants


def build_pairs(meta):
    rows = []
    for pid, g in meta.sort_values(["plant_id", "day_after_planting"]).groupby("plant_id"):
        fps = g["filepath"].tolist()
        days = g["day_after_planting"].tolist()
        for k in range(1, len(g)):
            rows.append({
                "pair_id": f"{pid}::{days[k]}",
                "plant_id": pid,
                "field": g["field"].iloc[0],
                "input_filepaths": fps[:k],
                "input_days": days[:k],
                "n_input_frames": k,
                "input_last_filepath": fps[k - 1],
                "input_last_day": days[k - 1],
                "target_filepath": fps[k],
                "target_day": days[k],
                "gap_days": days[k] - days[k - 1],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--phase0-split", required=True,
                    help="Phase 0 pairs_split.parquet; its plant->split assignments are kept as-is")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    meta = pd.read_parquet(args.metadata)
    assert not meta.duplicated(["plant_id", "day_after_planting"]).any()
    plant_ids = meta["plant_id"].unique().tolist()
    p0 = pd.read_parquet(args.phase0_split).drop_duplicates("plant_id").set_index("plant_id")["split"]
    assert set(p0.index) <= set(plant_ids), "Phase 0 plants missing from metadata"
    split_of = p0.to_dict()
    extra = sorted(set(plant_ids) - set(p0.index))
    e_train, e_val, e_test = split_plants(extra, seed=args.seed)  # single plant -> deterministic
    for name, ids in [("train", e_train), ("val", e_val), ("test", e_test)]:
        for pid in ids:
            split_of[pid] = name
    print(f"Frozen from Phase 0: {len(p0)} plants. Newly assigned via RandomState({args.seed}): "
          f"{ {pid: split_of[pid] for pid in extra} }")
    assert set(split_of) == set(plant_ids)
    sets = {s: {p for p, v in split_of.items() if v == s} for s in ["train", "val", "test"]}
    assert sets["train"].isdisjoint(sets["val"]) and sets["train"].isdisjoint(sets["test"]) \
        and sets["val"].isdisjoint(sets["test"])
    assert all(split_of[p] == p0[p] for p in p0.index), "Phase 0 assignment changed!"

    pairs = build_pairs(meta)
    assert pairs["pair_id"].is_unique
    pairs["split"] = pairs["plant_id"].map(split_of)
    assert (pairs.groupby("plant_id")["split"].nunique() == 1).all()

    print(f"Plants: {len(plant_ids)}  |  pairs: {len(pairs)}")
    for s in ["train", "val", "test"]:
        sub = pairs[pairs["split"] == s]
        print(f"  {s:5s}: {sub['plant_id'].nunique()} plants, {len(sub)} pairs")
    print("Pairs per field/split:")
    print(pairs.groupby(["field", "split"]).size().unstack().to_string())
    print("Gap days (all pairs):", pairs["gap_days"].describe().round(2).to_dict())
    print("Assertion passed: zero plant_id overlap across splits.")

    pairs.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
