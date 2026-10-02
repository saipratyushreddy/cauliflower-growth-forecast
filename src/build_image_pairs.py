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
  * BLURRY_EXCLUDE (9 images, Plot5 2021-08-11 / Plot1 2021-08-30) are removed as
    target and last input; elsewhere in a pair's history they are scrubbed.
    Field1 day 28 (dark but sharp, valid data) is deliberately NOT excluded.
  * --exclude-list drops every pair whose input OR target image is a
    degenerate (black/placeholder) frame. 234 files <10KB were found on Swan
    (232 byte-identical placeholders + 2 near-black frames), all Field1,
    day 91/93 (last 1-2 acquisitions of 119 plants). Plant split assignments
    are unaffected; only pairs are removed.

Usage:
    python src/build_image_pairs.py --metadata data/metadata.parquet \
        --phase0-split data/pairs_split.parquet --out data/image_pairs.parquet --seed 42
"""
import argparse

import pandas as pd

from split_data import SEED, split_plants

# Visually confirmed blurry (stitching/focus) frames; see README "Image quality exclusions".
# (plant_id, acquisition_date). Excluded as target AND as last input. Not bridged: no
# pair is created across a removed frame. Where such a frame appears only in the
# older history of a later pair, it is scrubbed from that history and the pair is kept.
BLURRY_EXCLUDE = (
    [(f"2021_Ref_Plot5_{p}", "2021-08-11") for p in ["A16", "A17", "A18", "B17"]]
    + [(f"2021_Ref_Plot1_{p}", "2021-08-30") for p in ["A4", "A5", "A6", "B6", "B7"]]
)


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
    ap.add_argument("--exclude-list", default=None,
                    help="md5sum-style file ('<md5>  <filepath>' per line) of degenerate images; "
                         "pairs with any such image as input or target are dropped")
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
    base_counts = pairs["split"].value_counts().to_dict()
    print(f"Unfiltered: {len(pairs)} pairs {base_counts}")

    def report(label, mask):
        d = pairs[mask]
        print(f"  {label}: {len(d)} pairs removed -> "
              f"{ {s: int((d['split'] == s).sum()) for s in ['train', 'val', 'test']} }")

    if args.exclude_list:
        bad = set(pd.read_csv(args.exclude_list, sep=r"\s+", header=None, names=["md5", "fp"])["fp"])
        assert bad <= set(meta["filepath"]), "exclude-list has paths not in metadata"
        drop = pairs["target_filepath"].isin(bad) | pairs["input_filepaths"].apply(lambda l: any(f in bad for f in l))
        print(f"Black/placeholder frames: {len(bad)} images ({pairs[drop]['plant_id'].nunique()} plants, "
              f"fields {sorted(pairs[drop]['field'].unique())})")
        report("black frames", drop)
        pairs = pairs[~drop].reset_index(drop=True)

    blur_rows = meta.merge(pd.DataFrame(BLURRY_EXCLUDE, columns=["plant_id", "acquisition_date"]),
                           on=["plant_id", "acquisition_date"])
    assert len(blur_rows) == len(BLURRY_EXCLUDE), "BLURRY_EXCLUDE entry not found in metadata"
    blur = set(blur_rows["filepath"])
    drop_t = pairs["target_filepath"].isin(blur)
    drop_i = pairs["input_last_filepath"].isin(blur)
    print(f"Blurry frames: {len(blur)} images ({blur_rows['plant_id'].nunique()} plants)")
    report("blurry as target", drop_t)
    report("blurry as last input (not already counted as target)", drop_i & ~drop_t)
    report("blurry total", drop_t | drop_i)
    # Alternative (not applied): also dropping every pair that merely has one in its history
    hist = pairs["input_filepaths"].apply(lambda l: any(f in blur for f in l))
    alt = drop_t | drop_i | hist
    print(f"  [for reference, NOT applied] dropping pairs with a blurry frame anywhere in history too "
          f"would remove {int(alt.sum())} pairs: { {s: int((pairs[alt]['split'] == s).sum()) for s in ['train', 'val', 'test']} }")
    pairs = pairs[~(drop_t | drop_i)].reset_index(drop=True)
    keep = pairs["input_filepaths"].apply(lambda l: [f not in blur for f in l])
    n_scrub = int(keep.apply(lambda k: not all(k)).sum())
    pairs["input_days"] = [[d for d, k in zip(ds, ks) if k] for ds, ks in zip(pairs["input_days"], keep)]
    pairs["input_filepaths"] = [[f for f, k in zip(fs, ks) if k] for fs, ks in zip(pairs["input_filepaths"], keep)]
    pairs["n_input_frames"] = pairs["input_filepaths"].apply(len)
    assert (pairs["n_input_frames"] >= 1).all()
    print(f"  {n_scrub} retained pairs had a blurry frame scrubbed from their older history")
    assert not pairs["target_filepath"].isin(blur).any() and not pairs["input_last_filepath"].isin(blur).any()
    assert pairs["pair_id"].is_unique

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
