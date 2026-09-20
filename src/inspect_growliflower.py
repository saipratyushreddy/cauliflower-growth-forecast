"""
Step 2 inspection script for the GrowliFlower dataset (Voxel51/GrowliFlower).

Parses samples.json (the raw FiftyOne export) directly rather than loading
through the `fiftyone` package, since we only need field-level metadata and
label coverage at this stage, not images or FiftyOne's App/DB machinery.

Usage:
    python src/inspect_growliflower.py --samples-json <path to samples.json>
"""
import argparse
import json
import re
from collections import defaultdict


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    return data["samples"]


def is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def classify_value(v):
    """Return 'numeric', 'numeric_with_annotation', or 'other' for a raw string trait value."""
    if is_missing(v):
        return "missing"
    s = v.strip()
    if NUMERIC_RE.match(s):
        return "numeric"
    # e.g. "10 (10)", "12,5", ranges, comments appended
    return "non_numeric_string"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-json", required=True)
    args = ap.parse_args()

    samples = load_samples(args.samples_json)
    print(f"Total samples in file: {len(samples)}")

    by_task = defaultdict(int)
    for s in samples:
        by_task[s.get("task")] += 1
    print("\nSamples by task:")
    for k, v in by_task.items():
        print(f"  {k}: {v}")

    # ---- Fields present across the dataset (union) ----
    all_fields = set()
    for s in samples:
        all_fields.update(s.keys())
    print(f"\nUnion of all top-level fields across samples ({len(all_fields)}):")
    for f in sorted(all_fields):
        print(f"  - {f}")

    # ---- Focus on task == reference (has in_situ_* trait fields) ----
    ref_samples = [s for s in samples if s.get("task") == "reference"]
    print(f"\n=== task == 'reference' samples: {len(ref_samples)} ===")

    plants = defaultdict(list)
    for s in ref_samples:
        pid = s.get("plant_id")
        plants[pid].append(s)

    print(f"Unique plant_id values under task=='reference': {len(plants)}")

    # Time points per plant
    counts = sorted(len(v) for v in plants.values())
    if counts:
        print(f"Acquisition records per plant: min={counts[0]}, max={counts[-1]}, "
              f"median={counts[len(counts)//2]}")

    # Field / plot breakdown
    field_counts = defaultdict(int)
    for pid, recs in plants.items():
        fields_for_plant = {r.get("field") for r in recs}
        for f in fields_for_plant:
            field_counts[f] += 1
    print(f"Plants by field (a plant should belong to exactly one field): {dict(field_counts)}")

    # ---- Trait coverage ----
    trait_fields = {
        "height": "in_situ_height",
        "plant diameter": "in_situ_diameter",
        "head diameter": "in_situ_head_diameter",
        "developmental stage": "in_situ_bbch_stage",
        "harvest status": "in_situ_harvested",
    }

    print("\n=== Trait coverage report (over plants with task=='reference') ===")
    for label, field in trait_fields.items():
        plants_with_any = 0
        plants_with_2plus = 0
        value_types = defaultdict(int)
        value_samples = []
        for pid, recs in plants.items():
            valid_vals = []
            for r in recs:
                v = r.get(field)
                cls = classify_value(v)
                value_types[cls] += 1
                if cls != "missing":
                    valid_vals.append(v)
                    if len(value_samples) < 15:
                        value_samples.append(v)
            if len(valid_vals) >= 1:
                plants_with_any += 1
            if len(valid_vals) >= 2:
                plants_with_2plus += 1

        print(f"\n--- {label} ({field}) ---")
        print(f"  Plants with >=1 valid measurement: {plants_with_any} / {len(plants)}")
        print(f"  Plants with >=2 valid measurements: {plants_with_2plus} / {len(plants)}")
        print(f"  Value classification counts (per-record, not per-plant): {dict(value_types)}")
        print(f"  Sample raw values: {value_samples}")

        # class balance if this looks categorical
        if label in ("developmental stage", "harvest status"):
            balance = defaultdict(int)
            for pid, recs in plants.items():
                for r in recs:
                    v = r.get(field)
                    if not is_missing(v):
                        balance[v.strip()] += 1
            print(f"  Class balance (record-level counts): {dict(sorted(balance.items(), key=lambda kv: -kv[1]))}")

    # ---- day_after_planting / acquisition_date regularity check ----
    print("\n=== Timestamp regularity (task=='reference') ===")
    sample_plant_ids = list(plants.keys())[:5]
    for pid in sample_plant_ids:
        recs = sorted(plants[pid], key=lambda r: r.get("day_after_planting", -1))
        days = [r.get("day_after_planting") for r in recs]
        dates = [r.get("acquisition_date") for r in recs]
        print(f"  plant_id={pid}: day_after_planting={days}")
        print(f"    acquisition_date={dates}")


if __name__ == "__main__":
    main()
