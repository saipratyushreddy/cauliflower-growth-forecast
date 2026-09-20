"""
Step 3a: Build the plant-keyed metadata table from samples.json.

Produces one row per (plant_id, acquisition_date) record under task=='reference',
with the image path, day_after_planting, and the cleaned target trait value
(plant diameter). This table is the single source of truth for constructing
(input_sequence, target) forecasting pairs downstream.

Usage:
    python src/build_metadata.py --samples-json <path> --out data/metadata.parquet
"""
import argparse
import json
import re

import pandas as pd

NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def clean_diameter(raw):
    """
    Parse in_situ_diameter into a float, or None if unusable.

    Cleaning rules (see outputs/step2_inspection_report.txt DECISION LOG):
      - Plain numeric string -> float(v)
      - "X (Y)" or "X(Y)" paired format -> use X only (first/primary value),
        discard the parenthetical Y. Decided 2026-09-19: no dual-reading
        protocol is documented in the source paper (arXiv:2204.00294 sec 3.3,
        which lists a single "maximum diameter" measurement), in_situ_comment
        is empty on all such records, and they cluster on Field 2's first
        2021 acquisition date (seedling stage) rather than carrying a "2P"
        (two-plants) comment flag that would suggest a different reading.
      - "?" or blank/None -> None (missing)
    """
    if is_missing(raw):
        return None
    s = raw.strip()
    if s == "?":
        return None
    if NUMERIC_RE.match(s):
        return float(s)
    # paired "X (Y)" or "X(Y)" format -> take X
    m = re.match(r"^(-?\d+(\.\d+)?)\s*\(", s)
    if m:
        return float(m.group(1))
    # anything else unparseable -> missing, but this should not happen given
    # the Step 2 audit (all 165 non-numeric values were either "?" or "X (Y)").
    return None


def build_metadata(samples_json_path):
    with open(samples_json_path) as f:
        data = json.load(f)
    samples = data["samples"]
    ref_samples = [s for s in samples if s.get("task") == "reference"]

    rows = []
    for s in ref_samples:
        raw_diam = s.get("in_situ_diameter")
        rows.append({
            "plant_id": s.get("plant_id"),
            "field": s.get("field"),
            "plot": s.get("plot"),
            "acquisition_date": s.get("acquisition_date"),
            "day_after_planting": s.get("day_after_planting"),
            "planting_date": s.get("planting_date"),
            "filepath": s.get("filepath"),
            "diameter_raw": raw_diam,
            "diameter": clean_diameter(raw_diam),
        })

    df = pd.DataFrame(rows)

    # Sanity: exactly one record per (plant_id, acquisition_date)?
    dup = df.duplicated(subset=["plant_id", "acquisition_date"], keep=False)
    if dup.any():
        print(f"WARNING: {dup.sum()} rows share a duplicate (plant_id, acquisition_date) key:")
        print(df[dup].sort_values(["plant_id", "acquisition_date"]))

    df = df.sort_values(["plant_id", "day_after_planting"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-json", required=True)
    ap.add_argument("--out", required=True, help="Output Parquet path")
    args = ap.parse_args()

    df = build_metadata(args.samples_json)

    n_plants = df["plant_id"].nunique()
    n_rows = len(df)
    n_valid_diam = df["diameter"].notna().sum()
    n_dirty_recovered = ((df["diameter_raw"].notna())
                         & (df["diameter_raw"].astype(str).str.strip() != "")
                         & (df["diameter"].notna())
                         & (~df["diameter_raw"].astype(str).str.strip().str.match(NUMERIC_RE))).sum()

    print(f"Metadata table: {n_rows} rows, {n_plants} unique plants")
    print(f"Rows with valid (cleaned) diameter: {n_valid_diam} / {n_rows}")
    print(f"Rows recovered via 'X (Y)' -> X parsing rule: {n_dirty_recovered}")

    df.to_parquet(args.out, index=False)
    print(f"Wrote metadata table to {args.out}")


if __name__ == "__main__":
    main()
