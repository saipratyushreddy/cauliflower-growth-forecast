"""
Multi-trait table: every in-situ trait of the GrowliFlower reference plants, one row per (plant, acquisition date), keyed like data/metadata.parquet.

Columns added to metadata.parquet's rows (same filepath key):
  height          in_situ_height, numeric only (271 plants, mostly Field1)
  head_diameter   in_situ_head_diameter, numeric only (late season: Field2 days 44-85, Field1 days 83-93)
  bbch            in_situ_bbch_stage, numeric codes only (malformed values such as '41,41', '3X', 'PD', '191' -> missing)
  harvested       True where in_situ_harvested is marked ('x', 'X', '(x)')
  harvest_timepoint  dataset's per-plant harvest wave (constant per plant)
  harvest_day     per plant: first day_after_planting marked harvested (NaN if never marked)
Units are those of the dataset; none is stated in the paper or the dataset card (see README caveats).

Usage:
    python src/build_traits.py --samples-json <.../samples.json> --metadata data/metadata.parquet --out data/traits.parquet
"""
import argparse
import json
import re

import pandas as pd

NUM = re.compile(r"^-?\d+(\.\d+)?$")
VALID_BBCH = set(range(10, 100))


def num(v):
    return float(v) if isinstance(v, str) and NUM.match(v.strip()) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-json", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = [x for x in json.load(open(a.samples_json))["samples"] if x.get("task") == "reference"]
    rows = []
    for x in s:
        bb = num(x.get("in_situ_bbch_stage"))
        rows.append({"filepath": x["filepath"], "height": num(x.get("in_situ_height")), "head_diameter": num(x.get("in_situ_head_diameter")),
                     "bbch": bb if bb is not None and int(bb) in VALID_BBCH else None,
                     "harvested": isinstance(x.get("in_situ_harvested"), str) and "x" in x["in_situ_harvested"].lower(),
                     "harvest_timepoint": x.get("harvest_timepoint")})
    t = pd.DataFrame(rows)
    m = pd.read_parquet(a.metadata)
    out = m.merge(t, on="filepath", how="left", validate="one_to_one")
    assert len(out) == len(m) and out["harvest_timepoint"].notna().all()
    hd = out[out.harvested].groupby("plant_id")["day_after_planting"].min().rename("harvest_day")
    out = out.merge(hd, on="plant_id", how="left")
    out.to_parquet(a.out, index=False)
    for c in ("diameter", "height", "head_diameter", "bbch"):
        g = out[out[c].notna()]
        print(f"{c:14s} records {len(g):5d}  plants {g.plant_id.nunique():3d}  by field {g.groupby('field').plant_id.nunique().to_dict()}")
    p = out.groupby(["field", "plant_id"])["harvest_day"].first()
    print("harvest_day    plants with a date:", p.notna().groupby("field").sum().to_dict(), " median:", p.groupby("field").median().to_dict())
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
