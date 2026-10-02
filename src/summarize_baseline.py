"""
Final evaluation protocol for the image-prediction track, aggregated from the
per-pair scores written by image_copy_forward_baseline.py (no images re-read).

Protocol: on the CLEANED pair set (data/image_pairs.parquet), test split:
  pooled, pooled excluding Field1-day-28 pairs, Field1, Field1 excluding day 28,
  Field1 day-28 only (and its two directions), Field2. Each reports mean AND median of
  RGB SSIM, RGB PSNR and structure SSIM. "Day-28 pair" = Field1 pair whose last input OR
  target is day_after_planting 28 (2020-08-25: dark, sharp, valid; stratified, not excluded).
Oracle colour-match metrics (use TARGET statistics; not achievable at inference) are written
under a separate key / table and must not be compared with baselines or models.

Per-pair scores are position-independent, so dropping excluded pairs here gives exactly
what re-scoring the cleaned set would.

Usage:
    python src/summarize_baseline.py --per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv \
        --image-pairs data/image_pairs.parquet --out outputs/img_stepB_final_protocol.json
"""
import argparse
import json

import numpy as np
import pandas as pd

HONEST = ["ssim", "psnr", "structure_ssim"]
ORACLE = ["oracle_colormatch_ssim", "oracle_colormatch_psnr"]
CAVEAT = ("ORACLE: uses the TARGET image's per-channel mean/std. Not achievable at inference; "
          "not comparable to the honest baseline or any trained model.")


def stats(df, cols):
    out = {"n_pairs": int(len(df)), "n_plants": int(df["plant_id"].nunique())}
    for c in cols:
        out[c] = {"mean": float(df[c].mean()), "median": float(df[c].median())}
    return out


def strata(df):
    f1 = df.field == "Field1"
    return {
        "pooled": df,
        "pooled_excl_f1_day28": df[~df.d28],
        "Field1": df[f1],
        "Field1_excl_day28": df[f1 & ~df.d28],
        "Field1_day28_only": df[df.d28],
        "  Field1 target is day28 (bright->dark)": df[df.d28 & (df.target_day == 28)],
        "  Field1 input is day28 (dark->bright)": df[df.d28 & (df.input_last_day == 28)],
        "Field2": df[df.field == "Field2"],
    }


def table(res, cols, title, fmt):
    lines = [f"\n{title}", "| Subset | Pairs | " + " | ".join(f"{fmt[c]} mean (median)" for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for name, st in res.items():
        lines.append(f"| {name.strip()} | {st['n_pairs']:,} | " + " | ".join(
            f"{st[c]['mean']:.3f} ({st[c]['median']:.3f})" if "psnr" not in c
            else f"{st[c]['mean']:.2f} ({st[c]['median']:.2f})" for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair-csv", required=True)
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)[["pair_id", "input_last_day", "target_day"]]
    sc = pd.read_csv(args.per_pair_csv)
    missing = set(pairs.pair_id) - set(sc.pair_id)
    assert not missing, f"{len(missing)} cleaned pairs have no score (stale per-pair CSV?)"
    df = sc.merge(pairs, on="pair_id", how="inner")
    assert len(df) == len(pairs)
    print(f"Per-pair CSV: {len(sc)} rows; cleaned pair set: {len(pairs)}; dropped from CSV: {len(sc) - len(df)}")
    df["d28"] = (df.field == "Field1") & ((df.input_last_day == 28) | (df.target_day == 28))
    print("Pairs by split:", df.split.value_counts().to_dict(),
          "| day-28 pairs by split:", df[df.d28].split.value_counts().to_dict())

    test = df[df.split == "test"]
    st = strata(test)
    honest = {k: stats(v, HONEST) for k, v in st.items()}
    oracle = {k: stats(v, ORACLE) for k, v in st.items()}

    f1d, f1n = oracle["Field1_day28_only"], oracle["Field1_excl_day28"]
    hd, hn = honest["Field1_day28_only"], honest["Field1_excl_day28"]
    closure = {}
    for m in ["psnr", "ssim"]:
        o = f"oracle_colormatch_{m}"
        gap = hn[m]["mean"] - hd[m]["mean"]
        closure[m] = {"honest_day28": hd[m]["mean"], "honest_rest_of_field1": hn[m]["mean"],
                      "oracle_day28": f1d[o]["mean"], "oracle_rest_of_field1": f1n[o]["mean"],
                      "oracle_gain_day28": f1d[o]["mean"] - hd[m]["mean"],
                      "oracle_gain_rest_of_field1": f1n[o]["mean"] - hn[m]["mean"],
                      "gap_rest_minus_day28": gap,
                      "fraction_of_gap_closed": (f1d[o]["mean"] - hd[m]["mean"]) / gap if gap > 1e-9 else None}

    fmt = {"ssim": "RGB SSIM", "psnr": "RGB PSNR dB", "structure_ssim": "Structure SSIM",
           "oracle_colormatch_ssim": "ORACLE SSIM", "oracle_colormatch_psnr": "ORACLE PSNR dB"}
    print(table(honest, HONEST, "HONEST copy-forward baseline, TEST split (cleaned pairs)", fmt))
    print(table(oracle, ORACLE, "ORACLE colour-match (uses target statistics; NOT achievable, NOT comparable)", fmt))
    print("\nDay-28 cross-check, Field1 test (means): honest vs oracle")
    print(json.dumps(closure, indent=2))

    out = {"cleaned_pairs": df.split.value_counts().to_dict(),
           "day28_pairs_by_split": df[df.d28].split.value_counts().to_dict(),
           "test_copy_forward": {k.strip(): v for k, v in honest.items()},
           "test_oracle_colormatch": {"CAVEAT": CAVEAT, **{k.strip(): v for k, v in oracle.items()}},
           "field1_day28_oracle_crosscheck": closure}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
