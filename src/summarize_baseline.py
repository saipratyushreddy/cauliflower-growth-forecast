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

Model mode: --no-oracle summarizes a trained model's per-pair CSV (which has no oracle
columns; the oracle is never part of a model), and --baseline-per-pair-csv adds a direct
comparison against copy-forward on the SAME pairs (means/medians and paired differences).

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
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--label", default="copy-forward baseline")
    ap.add_argument("--baseline-per-pair-csv", default=None,
                    help="copy-forward per-pair CSV to compare this model against (paired, same pairs)")
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)[["pair_id", "split", "input_last_day", "target_day"]]
    sc = pd.read_csv(args.per_pair_csv)
    if args.no_oracle:  # a trained model is only ever scored on TEST: require all test pairs, expect only those
        pairs = pairs[pairs.split == "test"]
    pairs = pairs.drop(columns="split")
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
    oracle = {} if args.no_oracle else {k: stats(v, ORACLE) for k, v in st.items()}

    hd, hn = honest["Field1_day28_only"], honest["Field1_excl_day28"]
    closure = {}
    for m in ([] if args.no_oracle else ["psnr", "ssim"]):
        f1d, f1n = oracle["Field1_day28_only"], oracle["Field1_excl_day28"]
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
    print(table(honest, HONEST, f"HONEST {args.label}, TEST split (cleaned pairs)", fmt))
    if not args.no_oracle:
        print(table(oracle, ORACLE, "ORACLE colour-match (uses target statistics; NOT achievable, NOT comparable)", fmt))
        print("\nDay-28 cross-check, Field1 test (means): honest vs oracle")
        print(json.dumps(closure, indent=2))

    compare = {}
    if args.baseline_per_pair_csv:
        bl = pd.read_csv(args.baseline_per_pair_csv)
        bl = bl[bl.pair_id.isin(test.pair_id)]
        assert set(bl.pair_id) == set(test.pair_id), "baseline and model are not on the same test pairs"
        both = test.merge(bl[["pair_id"] + HONEST], on="pair_id", suffixes=("", "_base"))
        both["d28"] = both.pair_id.map(test.set_index("pair_id")["d28"])
        sb = strata(both)
        lines = [f"\nPAIRED comparison vs copy-forward on the same {len(both)} test pairs "
                 f"(delta = model - copy-forward; win = share of pairs where model is better)",
                 "| Subset | Pairs | dSSIM mean (median) | SSIM win | dPSNR dB mean (median) | PSNR win | dStructure SSIM mean (median) | Struct win |",
                 "|---|---|---|---|---|---|---|---|"]
        for name, g in sb.items():
            row = {"n_pairs": int(len(g))}
            cells = []
            for c in HONEST:
                dlt = g[c] - g[c + "_base"]
                row[c] = {"delta_mean": float(dlt.mean()), "delta_median": float(dlt.median()),
                          "win_rate": float((dlt > 0).mean()),
                          "model_mean": float(g[c].mean()), "base_mean": float(g[c + "_base"].mean())}
                nd = 2 if "psnr" in c else 3
                cells += [f"{dlt.mean():+.{nd}f} ({dlt.median():+.{nd}f})", f"{(dlt > 0).mean():.0%}"]
            compare[name.strip()] = row
            lines.append(f"| {name.strip()} | {len(g):,} | " + " | ".join(cells) + " |")
        print("\n".join(lines))

    out = {"cleaned_pairs": df.split.value_counts().to_dict(),
           "day28_pairs_by_split": df[df.d28].split.value_counts().to_dict(),
           "label": args.label,
           "test_results": {k.strip(): v for k, v in honest.items()}}
    if not args.no_oracle:
        out["test_copy_forward"] = out.pop("test_results")
        out["test_oracle_colormatch"] = {"CAVEAT": CAVEAT, **{k.strip(): v for k, v in oracle.items()}}
        out["field1_day28_oracle_crosscheck"] = closure
    if compare:
        out["paired_vs_copy_forward"] = compare
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
