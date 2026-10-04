"""
Final comparison for the ConvLSTM runs, on the cleaned TEST pairs, per the project protocol: structure SSIM
against the best prior model stage (Step C), RGB SSIM/PSNR against Control 3, ALWAYS in the same table.

  Table 1  (headline, one table): per subset, structure SSIM [Step C | K=1 | K=4] and RGB SSIM / PSNR
           [Control 3 | K=1 | K=4], mean (median).
  Table 2  paired differences (mean (median), % of pairs where the first is better, 95% CI of the mean from
           a PLANT-level bootstrap) for: (1) K=1 vs Step C [architecture alone], (2) K=4 vs K=1 [history alone],
           (3) K=4 vs Step C [combined; headline], (4) K=1 vs Control 3, (5) K=4 vs Control 3 -- each on all
           three metrics.
  Table 3  K=4 minus K=1 by the number of history frames a pair actually has (1 frame => identical information
           for K=1 and K=4, so that row is the noise floor of the comparison).

Usage:
    python src/compare_temporal.py --image-pairs data/image_pairs.parquet \
        --step-c outputs/img_single_frame_per_pair.csv --control3 outputs/img_ctrl3_per_pair.csv \
        --k1 outputs/img_convlstm_k1_per_pair.csv --k4 outputs/img_convlstm_k4_per_pair.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from summarize_baseline import strata

METRICS = [("structure_ssim", "structure SSIM", 3), ("ssim", "RGB SSIM", 3), ("psnr", "RGB PSNR dB", 2)]


def load(path, test_ids):
    d = pd.read_csv(path)
    d = d[d.pair_id.isin(test_ids)].set_index("pair_id")
    assert set(d.index) == set(test_ids), f"{path}: not the same test pairs"
    return d.loc[sorted(test_ids)]


def boot_ci(delta, plants, n_boot=2000, seed=0):
    """95% CI of the mean paired difference, resampling PLANTS (pairs of a plant are correlated)."""
    df = pd.DataFrame({"d": delta, "p": plants}).dropna()
    g = df.groupby("p")["d"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--step-c", required=True)
    ap.add_argument("--control3", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--k4", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    te = pairs[pairs.split == "test"][["pair_id", "plant_id", "field", "input_last_day", "target_day", "n_input_frames"]]
    te = te.sort_values("pair_id").reset_index(drop=True)
    ids = set(te.pair_id)
    te["d28"] = (te.field == "Field1") & ((te.input_last_day == 28) | (te.target_day == 28))
    M = {"Step C": load(args.step_c, ids), "Control 3": load(args.control3, ids),
         "K=1": load(args.k1, ids), "K=4": load(args.k4, ids)}
    for k, v in M.items():
        assert list(v.index) == list(te.pair_id)
    print(f"{len(te)} test pairs, identical across Step C, Control 3, K=1, K=4\n")

    def frame(name, metric):
        return pd.Series(M[name][metric].values, index=te.index)

    def subs_of(df):
        return strata(df)

    # ---- Table 1 ----
    cols = [("structure_ssim", ["Step C", "K=1", "K=4"]), ("ssim", ["Control 3", "K=1", "K=4"]), ("psnr", ["Control 3", "K=1", "K=4"])]
    base = te.copy()
    for m, names in cols:
        for n in names:
            base[f"{m}|{n}"] = frame(n, m)
    S = subs_of(base)
    hdr = "| Subset | Pairs | " + " | ".join(f"{dict((a, b) for a, b, _ in METRICS)[m]}: {n}" for m, ns in cols for n in ns) + " |"
    print("TABLE 1 -- headline (test, mean (median)): structure SSIM vs Step C; RGB SSIM / PSNR vs Control 3")
    print(hdr)
    print("|---|---|" + "---|" * 9)
    for name, g in S.items():
        cells = []
        for m, ns in cols:
            nd = {"structure_ssim": 3, "ssim": 3, "psnr": 2}[m]
            for n in ns:
                v = g[f"{m}|{n}"]
                cells.append(f"{v.mean():.{nd}f} ({v.median():.{nd}f})")
        print(f"| {name.strip()} | {len(g):,} | " + " | ".join(cells) + " |")

    # ---- Table 2 ----
    comps = [("(1) K=1 vs Step C  [architecture alone]", "K=1", "Step C"),
             ("(2) K=4 vs K=1  [history alone, same architecture]", "K=4", "K=1"),
             ("(3) K=4 vs Step C  [combined; headline]", "K=4", "Step C"),
             ("(4) K=1 vs Control 3", "K=1", "Control 3"),
             ("(5) K=4 vs Control 3", "K=4", "Control 3")]
    out = {}
    for title, a, b in comps:
        print(f"\nTABLE 2{title[:3]} {title[4:]}: first minus second, paired; mean (median) [95% plant-bootstrap CI of mean], % pairs first better")
        print("| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |")
        print("|---|---|---|---|---|")
        d = te.copy()
        for m, _, _ in METRICS:
            d[m] = frame(a, m) - frame(b, m)
        for name, g in subs_of(d).items():
            cells = []
            for m, _, nd in METRICS:
                lo, hi = boot_ci(g[m].values, g["plant_id"].values)
                cells.append(f"{g[m].mean():+.{nd}f} ({g[m].median():+.{nd}f}) [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[m] > 0).mean():.0%}")
                out[f"{title[:3]}|{name.strip()}|{m}"] = {"mean": float(g[m].mean()), "ci": [float(lo), float(hi)],
                                                          "win": float((g[m] > 0).mean())}
            print(f"| {name.strip()} | {len(g):,} | " + " | ".join(cells) + " |")

    # ---- Table 3 ----
    print("\nTABLE 3 -- K=4 minus K=1 by number of history frames the pair has (n=1: identical information => noise floor)")
    print("| History frames | Pairs | d Structure SSIM mean (median) [95% CI], better | d RGB SSIM | d PSNR dB |")
    print("|---|---|---|---|---|")
    d = te.copy()
    for m, _, _ in METRICS:
        d[m] = frame("K=4", m) - frame("K=1", m)
    bucket = pd.cut(d.n_input_frames, [0, 1, 2, 3, 5, 99], labels=["1", "2", "3", "4-5", "6+"])
    for lab, g in d.groupby(bucket, observed=True):
        cells = []
        for m, _, nd in METRICS:
            lo, hi = boot_ci(g[m].values, g["plant_id"].values)
            cells.append(f"{g[m].mean():+.{nd}f} ({g[m].median():+.{nd}f}) [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g[m] > 0).mean():.0%}")
        print(f"| {lab} | {len(g):,} | " + " | ".join(cells) + " |")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
