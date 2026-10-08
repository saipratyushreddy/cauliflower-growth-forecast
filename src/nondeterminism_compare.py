"""
Nondeterminism test, comparison. Four runs of ONE configuration (Step C, seed 42, L40S): the original, the from-scratch fresh run, and two further
same-environment reruns (nd1, nd2). Six pairwise comparisons: first epoch at which the validation-loss curves differ by > 1e-6, epochs run,
selected epoch, pooled-mean differences of structure SSIM / RGB SSIM / PSNR, maximum per-pair difference, share of pairs that differ.

Decision rule (fixed BEFORE the reruns; "same-environment runs" = fresh, nd1, nd2, which share code, seed, data, venv and GPU model):
  * DIVERGENCE is the primary criterion. If nd1 and nd2 are near-identical to each other (no epoch with a val-loss difference > 1e-6 and max per-pair
    |difference| <= 1e-6) the training is reproducible within one environment, so the original-vs-fresh gap points to an ENVIRONMENT difference between
    the original and fresh runs (nd vs fresh and nd vs original then say which side the reruns agree with).
  * If the same-environment runs diverge from each other (a val-loss difference > 1e-6 at some epoch, per-pair scores differ), training is
    INTRINSICALLY nondeterministic in an unchanged environment, which suffices to produce the original-vs-fresh gap without any environment change.
  * Magnitude is then compared: SAME ORDER if the RMS of the three same-environment pairwise pooled-mean structure-SSIM differences is within a factor of 3
    of the RMS over all six pairs; otherwise reported as a different magnitude (no claim that the nondeterminism alone accounts for the size).

Usage: python src/nondeterminism_compare.py --orig <reference/outputs> --fresh <outputs> --nd1 <outputs_nd1> --nd2 <outputs_nd2>
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

METRICS = ["structure_ssim", "ssim", "psnr"]


def load(d):
    h = json.load(open(os.path.join(d, "img_single_frame_history.json")))
    p = pd.read_csv(os.path.join(d, "img_single_frame_per_pair.csv")).set_index("pair_id")
    return h, p


def main():
    ap = argparse.ArgumentParser()
    for k in ("orig", "fresh", "nd1", "nd2"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    R = {k: load(getattr(a, k)) for k in ("orig", "fresh", "nd1", "nd2")}
    print("| Run | epochs run | selected epoch | best val loss | test structure SSIM | test RGB SSIM | test PSNR |\n|---|---|---|---|---|---|---|")
    for k, (h, p) in R.items():
        b = min(h, key=lambda x: x["val"]["loss"])
        print(f"| {k} | {len(h)} | {b['epoch']} | {b['val']['loss']:.6f} | {p.structure_ssim.mean():.5f} | {p.ssim.mean():.5f} | {p.psnr.mean():.3f} |")
    print("\n| Pair | first epoch val loss differs by >1e-6 | pooled-mean diff structure SSIM | RGB SSIM | PSNR | max per-pair |diff| (structure SSIM) | pairs differing >1e-6 |\n|---|---|---|---|---|---|---|")
    rows = {}
    for x, y in itertools.combinations(R, 2):
        (h1, p1), (h2, p2) = R[x], R[y]
        n = min(len(h1), len(h2))
        first = next((i + 1 for i in range(n) if abs(h1[i]["val"]["loss"] - h2[i]["val"]["loss"]) > 1e-6), None)
        c = p1.index.intersection(p2.index)
        dm = {m: p2.loc[c, m].mean() - p1.loc[c, m].mean() for m in METRICS}
        dd = np.abs(p2.loc[c, "structure_ssim"].values - p1.loc[c, "structure_ssim"].values)
        rows[(x, y)] = (first, dm, dd.max(), (dd > 1e-6).mean())
        print(f"| {x} vs {y} | {first if first else 'none'} | {dm['structure_ssim']:+.5f} | {dm['ssim']:+.5f} | {dm['psnr']:+.3f} | {dd.max():.2e} | {(dd > 1e-6).mean():.0%} |")
    rms_all = np.sqrt(np.mean([r[1]["structure_ssim"] ** 2 for r in rows.values()]))
    same_env = [("fresh", "nd1"), ("fresh", "nd2"), ("nd1", "nd2")]
    rms_env = np.sqrt(np.mean([rows[k][1]["structure_ssim"] ** 2 for k in same_env]))
    print(f"\nRMS of the pooled-mean structure-SSIM differences: all six pairs {rms_all:.5f}; the three same-environment pairs {rms_env:.5f}; "
          f"original-vs-fresh {abs(rows[('orig', 'fresh')][1]['structure_ssim']):.5f}")
    diverged = [k for k in same_env if rows[k][0] is not None or rows[k][2] > 1e-6]
    ident = [k for k in same_env if rows[k][0] is None and rows[k][2] <= 1e-6]
    print("same-environment pairs that diverge: " + (", ".join(f"{x} vs {y} (first epoch {rows[(x, y)][0]}, max per-pair |d| {rows[(x, y)][2]:.2e})" for x, y in diverged) or "none"))
    print("same-environment pairs that are near-identical: " + (", ".join(f"{x} vs {y}" for x, y in ident) or "none"))
    if not diverged:
        print("VERDICT (rule): all same-environment runs are near-identical -> reproducible within one environment; the original-vs-fresh gap points to an ENVIRONMENT difference.")
    elif len(diverged) == len(same_env):
        order = "the SAME ORDER as" if (rms_all / 3 <= rms_env <= rms_all * 3) else "a DIFFERENT magnitude from"
        print(f"VERDICT (rule): the same-environment runs diverge from each other (all three pairs) -> INTRINSIC run-to-run nondeterminism in an unchanged environment; "
              f"their pooled-mean spread is {order} the all-pairs spread.")
    else:
        print("VERDICT (rule): mixed -- some same-environment pairs are identical and some diverge (e.g. a node- or time-dependent effect); see the numbers, no verdict.")

if __name__ == "__main__":
    main()
