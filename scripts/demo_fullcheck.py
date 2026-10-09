"""
Full sweep of the demo backend: EVERY (plant, cutoff) state of the bundle (all 1,233 test pairs) at drop 0 against the README's saved per-pair scores, and a no-crash /
invariant sweep over drop levels 25/50/75 on a regular sample. Prints maximum differences, pooled means (should equal the README's) and timing.

Usage: python scripts/demo_fullcheck.py --bundle demo_bundle --reference Transformer=... ConvLSTM=... Copy-forward=... [--drop-sample 5]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from demo_core import METRICS, DemoBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reference", nargs="+", required=True)
    ap.add_argument("--drop-sample", type=int, default=5, help="run drop levels 25/50/75 on every k-th state")
    a = ap.parse_args()
    refs = {s.split("=", 1)[0]: pd.read_csv(s.split("=", 1)[1]).set_index("pair_id") for s in a.reference}
    be = DemoBackend(a.bundle)
    t0 = time.time()
    rows, n_drop_runs, bad = [], 0, []
    states = [(p, n) for p, _ in be.plants() for n, _ in be.cutoffs(p)]
    print(f"{len(states)} states", flush=True)
    for i, (p, n) in enumerate(states):
        r = be.run(p, n, 0)
        pid = f"{p}::{r['target_day']}"
        row = {"pair_id": pid}
        for name in refs:
            for m in METRICS:
                row[f"{name}|{m}"] = r["metrics"][name][m]
        rows.append(row)
        if i % a.drop_sample == 0:
            for d in (25, 50, 75):
                rd = be.run(p, n, d)
                n_drop_runs += 1
                ok = np.isfinite([rd["metrics"][k][m] for k in rd["metrics"] for m in METRICS]).all() and len(rd["kept"]) >= 1 and sorted(rd["kept"]) == rd["kept"]
                if not ok:
                    bad.append((p, n, d))
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(states)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).set_index("pair_id")
    print(f"\nall {len(df)} drop-0 states ran; {n_drop_runs} drop>0 runs; invariant/finite failures: {len(bad)}")
    print("\napp (CPU) vs README per-pair scores (GPU), all test pairs:")
    print("| Method | Metric | max abs diff | pairs with diff > 5e-3 | app pooled mean | README pooled mean |\n|---|---|---|---|---|---|")
    for name, ref in refs.items():
        ref = ref.loc[df.index]
        for m in METRICS:
            x, y = df[f"{name}|{m}"].values, ref[m].values
            d = np.abs(x - y)
            print(f"| {name} | {m} | {np.nanmax(d):.2e} | {(d > 5e-3).sum()} | {np.nanmean(x):.5f} | {np.nanmean(y):.5f} |")
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
