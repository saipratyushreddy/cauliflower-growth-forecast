"""
Headless end-to-end test of the demo backend (and, with --ui, of app.py through Streamlit's AppTest): several plants (Field1 and Field2, short and long
histories), all drop levels, determinism, drop invariants, finite metrics, timing and, when saved per-pair CSVs are given, agreement of the app's drop-0
metrics with the README's per-pair scores.

Usage: python scripts/demo_selftest.py --bundle demo_bundle [--reference name=csv ...] [--ui]
  --reference Transformer=outputs/img_transformer_per_pair.csv ConvLSTM=outputs/img_convlstm_k4_per_pair.csv Copy-forward=outputs/img_stepB_copy_forward_per_pair.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from demo_core import DROP_LEVELS, METRICS, DemoBackend  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--ui", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-3, help="tolerance for drop-0 metrics vs the saved GPU per-pair scores (CPU vs GPU float differences)")
    a = ap.parse_args()
    t0 = time.time()
    be = DemoBackend(a.bundle)
    print(f"loaded bundle: {len(be.pairs)} test pairs, {len(be.plants())} plants ({time.time() - t0:.1f}s)")
    pl = be.plants()
    f1 = [p for p, f in pl if f == "Field1"]
    f2 = [p for p, f in pl if f == "Field2"]
    check("bundle has Field1 and Field2 test plants", bool(f1) and bool(f2), f"{len(f1)} Field1, {len(f2)} Field2")
    chosen = [(f1[0], None), (f2[0], None), (f2[min(3, len(f2) - 1)], None)]
    refs = {}
    for spec in a.reference:
        k, v = spec.split("=", 1)
        refs[k] = pd.read_csv(v).set_index("pair_id")
    for plant, _ in chosen:
        cuts = be.cutoffs(plant)
        ns = [n for n, _ in cuts]
        check(f"{plant}: cutoffs available", len(ns) >= 1, f"{ns}")
        picks = sorted({ns[0], ns[min(1, len(ns) - 1)], ns[-1]})   # shortest (1 image), a second, the longest history
        for n in picks:
            for drop in DROP_LEVELS:
                t = time.time()
                r = be.run(plant, n, drop)
                dt = time.time() - t
                r2 = be.run(plant, n, drop)
                same = all(np.array_equal(r["pred"][k], r2["pred"][k]) for k in r["pred"]) and r["kept"] == r2["kept"] and r["seed"] == r2["seed"]
                ok = np.isfinite([r["metrics"][m][k] for m in r["metrics"] for k in METRICS]).all()
                inv = (drop == 0 and not r["dropped"]) or (n == 1 and not r["dropped"]) or drop > 0
                inv = inv and sorted(r["kept"]) == r["kept"] and len(r["kept"]) >= 1 and set(r["kept"]) | set(r["dropped"]) == set(range(len(r["input_days"])))
                check(f"{plant} n={n} drop={drop}%: deterministic, invariants, finite metrics", same and ok and inv,
                      f"{dt:.1f}s; kept {r['kept']} of {len(r['input_days'])}; struct SSIM TF {r['metrics']['Transformer']['structure_ssim']:.3f} "
                      f"CL {r['metrics']['ConvLSTM']['structure_ssim']:.3f} CF {r['metrics']['Copy-forward']['structure_ssim']:.3f}")
            if refs:
                r = be.run(plant, n, 0)
                pid = f"{plant}::{r['target_day']}"
                for name, ref in refs.items():
                    if pid in ref.index:
                        d = max(abs(r["metrics"][name][m] - float(ref.loc[pid, m])) for m in ("structure_ssim", "ssim"))
                        dp = abs(r["metrics"][name]["psnr"] - float(ref.loc[pid, "psnr"]))
                        check(f"{plant} n={n}: drop-0 {name} matches the README per-pair scores", d <= a.tol and dp <= 20 * a.tol,
                              f"max |d| SSIM metrics {d:.2e}, PSNR {dp:.2e} dB")
    if a.ui:
        from streamlit.testing.v1 import AppTest
        os.environ["DEMO_BUNDLE"] = a.bundle
        at = AppTest.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py"), default_timeout=600)
        at.run()
        check("UI: first render without exception", not at.exception, str([e.value for e in at.exception]))
        # (AppTest cannot serialise values of sliders that use a display format, so widget state is set through session_state)
        for plant, _ in chosen:
            cuts = be.cutoffs(plant)
            for which, n in (("shortest", cuts[0][0]), ("longest", cuts[-1][0])):
                for drop in (0, 50):
                    at.session_state["plant"] = plant
                    at.run()
                    if len(cuts) > 1:
                        at.session_state["cutoff"] = n
                    at.session_state["drop"] = drop
                    at.run()
                    heads = [x.value for x in at.subheader]
                    shown = any(plant in h and f"day {dict(cuts)[n]}" in h for h in heads)
                    warn = any("not validated" in w.value for w in at.warning)
                    check(f"UI: {plant} {which} cutoff (n={n}) drop={drop}%: renders, right target, caveat {'shown' if drop else 'absent'}",
                          not at.exception and shown and (warn == (drop > 0)),
                          f"{len(at.exception)} exceptions; header {heads[:1]}; {len(at.image)} images")
    print(f"\n==== {'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'} ({time.time() - t0:.0f}s) ====")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
