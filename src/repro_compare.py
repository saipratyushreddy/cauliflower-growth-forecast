"""
Reproducibility pass, STEP C: compare a from-scratch re-run (FRESH) with the original run (REFERENCE) and with the README.
Reports PASS / FAIL per run and diagnostic; it never edits the README and never guesses a cause for a mismatch.

  1  per-pair score files (every method, control, diagnostic arm): aligned on pair_id; max / mean abs difference of structure SSIM, RGB SSIM and
     PSNR, pooled mean and median at full precision. PASS = identical or equal within 1e-6; otherwise FAIL with the exact differences.
  2  training histories: epochs run, selected epoch, best val loss / SSIM / PSNR, and the FIRST epoch at which the val-loss curves differ by > 1e-6
  3  every table printed by the comparison scripts that produced the README (compare_temporal, compare_seeds, compare_transformer_k1,
     compare_generic_date, summarize_baseline x2) and by the controls / oracle / swap job logs: re-run on both trees, compared row by row on the
     DISPLAYED values (the precision shown in the README); PASS = every row text-identical
  4  README: numeric tokens of each output row are searched in the README tables. Rows of the ORIGINAL run that the README does not contain
     are the README's own coverage gap; PASS = no row that the README contains for the original run is missing for the fresh run

Usage:
    python src/repro_compare.py --fresh-dir <fresh clone> --ref-dir <fresh clone>/reference --readme README.md --image-pairs data/image_pairs.parquet \
        --report repro_report.json
    (layout: <dir>/outputs/*.csv|json and the job logs in <dir>/ (fresh) or <ref-dir>/logs (reference))
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
METRICS = ["structure_ssim", "ssim", "psnr"]
NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?|[-+]?\.\d+")
RESULTS = []


def record(item, status, detail=""):
    RESULTS.append({"item": item, "status": status, "detail": detail})
    print(f"[{status}] {item}" + (f" -- {detail}" if detail else ""), flush=True)


def pair_csvs():
    L = {"copy-forward": "img_stepB_copy_forward_per_pair.csv", "Control 1 (blur)": "img_ctrl1_per_pair.csv",
         "Control 2 (flight colour)": "img_ctrl2_per_pair.csv", "Control 3 (colour+blur)": "img_ctrl3_per_pair.csv",
         "generic-date control": "img_ctrl_genericdate_per_pair.csv", "Step C (single frame)": "img_single_frame_per_pair.csv",
         "ConvLSTM K=1": "img_convlstm_k1_per_pair.csv", "ConvLSTM K=4 seed 42": "img_convlstm_k4_per_pair.csv",
         "ConvLSTM K=4 seed 43": "img_convlstm_k4_seed43_per_pair.csv", "Transformer (full)": "img_transformer_per_pair.csv",
         "Transformer K=1": "img_transformer_k1_per_pair.csv"}
    for m in ("TF", "TFK1"):
        for lab in ("before", "after", "aftermean"):
            L[f"oracle colour-match {m} {lab}"] = f"img_oracle_cm_{m}_{lab}_per_pair.csv"
    for lab in ["orig", "k1", "copylast"] + [f"swap{k}" for k in range(5)]:
        L[f"swapped-history test arm '{lab}'"] = f"img_swap_{lab}_per_pair.csv"
    return L


def histories():
    return {"Step C": "img_single_frame_history.json", "ConvLSTM K=1": "img_convlstm_k1_history.json",
            "ConvLSTM K=4 seed 42": "img_convlstm_k4_history.json", "ConvLSTM K=4 seed 43": "img_convlstm_k4_seed43_history.json",
            "Transformer (full)": "img_transformer_history.json", "Transformer K=1": "img_transformer_k1_history.json"}


def compare_pairs(fo, ro):
    for name, f in pair_csvs().items():
        a, b = os.path.join(fo, f), os.path.join(ro, f)
        if not (os.path.exists(a) and os.path.exists(b)):
            record(f"per-pair scores: {name}", "SKIP", f"missing {'fresh' if not os.path.exists(a) else 'reference'} file {f}")
            continue
        fr, rf = pd.read_csv(a).set_index("pair_id"), pd.read_csv(b).set_index("pair_id")
        common = fr.index.intersection(rf.index)
        only_f, only_r = len(fr.index.difference(rf.index)), len(rf.index.difference(fr.index))
        if len(common) == 0:
            record(f"per-pair scores: {name}", "FAIL", "no common pair_id")
            continue
        parts, worst = [], 0.0
        for m in METRICS:
            x, y = fr.loc[common, m].astype(float).values, rf.loc[common, m].astype(float).values
            mask = ~(np.isnan(x) & np.isnan(y))
            d = np.abs(x[mask] - y[mask])
            worst = max(worst, np.nanmax(d) if len(d) else 0.0)
            parts.append(f"{m}: max|d|={np.nanmax(d):.2e} mean fresh {np.nanmean(x):.5f} vs ref {np.nanmean(y):.5f}, median {np.nanmedian(x):.5f} vs {np.nanmedian(y):.5f}")
        status = "PASS" if worst <= 1e-6 else "FAIL"
        note = f"{len(common)} pairs" + (f" (+{only_f} only in fresh, +{only_r} only in reference: not compared)" if only_f or only_r else "")
        tier = "identical" if worst == 0 else ("equal within 1e-6" if worst <= 1e-6 else "DIFFERENT")
        record(f"per-pair scores: {name}", status, f"{tier}; {note}; " + "; ".join(parts))


def compare_histories(fo, ro):
    for name, f in histories().items():
        a, b = os.path.join(fo, f), os.path.join(ro, f)
        if not (os.path.exists(a) and os.path.exists(b)):
            record(f"training history: {name}", "SKIP", "missing file")
            continue
        h1, h2 = json.load(open(a)), json.load(open(b))
        be = lambda h: min(h, key=lambda x: x["val"]["loss"])  # noqa: E731
        b1, b2 = be(h1), be(h2)
        n = min(len(h1), len(h2))
        first = next((i + 1 for i in range(n) if abs(h1[i]["val"]["loss"] - h2[i]["val"]["loss"]) > 1e-6), None)
        same = len(h1) == len(h2) and b1["epoch"] == b2["epoch"] and first is None
        record(f"training history: {name}", "PASS" if same else "FAIL",
               f"epochs run {len(h1)} vs {len(h2)}; selected epoch {b1['epoch']} vs {b2['epoch']}; best val loss {b1['val']['loss']:.6f} vs {b2['val']['loss']:.6f}; "
               f"val SSIM {b1['val']['ssim']:.4f} vs {b2['val']['ssim']:.4f}; PSNR {b1['val']['psnr']:.2f} vs {b2['val']['psnr']:.2f}; "
               f"first epoch where val loss differs by >1e-6: {first if first else 'none'}")


def table_rows(text):
    rows = []
    for l in text.splitlines():
        l = l.rstrip()
        if l.startswith("|") and not re.fullmatch(r"\|[\s\-|:]+\|", l):
            rows.append(l)
    return rows


def tokens(row):
    """Numbers in the row's cells; a leading text label (any letter) is skipped so that digits inside labels do not matter."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if cells and re.search(r"[A-Za-z]", cells[0]):
        cells = cells[1:]
    return [t.replace(",", "") for t in NUM.findall(" | ".join(cells))]


def run_script(script, args):
    cp = subprocess.run([sys.executable, os.path.join(SRC, script)] + args, cwd=SRC, capture_output=True, text=True)
    return cp.stdout, cp.returncode, cp.stderr


def script_jobs(od, scratch):
    P = lambda f: os.path.join(od, f)  # noqa: E731
    return {
        "compare_temporal.py (ConvLSTM tables)": ("compare_temporal.py", ["--image-pairs", IMAGE_PAIRS, "--step-c", P("img_single_frame_per_pair.csv"), "--control3", P("img_ctrl3_per_pair.csv"),
                                                   "--k1", P("img_convlstm_k1_per_pair.csv"), "--k4", P("img_convlstm_k4_per_pair.csv")]),
        "compare_seeds.py (seed tables)": ("compare_seeds.py", ["--image-pairs", IMAGE_PAIRS, "--k1", P("img_convlstm_k1_per_pair.csv"), "--k4-a", P("img_convlstm_k4_per_pair.csv"),
                                              "--k4-b", P("img_convlstm_k4_seed43_per_pair.csv"), "--step-c", P("img_single_frame_per_pair.csv"), "--control3", P("img_ctrl3_per_pair.csv")]),
        "compare_transformer_k1.py (Transformer tables)": ("compare_transformer_k1.py", ["--image-pairs", IMAGE_PAIRS, "--step-c", P("img_single_frame_per_pair.csv"), "--control3", P("img_ctrl3_per_pair.csv"),
                                                          "--k1", P("img_convlstm_k1_per_pair.csv"), "--k4", P("img_convlstm_k4_per_pair.csv"), "--k4-seed2", P("img_convlstm_k4_seed43_per_pair.csv"),
                                                          "--tf", P("img_transformer_per_pair.csv"), "--tf-k1", P("img_transformer_k1_per_pair.csv")]),
        "compare_generic_date.py (generic-date tables)": ("compare_generic_date.py", ["--image-pairs", IMAGE_PAIRS, "--generic", P("img_ctrl_genericdate_per_pair.csv"), "--copy-forward", P("img_stepB_copy_forward_per_pair.csv"),
                                                          "--c1", P("img_ctrl1_per_pair.csv"), "--c2", P("img_ctrl2_per_pair.csv"), "--c3", P("img_ctrl3_per_pair.csv"), "--step-c", P("img_single_frame_per_pair.csv"),
                                                          "--k1", P("img_convlstm_k1_per_pair.csv"), "--k4", P("img_convlstm_k4_per_pair.csv"), "--k4-seed2", P("img_convlstm_k4_seed43_per_pair.csv"),
                                                          "--tf", P("img_transformer_per_pair.csv"), "--tf-k1", P("img_transformer_k1_per_pair.csv")]),
        "summarize_baseline.py (copy-forward + oracle tables)": ("summarize_baseline.py", ["--per-pair-csv", P("img_stepB_copy_forward_per_pair.csv"), "--image-pairs", IMAGE_PAIRS, "--out", os.path.join(scratch, "sb_cf.json")]),
        "summarize_baseline.py (Step C table, paired vs copy-forward)": ("summarize_baseline.py", ["--per-pair-csv", P("img_single_frame_per_pair.csv"), "--image-pairs", IMAGE_PAIRS, "--no-oracle", "--label", "single-frame FiLM-UNet",
                                                                        "--baseline-per-pair-csv", P("img_stepB_copy_forward_per_pair.csv"), "--out", os.path.join(scratch, "sb_sc.json")]),
    }


def pick_log(dirs, patterns, must):
    cands = []
    for d in dirs:
        for pat in patterns:
            cands += glob.glob(os.path.join(d, pat))
    good = [c for c in cands if must in open(c, errors="ignore").read()]
    return max(good, key=os.path.getmtime) if good else None


def row_compare(item, fresh_rows, ref_rows, readme_tok):
    n = min(len(fresh_rows), len(ref_rows))
    diff = []
    for i in range(n):
        if fresh_rows[i] != ref_rows[i]:
            tf, tr = tokens(fresh_rows[i]), tokens(ref_rows[i])
            md = max((abs(float(a) - float(b)) for a, b in zip(tf, tr)), default=0.0) if len(tf) == len(tr) else float("nan")
            diff.append((i, md, fresh_rows[i], ref_rows[i]))
    same_len = len(fresh_rows) == len(ref_rows)
    status = "PASS" if same_len and not diff else "FAIL"
    detail = f"{len(fresh_rows)} fresh rows vs {len(ref_rows)} reference rows; {len(diff)} differ at the displayed precision"
    if diff:
        worst = max(diff, key=lambda d: (np.nan_to_num(d[1], nan=1e9)))
        detail += f"; largest numeric difference {worst[1]:.4g} in row {worst[0]}: FRESH `{worst[2][:140]}` vs REF `{worst[3][:140]}`"
    record(item, status, detail)
    # README: rows the README contains for the reference run but not for the fresh run
    def in_readme(r):
        t = tokens(r)
        return any(len(t) <= len(rt) and any(rt[i:i + len(t)] == t for i in range(len(rt) - len(t) + 1)) for rt in readme_tok)
    ref_in = [in_readme(r) for r in ref_rows]          # per ROW (lists, not dicts: identical rows are counted each time)
    fresh_in = [in_readme(r) for r in fresh_rows]
    n_ref_in, n_fr_in = sum(ref_in), sum(fresh_in)
    # a row that the README contains in the reference run but whose fresh counterpart (same row index) is absent from the README
    mism = [(i, fresh_rows[i]) for i in range(n) if ref_in[i] and not fresh_in[i]]
    record(f"README match: {item}", "PASS" if not mism and same_len else "FAIL",
           f"reference rows found in README {n_ref_in}/{len(ref_rows)}; fresh rows found in README {n_fr_in}/{len(fresh_rows)}; "
           f"rows that match the README in the reference run but not in the fresh run: {len(mism)}" + (f" (first: `{mism[0][1][:140]}`)" if mism else ""))


def main():
    global IMAGE_PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--readme", required=True)
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--scratch-dir", default=None)
    args = ap.parse_args()
    t0 = time.time()
    IMAGE_PAIRS = os.path.abspath(args.image_pairs)
    fo, ro = os.path.join(args.fresh_dir, "outputs"), os.path.join(args.ref_dir, "outputs")
    scratch = args.scratch_dir or os.path.join(args.fresh_dir, "repro_scratch")
    os.makedirs(scratch, exist_ok=True)
    rtok = [tokens(l) for l in table_rows(open(args.readme).read())]

    print("== 1. per-pair scores ==")
    compare_pairs(fo, ro)
    print("\n== 2. training histories ==")
    compare_histories(fo, ro)
    print("\n== 3/4. tables printed by the scripts that produced the README ==")
    for item, (script, a) in script_jobs(fo, scratch).items():
        _, ra = script_jobs(ro, scratch)[item]
        of, rcf, ef = run_script(script, a)
        orf, rcr, er = run_script(script, ra)
        if rcf != 0 or rcr != 0:
            record(item, "SKIP", f"script failed (fresh rc={rcf}, reference rc={rcr}): {(ef if rcf else er).strip().splitlines()[-1][:160] if (ef or er).strip() else ''}")
            continue
        row_compare(item, table_rows(of), table_rows(orf), rtok)
    flogs = [args.fresh_dir]
    rlogs = [os.path.join(args.ref_dir, "logs"), ro, args.ref_dir]
    for item, pats_f, pats_r, must in [
        ("controls job log (master tables, paired tables, per-control summaries)", ["rp-controls_*.out", "img-ctrls_*.out"], ["img-ctrls_*.out"], "TEST EVALUATION (CONTROL 3"),
        ("oracle colour-match job log", ["rp-oracle_*.out", "img-oracle-cm_*.out"], ["img-oracle-cm_*.out"], "ORACLE diagnostic"),
        ("swapped-history job log", ["rp-swap_*.out", "img-swap-hist_*.out"], ["img-swap-hist_*.out"], "Table 1 -- test means"),
        ("generic-date job log (printed means)", ["rp-genericdate_*.out"], ["img-genericdate_*.out"], "TEST EVALUATION (generic-date")]:
        lf, lr = pick_log(flogs, pats_f, must), pick_log(rlogs, pats_r, must)
        if not lf or not lr:
            record(item, "SKIP", f"log not found ({'fresh' if not lf else 'reference'})")
            continue
        row_compare(item, table_rows(open(lf).read()), table_rows(open(lr).read()), rtok)
    # controls: validation sigma curve
    fa, rb = os.path.join(fo, "img_ctrl1_val_sigma_curve.csv"), os.path.join(ro, "img_ctrl1_val_sigma_curve.csv")
    if os.path.exists(fa) and os.path.exists(rb):
        d = (pd.read_csv(fa) - pd.read_csv(rb)).abs().max()
        record("Control 1 validation sigma curve (15 sigmas)", "PASS" if d.max() <= 1e-6 else "FAIL", f"max |diff| per column: {d.round(8).to_dict()}")
    else:
        record("Control 1 validation sigma curve", "SKIP", "missing file")
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_skip = sum(r["status"] == "SKIP" for r in RESULTS)
    print(f"\n==== STEP C: {sum(r['status'] == 'PASS' for r in RESULTS)} PASS, {n_fail} FAIL, {n_skip} SKIP ({time.time() - t0:.0f}s) ====")
    json.dump({"results": RESULTS, "elapsed_sec": time.time() - t0}, open(args.report, "w"), indent=1)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
