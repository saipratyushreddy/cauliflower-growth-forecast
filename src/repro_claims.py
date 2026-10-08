"""
Claim-by-claim comparison of the README's quantitative statements (image track) with the same quantities recomputed from a from-scratch run.
For every claim: the original value (recomputed from the original per-pair scores, or taken from the saved job log for the swap / oracle claims),
the fresh value, the difference, and flags: SAME SIGN, CI-STATUS (does the 95% plant-bootstrap CI exclude 0 in each run), and whether the
difference is smaller than the original CI half-width. Diagnostic only; it edits nothing.

Usage: python src/repro_claims.py --orig-dir outputs --fresh-dir repro_fresh/outputs --image-pairs data/image_pairs.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_generic_date import ratio_ci  # noqa: E402
from compare_temporal import boot_ci, load  # noqa: E402

FILES = {"SC": "img_single_frame_per_pair.csv", "K1": "img_convlstm_k1_per_pair.csv", "K4": "img_convlstm_k4_per_pair.csv",
         "K4b": "img_convlstm_k4_seed43_per_pair.csv", "TF": "img_transformer_per_pair.csv", "TFK1": "img_transformer_k1_per_pair.csv",
         "C3": "img_ctrl3_per_pair.csv", "GEN": "img_ctrl_genericdate_per_pair.csv"}
M = "structure_ssim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--fresh-dir", required=True)
    ap.add_argument("--image-pairs", required=True)
    args = ap.parse_args()
    p = pd.read_parquet(args.image_pairs)
    te = p[p.split == "test"][["pair_id", "plant_id", "field", "target_day", "input_last_day", "n_input_frames"]].sort_values("pair_id").reset_index(drop=True)
    ids = set(te.pair_id)
    O = {k: load(os.path.join(args.orig_dir, f), ids) for k, f in FILES.items()}
    F = {k: load(os.path.join(args.fresh_dir, f), ids) for k, f in FILES.items()}
    pl = te.plant_id.values
    early = ((te.field == "Field2") & te.target_day.isin([16, 22])).values
    inp28 = ((te.field == "Field1") & (te.input_last_day == 28)).values
    one = (te.n_input_frames == 1).values
    allm = np.ones(len(te), bool)
    rows = []

    def gap(D, a, b, mask, metric=M):
        d = D[a][metric].values - (D[b][metric].values if b else 0)
        lo, hi = boot_ci(d[mask], pl[mask])
        return d[mask].mean(), lo, hi

    def add(claim, orig, fresh, nd=4, orig_ci=None, fresh_ci=None):
        same = np.sign(orig) == np.sign(fresh)
        ohw = (orig_ci[1] - orig_ci[0]) / 2 if orig_ci else None
        within = (abs(fresh - orig) <= ohw) if ohw else None
        ostat = ("excl 0" if (orig_ci[0] > 0 or orig_ci[1] < 0) else "incl 0") if orig_ci else ""
        fstat = ("excl 0" if (fresh_ci[0] > 0 or fresh_ci[1] < 0) else "incl 0") if fresh_ci else ""
        rows.append((claim, orig, fresh, fresh - orig, same, ostat, fstat, within, nd))

    def claim(label, a, b, mask, nd=4, metric=M):
        om, ol, oh = gap(O, a, b, mask, metric)
        fm, fl, fh = gap(F, a, b, mask, metric)
        add(label, om, fm, nd, (ol, oh), (fl, fh))

    mean = lambda D, k, mask=allm: D[k][M].values[mask].mean()  # noqa: E731
    for k, lab in [("SC", "Step C"), ("K1", "ConvLSTM K=1"), ("K4", "ConvLSTM K=4 s42"), ("K4b", "ConvLSTM K=4 s43"), ("TFK1", "Transformer K=1"), ("TF", "Transformer")]:
        add(f"A. structure SSIM, pooled: {lab}", mean(O, k), mean(F, k))
    claim("B1. ConvLSTM K=1 - Step C (architecture alone)", "K1", "SC", allm)
    claim("B2. ConvLSTM K=4 s42 - K=1 (history alone)", "K4", "K1", allm)
    claim("B3. ConvLSTM K=4 s43 - K=1 (history alone)", "K4b", "K1", allm)
    claim("B4. ConvLSTM K=4 s42 - Step C (combined)", "K4", "SC", allm)
    claim("C1. Transformer K=1 - Step C (architecture alone)", "TFK1", "SC", allm)
    claim("C2. Transformer - Transformer K=1 (history alone)", "TF", "TFK1", allm)
    claim("C3. Transformer - Step C (combined)", "TF", "SC", allm)
    claim("C4. Transformer - ConvLSTM K=4 s42", "TF", "K4", allm)
    claim("C5. Transformer - ConvLSTM K=4 s43", "TF", "K4b", allm)
    claim("D1. seed-to-seed: K=4 s43 - K=4 s42 (noise reference)", "K4b", "K4", allm)
    claim("D2. identical-input (1 history frame): K=4 s42 - K=1", "K4", "K1", one)
    claim("D3. identical-input (1 history frame): Transformer - Transformer K=1", "TF", "TFK1", one)
    claim("E1. early Field2 (n=146): Transformer - Transformer K=1", "TF", "TFK1", early)
    claim("E2. early Field2: Transformer - ConvLSTM K=4 s42", "TF", "K4", early)
    claim("E3. early Field2: Transformer - ConvLSTM K=4 s43", "TF", "K4b", early)
    claim("E4. OTHER pairs: Transformer - Transformer K=1", "TF", "TFK1", ~early)
    claim("E5. OTHER pairs: Transformer - ConvLSTM K=4 s42", "TF", "K4", ~early)
    claim("F1. early Field2: generic-date - Transformer K=1", "GEN", "TFK1", early)
    for lab, D in [("orig", O), ("fresh", F)]:
        g, t = D["TF"][M].values - D["TFK1"][M].values, D["GEN"][M].values - D["TFK1"][M].values
        D["_share"] = (t[early].mean() / g[early].mean(), *ratio_ci(t[early], g[early], pl[early]))
        D["_early_share_of_pooled"] = g[early].sum() / g.sum()
    add("F2. share of the early-Field2 Transformer gap reproduced by generic-date", O["_share"][0], F["_share"][0], 2, O["_share"][1:], F["_share"][1:])
    add("E6. early-Field2 pairs' share of the Transformer's pooled history effect", O["_early_share_of_pooled"], F["_early_share_of_pooled"], 2)
    claim("G1. day-28-input pairs (n=37): ConvLSTM K=4 s42 - K=1 (untested hypothesis)", "K4", "K1", inp28)
    claim("G2. day-28-input pairs: ConvLSTM K=4 s43 - K=1", "K4b", "K1", inp28)
    claim("H1. protocol: Transformer - Control 3 (structure SSIM)", "TF", "C3", allm)
    claim("H2. protocol: Step C - Control 3 (structure SSIM)", "SC", "C3", allm)
    claim("H3. protocol: Transformer - Control 3, RGB SSIM", "TF", "C3", allm, 4, "ssim")
    claim("H4. protocol: Transformer - Control 3, PSNR dB", "TF", "C3", allm, 3, "psnr")
    # group claim
    og = [mean(O, k) for k in ("K4", "K4b", "TF")]; on = [mean(O, k) for k in ("SC", "K1", "TFK1")]
    fg = [mean(F, k) for k in ("K4", "K4b", "TF")]; fn = [mean(F, k) for k in ("SC", "K1", "TFK1")]
    add("I1. every history run above every no-history run: min(history) - max(no-history)", min(og) - max(on), min(fg) - max(fn))
    add("I2. mean(history runs) - mean(no-history runs)", np.mean(og) - np.mean(on), np.mean(fg) - np.mean(fn))

    # swap / oracle: original values from the saved README / job logs (the original score files for these arms are not held locally)
    S = {k: pd.read_csv(os.path.join(args.fresh_dir, f"img_swap_{k}_per_pair.csv")).set_index("pair_id") for k in ["orig", "k1", "copylast", "swap0", "swap1", "swap2", "swap3", "swap4"]}
    sub = S["orig"].index
    ps = te.set_index("pair_id").loc[sub, "plant_id"].values
    k1 = S["k1"].loc[sub, M].values
    sw = np.mean([S[f"swap{i}"].loc[sub, M].values for i in range(5)], axis=0)
    def swapgap(arr):
        g = arr - k1
        lo, hi = boot_ci(g, ps)
        return g.mean(), (lo, hi)
    g0, c0 = swapgap(S["orig"].loc[sub, M].values)
    add("S1. swap test (146 pairs): original-history gap vs K=1 [README +0.0160]", 0.0160, g0, 4, (0.0143, 0.0178), c0)
    g1, c1 = swapgap(sw)
    add("S2. swap test: swapped-history gap vs K=1, mean of 5 draws [README -0.0306]", -0.0306, g1, 4, (-0.0329, -0.0285), c1)
    g2, c2 = swapgap(S["copylast"].loc[sub, M].values)
    add("S3. swap test: earlier frames = own last frame, gap vs K=1 [README -0.0272]", -0.0272, g2, 4, (-0.0294, -0.0250), c2)
    d = sw - S["orig"].loc[sub, M].values
    lo, hi = boot_ci(d, ps)
    add("S4. swap test: swapped minus original history [README -0.0466]", -0.0466, d.mean(), 4, (-0.0491, -0.0442), (lo, hi))
    OR = {k: pd.read_csv(os.path.join(args.fresh_dir, f"img_oracle_cm_{k}_per_pair.csv")).set_index("pair_id") for k in
          ["TF_before", "TFK1_before", "TF_after", "TFK1_after", "TF_aftermean", "TFK1_aftermean"]}
    pe = te.set_index("pair_id").loc[OR["TF_before"].index]
    em = ((pe.field == "Field2") & pe.target_day.isin([16, 22])).values
    pp = pe.plant_id.values
    def og_(tag):
        g = OR[f"TF_{tag}"][M].values - OR[f"TFK1_{tag}"][M].values
        lo, hi = boot_ci(g[em], pp[em])
        return g[em].mean(), g[em], (lo, hi)
    gb, vb, cb = og_("before")
    ga, va, ca = og_("after")
    gm, vm, cm = og_("aftermean")
    add("O1. oracle: early-Field2 gap BEFORE colour-matching [README +0.0160]", 0.0160, gb, 4, (0.0143, 0.0178), cb)
    add("O2. oracle: gap AFTER mean+std match [README +0.0159]", 0.0159, ga, 4, (0.0141, 0.0178), ca)
    add("O3. oracle: gap AFTER mean-only match [README +0.0161]", 0.0161, gm, 4, (0.0143, 0.0179), cm)
    add("O4. oracle: share of the gap remaining after mean+std [README 0.99 [0.97,1.01]]", 0.99, ga / gb, 2, (0.97, 1.01), tuple(ratio_ci(va, vb, pp[em])))

    print("| Claim | Original / README | Fresh run | Fresh - original | Same sign | CI vs 0 (orig / fresh) | |diff| < original CI half-width |")
    print("|---|---|---|---|---|---|---|")
    for claim_, o, f, d_, same, os_, fs_, within, nd in rows:
        print(f"| {claim_} | {o:+.{nd}f} | {f:+.{nd}f} | {d_:+.{nd}f} | {'yes' if same else '**NO**'} | {os_} / {fs_} | {'' if within is None else ('yes' if within else 'no')} |")


if __name__ == "__main__":
    main()
