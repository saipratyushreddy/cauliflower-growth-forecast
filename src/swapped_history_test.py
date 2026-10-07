"""
SWAPPED-HISTORY TEST (inference only, saved checkpoints): does the early-Field2 history gain depend on THIS plant's own earlier frames?

On the early Field2 subset (target days 16, 22), every EARLIER input frame (all but the last) is replaced by the frame of a DIFFERENT
plant of the same field at the same acquisition date (so the time offsets / conditioning are unchanged); the last input frame and the
target stay the plant's own. Donors are other TEST plants of the field (never seen in training); one donor per (plant, draw), several
random draws. Pairs with no valid donor are excluded and counted.

Arms, for the full Transformer: (a) original history, (b) swapped history (per draw and averaged over draws),
(c) SUPPLEMENTARY: earlier frames replaced by copies of the plant's own last frame (offsets kept).
The K=1 Transformer is run on the original and on the swapped histories as a sanity check (it reads only the last frame, so its scores
must be identical).

Usage:
    python src/swapped_history_test.py --image-pairs data/image_pairs.parquet --metadata data/metadata.parquet --cache-dir data \
        --out-dir outputs --tf-ckpt checkpoints/img_transformer_best.pt --tf-k1-ckpt checkpoints/img_transformer_k1_best.pt \
        --tf-csv outputs/img_transformer_per_pair.csv --tf-k1-csv outputs/img_transformer_k1_per_pair.csv --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from compare_generic_date import ratio_ci
from compare_temporal import boot_ci
from img_transformer import TemporalTransformerUNet
from train_img_single_frame import load_rows, score_pairs
from train_img_transformer import build_histories, predict


def load_model(ck, device):
    sd = torch.load(ck, map_location=device)
    a = sd["args"]
    m = TemporalTransformerUNet(d_model=a["d_model"], nhead=a["nhead"], num_layers=a["num_layers"], dim_ff=a["dim_ff"],
                                dropout=a["dropout"], grad_checkpoint=False).to(device)
    m.load_state_dict(sd["model"])
    return m, a.get("max_history", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tf-ckpt", required=True)
    ap.add_argument("--tf-k1-ckpt", required=True)
    ap.add_argument("--tf-csv", default=None)
    ap.add_argument("--tf-k1-csv", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--early-days", type=int, nargs="+", default=[16, 22])
    ap.add_argument("--field", default="Field2")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--donor-scope", choices=["test", "all"], default="test", help="'all' is for smoke tests only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = torch.device(args.device)

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    cache_idx = pd.read_parquet(os.path.join(args.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    meta = pd.read_parquet(args.metadata)
    frame = {(r.plant_id, int(r.day_after_planting)): (r.filepath, r.acquisition_date) for r in meta.itertuples()}
    te = pairs[pairs.split == "test"].sort_values("pair_id").reset_index(drop=True)
    sub = te[(te.field == args.field) & te.target_day.isin(args.early_days)].reset_index(drop=True)
    print(f"early {args.field} subset (targets {args.early_days}): {len(sub)} pairs, {sub.plant_id.nunique()} plants", flush=True)

    scope = pairs if args.donor_scope == "all" else te
    donor_plants = sorted(scope[scope.field == args.field].plant_id.unique())

    # donor candidates per plant: every earlier day (of ALL its subset pairs) must exist for the donor at the SAME acquisition date and be cached
    need = {}
    for r in sub.itertuples():
        need.setdefault(r.plant_id, set()).update(int(d) for d in list(r.input_days)[:-1])
    cands, excluded_plants = {}, []
    for pl, days in need.items():
        ok = []
        for dn in donor_plants:
            if dn == pl:
                continue
            good = all((dn, d) in frame and (pl, d) in frame and frame[(dn, d)][1] == frame[(pl, d)][1]
                       and frame[(dn, d)][0] in cache_idx for d in days)
            if good:
                ok.append(dn)
        cands[pl] = ok
        if not ok:
            excluded_plants.append(pl)
    excl = sub.plant_id.isin(excluded_plants)
    print(f"donor candidates per plant: min {min(len(v) for v in cands.values())}, median {int(np.median([len(v) for v in cands.values()]))}; "
          f"EXCLUDED pairs (no valid same-field/same-date donor): {int(excl.sum())} of {len(sub)}", flush=True)
    sub = sub[~excl].reset_index(drop=True)
    if sub.empty:
        print("nothing left to evaluate")
        return

    H_orig = build_histories(sub, cache_idx)                      # full histories (rows, offsets)
    rows_orig, offs = H_orig

    def swapped_rows(donors):
        out = []
        for r, rows in zip(sub.itertuples(), rows_orig):
            days = list(r.input_days)
            new = [cache_idx[frame[(donors[r.plant_id], int(d))][0]] for d in days[:-1]] + [rows[-1]]
            out.append(np.array(new, dtype=np.int64))
        return out

    rows_copy = [np.array([rows[-1]] * (len(rows) - 1) + [rows[-1]], dtype=np.int64) for rows in rows_orig]   # supplementary arm

    tf, mh_tf = load_model(args.tf_ckpt, device)
    k1, mh_k1 = load_model(args.tf_k1_ckpt, device)
    assert mh_tf == 0 and mh_k1 == 1

    def run(model, rows, cap=0):
        r = [x[-cap:] for x in rows] if cap else rows
        o = [x[-cap:] for x in offs] if cap else offs
        preds, wts = predict(model, arr, sub, (r, o), device)
        return score_pairs(preds, arr, sub), wts

    tf_orig, w_orig = run(tf, rows_orig)
    k1_orig, _ = run(k1, rows_orig, cap=1)
    for name, res, csv in [("Transformer", tf_orig, args.tf_csv), ("Transformer K=1", k1_orig, args.tf_k1_csv)]:
        if csv:
            ref = pd.read_csv(csv).set_index("pair_id").loc[sub.pair_id]
            d = np.abs(res.structure_ssim.values - ref.structure_ssim.values)
            print(f"integrity [{name}] original history vs saved structure SSIM: max |diff| {d.max():.2e}, mean {d.mean():.2e}", flush=True)

    rng = np.random.RandomState(0)
    draws, w_sw = [], []
    for k in range(args.draws):
        donors = {pl: cands[pl][rng.randint(len(cands[pl]))] for pl in need if cands[pl]}
        rows_sw = swapped_rows(donors)
        res, w = run(tf, rows_sw)
        draws.append(res)
        w_sw.append(w)
        if k == 0:
            k1_sw, _ = run(k1, rows_sw, cap=1)           # sanity: K=1 reads only the last frame
            diff = np.abs(k1_sw.structure_ssim.values - k1_orig.structure_ssim.values).max()
            print(f"sanity [K=1]: original vs swapped-history structure SSIM max |diff| = {diff:.1e} (must be 0)", flush=True)
    tf_copy, w_copy = run(tf, rows_copy)

    for lab, df in [("orig", tf_orig), ("k1", k1_orig), ("copylast", tf_copy)] + [(f"swap{k}", d) for k, d in enumerate(draws)]:
        df.to_csv(os.path.join(args.out_dir, f"{args.tag}img_swap_{lab}_per_pair.csv"), index=False)

    pl = sub.plant_id.values
    mets = [("structure_ssim", 4), ("ssim", 4), ("psnr", 3)]
    avg = {m: np.mean([d[m].values for d in draws], axis=0) for m, _ in mets}
    wl = lambda ws: float(np.mean([w[-1] for w in ws]))  # noqa: E731
    print(f"\n{len(sub)} pairs. Mean attention weight on the LAST frame: original {wl(w_orig):.3f} | swapped (mean over draws) "
          f"{np.mean([wl(w) for w in w_sw]):.3f} | copy-last-frame {wl(w_copy):.3f}")
    print("\nTable 1 -- test means on the subset (structure SSIM / RGB SSIM / PSNR dB)")
    print("| Arm | Structure SSIM | RGB SSIM | PSNR |\n|---|---|---|---|")
    print("| Transformer K=1 (single frame) | " + " | ".join(f"{k1_orig[m].mean():.{nd}f}" for m, nd in mets) + " |")
    print("| Transformer, original history | " + " | ".join(f"{tf_orig[m].mean():.{nd}f}" for m, nd in mets) + " |")
    print("| Transformer, SWAPPED history (mean over draws) | " + " | ".join(f"{avg[m].mean():.{nd}f}" for m, nd in mets) + " |")
    print("| (supplementary) Transformer, earlier frames = copies of own last frame | " + " | ".join(f"{tf_copy[m].mean():.{nd}f}" for m, nd in mets) + " |")

    print("\nTable 2 -- gap (Transformer minus its own K=1 control) on the subset; mean [95% plant-bootstrap CI], % pairs better")
    print("| Arm | Structure SSIM | RGB SSIM | PSNR dB | share of the original structure-SSIM gap |\n|---|---|---|---|---|")
    g0 = tf_orig.structure_ssim.values - k1_orig.structure_ssim.values

    def row(label, get):
        cells = []
        for m, nd in mets:
            g = get(m) - k1_orig[m].values
            lo, hi = boot_ci(g, pl)
            cells.append(f"{g.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g > 0).mean():.0%}")
        gs = get("structure_ssim") - k1_orig.structure_ssim.values
        rl, rh = ratio_ci(gs, g0, pl)
        print(f"| {label} | " + " | ".join(cells) + f" | {gs.mean() / g0.mean():.2f} [{rl:.2f}, {rh:.2f}] |")

    row("(a) original history", lambda m: tf_orig[m].values)
    for k, d in enumerate(draws):
        row(f"(b) swapped history, draw {k + 1}", lambda m, d=d: d[m].values)
    row("(b) swapped history, mean over draws", lambda m: avg[m])
    row("(c, supplementary) earlier frames = own last frame", lambda m: tf_copy[m].values)

    print("\nTable 3 -- swapped minus original history (Transformer, same pairs): the direct effect of the swap; mean [95% CI]")
    print("| Metric | swapped (mean over draws) - original |\n|---|---|")
    for m, nd in mets:
        g = avg[m] - tf_orig[m].values
        lo, hi = boot_ci(g, pl)
        print(f"| {m} | {g.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}], {(g > 0).mean():.0%} |")


if __name__ == "__main__":
    main()
