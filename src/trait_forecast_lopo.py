"""
Leave-one-plot-out evaluation of the trait forecasts (design in docs/06, section "Held-out plots", written before running).

Per fold (test plot p of field F): calibration plot = the next plot of F in cyclic order (prediction ranges only); training = all other plots of both fields,
of which 15% of plants (fixed seed) are held out for early stopping and the ridge alpha. Tasks run where F has data: diameter (all folds), head diameter and
harvest (Field2 folds), height (Field1 folds). Methods and arms as in src/trait_forecast.py; the LSTM configuration per task and arm is FIXED to the plant-wise
run's choice (outputs/trait_forecast_grid.csv), seeds 42/43/44 averaged. Ranges: split-conformal on the calibration plot.

Outputs: outputs/lopo_per_pair.csv (every fold's test and calibration predictions), outputs/lopo_summary.csv, outputs/lopo_by_plot.csv,
outputs/lopo_vs_plantwise.csv (same 110 plant-wise test plants: plant-split vs held-out-plot prediction, paired).

Usage:
    python src/trait_forecast_lopo.py --traits data/traits.parquet --image-pairs data/image_pairs.parquet --exclude-list outputs/tiny_images_md5.txt \
        --embeddings-dir data/embeddings --grid outputs/trait_forecast_grid.csv --plantwise outputs/trait_forecast_per_pair.csv --out-dir outputs \
        --checkpoint-dir checkpoints --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cnn_lstm_train_core as L
from cnn_lstm_dataset import collate_fn
from cnn_lstm_model import CNNLSTM
from trait_forecast import (BASE, CUTOFFS, HARVEST_MEAS, SEEDS, SeqDataset, boot, conformal, fit_ridge, growth_table, harvest_pairs, last_features,
                            load_embeddings, next_pairs, plant_sequences)

TASK_FIELDS = {"diameter": ["Field1", "Field2"], "head_diameter": ["Field2"], "harvest": ["Field2"], "height": ["Field1"]}
VAL_FRAC = 0.15


def plot_of(pid):
    return pid.rsplit("_", 1)[0]


def stats(tr, harvest):
    st = {"day_m": float(np.mean(np.concatenate(tr.days.values))), "day_s": float(np.std(np.concatenate(tr.days.values))),
          "y_m": float(tr.y.mean()), "y_s": float(tr.y.std())}
    if harvest:
        mm = [np.concatenate([np.asarray(r.meas[j], dtype=float) for r in tr.itertuples()]) for j in range(len(HARVEST_MEAS))]
    else:
        gaps = np.concatenate([r.target_day - np.asarray(r.days) for r in tr.itertuples()])
        st.update(gap_m=float(gaps.mean()), gap_s=float(gaps.std()))
        mm = [np.concatenate([np.asarray(m, dtype=float) for ms in tr.meas for m in ms])]
    st.update(meas_m=[float(np.nanmean(x)) for x in mm], meas_s=[float(np.nanstd(x)) for x in mm])
    return st


def baselines(tr, va, evs, harvest):
    """{(arm, method): [pred for each df in evs]}; fitted on tr (va only for the ridge alpha of the harvest linear model)."""
    out = {}
    if harvest:
        prior, gm = tr.groupby("cutoff").y.mean(), tr.y.mean()
        cuts = sorted(tr.cutoff.unique())
        out[("image", "age_only")] = [x.cutoff.map(prior).fillna(gm).values for x in evs]
        def hx(x):
            return np.column_stack([x.last_head.fillna(tr.last_head.mean()), x.last_head.isna(), x.last_diam.fillna(tr.last_diam.mean()),
                                    pd.get_dummies(x.cutoff.astype(pd.CategoricalDtype(cuts))).values]).astype(float)
        mdl, _ = fit_ridge(hx(tr), tr.y.values, hx(va), va.y.values)
        out[("meas", "last_measurements_linear")] = [mdl.predict(hx(x)) for x in evs]
    else:
        age, fmean, gm = tr.groupby(["field", "target_day"]).y.mean(), tr.groupby("field").y.mean(), tr.y.mean()
        out[("image", "age_only")] = [np.array([age.get((f, t), fmean.get(f, gm)) for f, t in zip(x.field, x.target_day)]) for x in evs]
        out[("meas", "persistence")] = [x.last_value.values for x in evs]
        gs, gd = growth_table(tr, ["field", "last_day", "target_day"]), growth_table(tr, ["field", "target_day"])
        out[("meas", "last_plus_growth")] = [x.last_value.values + np.array([gs.get((f, l, t), gd.get((f, t), 0.0))
                                                                             for f, l, t in zip(x.field, x.last_day, x.target_day)]) for x in evs]
    return out


def run_fold(task, d, test_plot, cal_plot, emb, cfgs, args, device, ck_dir):
    harvest = task == "harvest"
    plot = d.plant_id.map(plot_of)
    te, ca = d[plot == test_plot].reset_index(drop=True), d[plot == cal_plot].reset_index(drop=True)
    rest = d[~plot.isin([test_plot, cal_plot])]
    plants = np.array(sorted(rest.plant_id.unique()))
    rng = np.random.RandomState(0)
    vp = set(rng.choice(plants, size=max(1, int(round(VAL_FRAC * len(plants)))), replace=False))
    tr, va = rest[~rest.plant_id.isin(vp)].reset_index(drop=True), rest[rest.plant_id.isin(vp)].reset_index(drop=True)
    if not harvest:   # evaluation / calibration sets: every method defined
        te, ca = te[te.last_value.notna()].reset_index(drop=True), ca[ca.last_value.notna()].reset_index(drop=True)
    if len(te) == 0 or len(ca) == 0:
        return []
    st = stats(tr, harvest)
    preds = baselines(tr, va, [ca, te], harvest)
    for arm in ("image", "meas"):
        mk = lambda x: SeqDataset(x, emb, st, use_gap=not harvest, use_meas=(arm == "meas"))
        dtr, dva, dca, dte = mk(tr), mk(va), mk(ca), mk(te)
        mdl, _ = fit_ridge(last_features(dtr), tr.y.values, last_features(dva), va.y.values)
        preds[(arm, "ridge")] = [mdl.predict(last_features(dca)), mdl.predict(last_features(dte))]
        g = cfgs[(task, arm)]
        cfg = {**BASE, "hidden_dim": g[0], "num_layers": g[1], "input_dim": dtr[0]["features"].shape[1]}
        if args.smoke:
            cfg["max_epochs"] = 2
        pc, pt = [], []
        for seed in (SEEDS[:1] if args.smoke else SEEDS):
            ck = os.path.join(ck_dir, f"{args.out_prefix}_{task}_{arm}_{test_plot}_s{seed}.pt")
            L.train_one_config(dtr, dva, cfg, device, ck, st["y_m"], st["y_s"], seed=seed, verbose=False)
            m = CNNLSTM(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
            m.load_state_dict(torch.load(ck, weights_only=True))
            for ds, ref, out in ((dca, ca, pc), (dte, te, pt)):
                p = L.predict(m, DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn), device, st["y_m"], st["y_s"]).set_index("pair_id")
                out.append(p.loc[ref.pair_id, "y_pred"].values)
        preds[(arm, "cnn_lstm")] = [np.mean(pc, 0), np.mean(pt, 0)]
    frames = []
    for (arm, meth), (p_ca, p_te) in preds.items():
        res = np.abs(np.asarray(p_ca, float) - ca.y.values)
        q80, q90 = conformal(res, 0.2), conformal(res, 0.1)
        hz = (ca.cutoff if harvest else ca.gap).values
        qh = {h: (conformal(res[hz == h], 0.2), conformal(res[hz == h], 0.1)) for h in np.unique(hz) if (hz == h).sum() >= 10}
        for role, x, p in (("calibration", ca, p_ca), ("test", te, p_te)):
            hx_ = (x.cutoff if harvest else x.gap).values
            frames.append(pd.DataFrame({"q80_h": [qh.get(h, (q80, q90))[0] for h in hx_], "q90_h": [qh.get(h, (q80, q90))[1] for h in hx_],"task": task, "fold": test_plot, "role": role, "arm": arm, "method": meth, "pair_id": x.pair_id, "plant_id": x.plant_id,
                                        "field": x.field, "plot": x.plant_id.map(plot_of), "horizon": x.cutoff if harvest else x.gap, "y_true": x.y,
                                        "y_pred": np.asarray(p, float), "q80": q80, "q90": q90}))
    print(f"[{task}] fold {test_plot}: train {len(tr)} / val {len(va)} / calibration {cal_plot} {len(ca)} / test {len(te)} pairs", flush=True)
    return frames


def summarise(pp, plantwise, out_dir, prefix="lopo"):
    t = pp[pp.role == "test"].copy()
    t["ae"] = (t.y_pred - t.y_true).abs()
    rows = []
    for (task, arm, meth), g in t.groupby(["task", "arm", "method"]):
        lo, hi = boot(g.ae.values, g.plant_id.values)
        bp = g.groupby("plot").ae.mean()
        rows.append({"task": task, "arm": arm, "method": meth, "n": len(g), "n_plants": g.plant_id.nunique(), "n_plots": g["plot"].nunique(), "mae": g.ae.mean(),
                     "ci_lo": lo, "ci_hi": hi, "plot_mae_min": bp.min(), "plot_mae_max": bp.max(), "cov80": (g.ae <= g.q80).mean(),
                     "width80": (2 * g.q80).mean(), "cov90": (g.ae <= g.q90).mean(), "width90": (2 * g.q90).mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)
    t.assign(in80=t.ae <= t.q80).groupby(["task", "arm", "method", "plot"]).agg(n=("ae", "size"), mae=("ae", "mean"), cov80=("in80", "mean")).reset_index() \
        .to_csv(os.path.join(out_dir, f"{prefix}_by_plot.csv"), index=False)
    cmp = []
    if plantwise is not None:
        pw = plantwise.assign(ae_pw=(plantwise.y_pred - plantwise.y_true).abs())[["task", "arm", "method", "pair_id", "plant_id", "ae_pw"]]
        j = t.merge(pw, on=["task", "arm", "method", "pair_id", "plant_id"])
        for (task, arm, meth), g in j.groupby(["task", "arm", "method"]):
            lo, hi = boot((g.ae - g.ae_pw).values, g.plant_id.values)
            cmp.append({"task": task, "arm": arm, "method": meth, "n": len(g), "n_plants": g.plant_id.nunique(), "mae_plant_split": g.ae_pw.mean(),
                        "mae_held_out_plot": g.ae.mean(), "diff": (g.ae - g.ae_pw).mean(), "ci_lo": lo, "ci_hi": hi})
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(os.path.join(out_dir, f"{prefix}_vs_plantwise.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    if len(cmp):
        print(cmp.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", required=True)
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--exclude-list", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--grid", required=True, help="outputs/trait_forecast_grid.csv from the plant-wise run (fixes the LSTM configuration)")
    ap.add_argument("--plantwise", default=None, help="outputs/trait_forecast_per_pair.csv, for the same-plants comparison")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", nargs="+", default=list(TASK_FIELDS))
    ap.add_argument("--harvest-cutoffs", type=int, nargs="+", default=None, help="cutoff days for the harvest task (default: trait_forecast.CUTOFFS)")
    ap.add_argument("--out-prefix", default="lopo", help="output file prefix")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 1 seed, first fold per field only")
    a = ap.parse_args()
    device = torch.device(a.device)
    traits = pd.read_parquet(a.traits)
    split_of = pd.read_parquet(a.image_pairs)[["plant_id", "split"]].drop_duplicates().set_index("plant_id")["split"].to_dict()
    bad = set(pd.read_csv(a.exclude_list, sep=r"\s+", header=None, names=["md5", "fp"])["fp"])
    seqs = plant_sequences(traits, bad)
    grid = pd.read_csv(a.grid)
    cfgs = {(r.task, r.arm): (int(r.hidden_dim), int(r.num_layers)) for r in grid.sort_values("val_mae").groupby(["task", "arm"]).head(1).itertuples()}
    print("fixed LSTM configurations (hidden, layers):", cfgs, flush=True)
    data = {t: (harvest_pairs(seqs, split_of, a.harvest_cutoffs or CUTOFFS) if t == "harvest" else next_pairs(seqs, split_of, t)) for t in a.tasks}
    emb = load_embeddings(a.embeddings_dir, sorted({f for d in data.values() for fps in d.filepaths for f in fps}))
    plots = {f: sorted({plot_of(p) for p in traits[traits.field == f].plant_id.unique()}) for f in ("Field1", "Field2")}
    print("plots:", plots, flush=True)
    ck_dir = os.path.join(a.checkpoint_dir, "trait_lopo")
    os.makedirs(ck_dir, exist_ok=True)
    frames = []
    for task in a.tasks:
        for f in TASK_FIELDS[task]:
            ps = plots[f]
            for i, p in enumerate(ps[:1] if a.smoke else ps):
                frames += run_fold(task, data[task], p, ps[(i + 1) % len(ps)], emb, cfgs, a, device, ck_dir)
                pd.concat(frames).to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_per_pair.csv"), index=False)
    pp = pd.concat(frames)
    summarise(pp, pd.read_csv(a.plantwise) if a.plantwise and os.path.exists(a.plantwise) else None, a.out_dir, a.out_prefix)


if __name__ == "__main__":
    main()
