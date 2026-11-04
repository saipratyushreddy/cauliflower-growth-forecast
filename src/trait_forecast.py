"""
Direction 2: multi-trait forecasting with prediction ranges (design in docs/06-multi-trait-and-trait-link.md, written before any run).

Tasks (plant-wise split frozen from Phase 0, the 739th plant in train as in the image track; black/placeholder photos removed from inputs):
  next-measurement   diameter, head_diameter, height: predict the trait at the next acquisition date from all earlier photos.
  harvest            Field2 only: days from a cutoff date (44, 50, 57, 65) to the plant's first harvest mark, from the photos up to the cutoff.

Arms (kept separate throughout; never compared as if they had the same information):
  image   photos + photo days (+ forecast gap for next-measurement tasks). No manual measurement.
  meas    image arm + every earlier manual measurement of the trait (harvest: of head diameter and diameter), with a missing flag.

Methods per arm:
  baselines   image arm: age-only (train mean per field and target day; harvest: train mean per cutoff).
              meas arm: persistence, last measurement + mean train growth for that step (harvest: linear model on the last head diameter and diameter).
  ridge       on the last photo's embedding + days (+ last measurement, meas arm); alpha on validation.
  cnn_lstm    the diameter CNN-LSTM with extra per-photo inputs; config chosen on validation MAE from hidden {64,128,256} x layers {1,2} (dropout 0.2, lr 1e-3,
              seed 42), then the chosen config trained with seeds 42, 43, 44; test predictions are the mean of the 3 seeds.
Prediction ranges: split-conformal. q_a = the ceil((n+1)(1-a))-th smallest validation absolute residual; interval = prediction +/- q_a for a = 0.2 and 0.1;
reported: test coverage and mean width, overall and by field. Validation pairs of one plant are not independent, so coverage is approximate.
Evaluation set: test pairs where every method of the task is defined (next-measurement: the last input date has a measurement of the trait).

Outputs: outputs/trait_forecast_results.csv, outputs/trait_forecast_paired.csv, outputs/trait_forecast_per_pair.csv, outputs/trait_forecast_grid.csv.

Usage:
    python src/trait_forecast.py --traits data/traits.parquet --image-pairs data/image_pairs.parquet --exclude-list outputs/tiny_images_md5.txt \
        --embeddings-dir data/embeddings --out-dir outputs --checkpoint-dir checkpoints --device cuda
    python src/trait_forecast.py ... --counts-only        # pair counts per task and split; needs no embeddings
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Dataset

import cnn_lstm_train_core as L
from cnn_lstm_dataset import collate_fn
from cnn_lstm_model import CNNLSTM

NEXT_TRAITS = ["diameter", "head_diameter", "height"]
CUTOFFS = [44, 50, 57, 65]
HARVEST_MEAS = ["head_diameter", "diameter"]
GRID = [{"hidden_dim": h, "num_layers": n} for h, n in itertools.product([64, 128, 256], [1, 2])]
BASE = {"dropout": 0.2, "lr": 1e-3, "batch_size": 64, "max_epochs": 200, "patience": 15}
SEEDS = [42, 43, 44]
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]


# ------------------------------------------------------------------ pairs
def plant_sequences(traits, bad):
    """plant_id -> DataFrame of its acquisitions (day order), black/placeholder photos removed."""
    t = traits[~traits.filepath.isin(bad)].sort_values(["plant_id", "day_after_planting"])
    return {pid: g.reset_index(drop=True) for pid, g in t.groupby("plant_id")}


def next_pairs(seqs, split_of, trait):
    rows = []
    for pid, g in seqs.items():
        for k in range(1, len(g)):
            if pd.isna(g.at[k, trait]):
                continue
            h = g.iloc[:k]
            rows.append({"pair_id": f"{pid}::{int(g.at[k, 'day_after_planting'])}", "plant_id": pid, "field": g.at[0, "field"], "split": split_of[pid],
                         "filepaths": h.filepath.tolist(), "days": h.day_after_planting.astype(float).tolist(),
                         "meas": [h[trait].tolist()], "target_day": float(g.at[k, "day_after_planting"]),
                         "last_day": float(h.day_after_planting.iloc[-1]), "last_value": h[trait].iloc[-1], "y": float(g.at[k, trait])})
    d = pd.DataFrame(rows)
    d["gap"] = d.target_day - d.last_day
    return d


def harvest_pairs(seqs, split_of, cutoffs=CUTOFFS):
    rows = []
    for pid, g in seqs.items():
        hd = g.harvest_day.iloc[0]
        if g.at[0, "field"] != "Field2" or pd.isna(hd):
            continue
        for c in cutoffs:
            h = g[g.day_after_planting <= c]
            if hd <= c or len(h) == 0:
                continue
            rows.append({"pair_id": f"{pid}::cut{c}", "plant_id": pid, "field": "Field2", "split": split_of[pid], "cutoff": c,
                         "filepaths": h.filepath.tolist(), "days": h.day_after_planting.astype(float).tolist(),
                         "meas": [h[m].tolist() for m in HARVEST_MEAS], "target_day": float(hd), "last_day": float(h.day_after_planting.iloc[-1]),
                         "last_head": h.head_diameter.dropna().iloc[-1] if h.head_diameter.notna().any() else np.nan,
                         "last_diam": h.diameter.dropna().iloc[-1] if h.diameter.notna().any() else np.nan, "y": float(hd - c)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ features
class SeqDataset(Dataset):
    """Per timestep: [embedding(512), photo day z, (gap to target z)] (+ per measured trait: value z, missing flag)."""

    def __init__(self, df, emb, st, use_gap, use_meas):
        self.df, self.emb, self.st, self.use_gap, self.use_meas = df.reset_index(drop=True), emb, st, use_gap, use_meas
        self.items = [self._build(r) for r in self.df.itertuples(index=False)]   # features are fixed: build once

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def _build(self, r):
        days = np.asarray(r.days, dtype=np.float32)
        cols = [torch.stack([self.emb[f] for f in r.filepaths]), torch.from_numpy((days - self.st["day_m"]) / self.st["day_s"]).unsqueeze(1)]
        if self.use_gap:
            cols.append(torch.from_numpy((r.target_day - days - self.st["gap_m"]) / self.st["gap_s"]).float().unsqueeze(1))
        if self.use_meas:
            for j, v in enumerate(r.meas):
                v = np.asarray(v, dtype=np.float32)
                miss = np.isnan(v)
                z = np.where(miss, 0.0, (v - self.st["meas_m"][j]) / self.st["meas_s"][j]).astype(np.float32)
                cols += [torch.from_numpy(z).unsqueeze(1), torch.from_numpy(miss.astype(np.float32)).unsqueeze(1)]
        x = torch.cat([c.float() for c in cols], dim=1)
        return {"features": x, "length": x.shape[0], "target_norm": torch.tensor((r.y - self.st["y_m"]) / self.st["y_s"], dtype=torch.float32),
                "target_raw": torch.tensor(r.y, dtype=torch.float32), "pair_id": r.pair_id, "plant_id": r.plant_id}


def last_features(ds):
    """Ridge design matrix: the LAST timestep's feature vector of each pair."""
    return np.stack([ds[i]["features"][-1].numpy() for i in range(len(ds))])


def load_embeddings(emb_dir, needed):
    idx = pd.read_parquet(os.path.join(emb_dir, "embedding_index.parquet")).set_index("image_id")
    out, cache = {}, {}
    for fp in needed:
        sf, off = idx.loc[fp, ["shard_file", "offset"]]
        if sf not in cache:
            cache[sf] = torch.load(os.path.join(emb_dir, sf), weights_only=True)["embeddings"]
        out[fp] = cache[sf][int(off)]
    return out


# ------------------------------------------------------------------ evaluation helpers
def conformal(val_res, a):
    r = np.sort(np.abs(val_res))
    k = min(int(np.ceil((len(r) + 1) * (1 - a))), len(r))
    return float(r[k - 1])


def boot(diff, plants, n=2000, seed=0):
    g = pd.DataFrame({"d": diff, "p": plants}).groupby("p")["d"].agg(["sum", "count"])
    i = np.random.RandomState(seed).randint(0, len(g), (n, len(g)))
    m = g["sum"].values[i].sum(1) / g["count"].values[i].sum(1)
    return np.percentile(m, [2.5, 97.5])


def fit_ridge(Xtr, ytr, Xva, yva):
    best = min(ALPHAS, key=lambda a: np.abs(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - yva).mean())
    return Ridge(alpha=best).fit(Xtr, ytr), best


def growth_table(tr, key_cols):
    g = tr.dropna(subset=["last_value"]).assign(step=lambda d: d.y - d.last_value)
    return g.groupby(key_cols)["step"].mean()


# ------------------------------------------------------------------ main
def run_task(task, d, emb, args, device, rows, paired, per_pair, grid_rows):
    harvest = task == "harvest"
    tr, va, te = (d[d.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if harvest:
        ev_mask = lambda x: np.ones(len(x), bool)
    else:
        ev_mask = lambda x: x.last_value.notna().values
    allm = np.concatenate([np.asarray(m, dtype=float) for ms in tr.meas for m in ms]) if not harvest else None
    st = {"day_m": float(np.mean(np.concatenate(tr.days.values))), "day_s": float(np.std(np.concatenate(tr.days.values))),
          "y_m": float(tr.y.mean()), "y_s": float(tr.y.std())}
    if not harvest:
        gaps = np.concatenate([r.target_day - np.asarray(r.days) for r in tr.itertuples()])
        st.update(gap_m=float(gaps.mean()), gap_s=float(gaps.std()), meas_m=[float(np.nanmean(allm))], meas_s=[float(np.nanstd(allm))])
    else:
        mm = [np.concatenate([np.asarray(r.meas[j], dtype=float) for r in tr.itertuples()]) for j in range(len(HARVEST_MEAS))]
        st.update(meas_m=[float(np.nanmean(x)) for x in mm], meas_s=[float(np.nanstd(x)) for x in mm])
    preds = {}   # (arm, method) -> (val_pred, test_pred)

    # --- baselines
    if harvest:
        prior = tr.groupby("cutoff").y.mean()
        preds[("image", "age_only")] = (va.cutoff.map(prior).values, te.cutoff.map(prior).values)
        def hx(x):
            return np.column_stack([x.last_head.fillna(tr.last_head.mean()), x.last_head.isna(), x.last_diam.fillna(tr.last_diam.mean()),
                                    pd.get_dummies(x.cutoff.astype(pd.CategoricalDtype(CUTOFFS))).values])
        mdl, _ = fit_ridge(hx(tr).astype(float), tr.y.values, hx(va).astype(float), va.y.values)
        preds[("meas", "last_measurements_linear")] = (mdl.predict(hx(va).astype(float)), mdl.predict(hx(te).astype(float)))
    else:
        age = tr.groupby(["field", "target_day"]).y.mean()
        fmean = tr.groupby("field").y.mean()
        a_of = lambda x: np.array([age.get((f, t), fmean[f]) for f, t in zip(x.field, x.target_day)])
        preds[("image", "age_only")] = (a_of(va), a_of(te))
        preds[("meas", "persistence")] = (va.last_value.values, te.last_value.values)
        gs, gd = growth_table(tr, ["field", "last_day", "target_day"]), growth_table(tr, ["field", "target_day"])
        g_of = lambda x: x.last_value.values + np.array([gs.get((f, l, t), gd.get((f, t), 0.0)) for f, l, t in zip(x.field, x.last_day, x.target_day)])
        preds[("meas", "last_plus_growth")] = (g_of(va), g_of(te))

    # --- learned methods, per arm
    for arm in ("image", "meas"):
        mk = lambda x: SeqDataset(x, emb, st, use_gap=not harvest, use_meas=(arm == "meas"))
        dtr, dva, dte = mk(tr), mk(va), mk(te)
        in_dim = dtr[0]["features"].shape[1]
        Xtr, Xva, Xte = last_features(dtr), last_features(dva), last_features(dte)
        mdl, alpha = fit_ridge(Xtr, tr.y.values, Xva, va.y.values)
        preds[(arm, "ridge")] = (mdl.predict(Xva), mdl.predict(Xte))
        ck_dir = os.path.join(args.checkpoint_dir, "trait_forecast")
        os.makedirs(ck_dir, exist_ok=True)
        best = None
        for g in GRID:
            cfg = {**BASE, **g, "input_dim": in_dim}
            if args.smoke:
                cfg["max_epochs"] = 2
            ck = os.path.join(ck_dir, f"{task}_{arm}_h{g['hidden_dim']}_L{g['num_layers']}_s42.pt")
            vmae, _, hist, _ = L.train_one_config(dtr, dva, cfg, device, ck, st["y_m"], st["y_s"], seed=42, verbose=False)
            grid_rows.append({"task": task, "arm": arm, **g, "val_mae": vmae, "epochs": len(hist)})
            print(f"[{task}/{arm}] grid {g} val MAE {vmae:.3f}", flush=True)
            if best is None or vmae < best[0]:
                best = (vmae, cfg, ck)
            if args.smoke:
                break
        vp, tp = [], []
        for seed in (SEEDS[:1] if args.smoke else SEEDS):
            cfg = best[1]
            ck = best[2] if seed == 42 else os.path.join(ck_dir, f"{task}_{arm}_best_s{seed}.pt")
            if seed != 42:
                L.train_one_config(dtr, dva, cfg, device, ck, st["y_m"], st["y_s"], seed=seed, verbose=False)
            m = CNNLSTM(input_dim=in_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
            m.load_state_dict(torch.load(ck, weights_only=True))
            for ds, out, ref in ((dva, vp, va), (dte, tp, te)):
                p = L.predict(m, DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn), device, st["y_m"], st["y_s"]).set_index("pair_id")
                out.append(p.loc[ref.pair_id, "y_pred"].values)
        preds[(arm, "cnn_lstm")] = (np.mean(vp, 0), np.mean(tp, 0))
        print(f"[{task}/{arm}] ridge alpha {alpha}; LSTM config {best[1]['hidden_dim']}x{best[1]['num_layers']}", flush=True)

    # --- scoring on the shared evaluation set
    vm, tm_ = ev_mask(va), ev_mask(te)
    E = te[tm_].reset_index(drop=True)
    for (arm, meth), (pv, pt) in preds.items():
        pv_, pt_ = np.asarray(pv, float)[vm], np.asarray(pt, float)[tm_]
        err = pt_ - E.y.values
        lo, hi = boot(np.abs(err), E.plant_id.values)
        r = {"task": task, "arm": arm, "method": meth, "n": len(E), "n_plants": E.plant_id.nunique(), "mae": np.abs(err).mean(), "mae_ci_lo": lo,
             "mae_ci_hi": hi, "rmse": float(np.sqrt((err ** 2).mean()))}
        vres = pv_ - va[vm].y.values
        for a in (0.2, 0.1):
            q = conformal(vres, a)
            r[f"cov{int(100 * (1 - a))}"] = float((np.abs(err) <= q).mean())
            r[f"width{int(100 * (1 - a))}"] = 2 * q
        for f in sorted(E.field.unique()):
            r[f"mae_{f}"] = np.abs(err[E.field.values == f]).mean()
        rows.append(r)
        per_pair.append(pd.DataFrame({"task": task, "arm": arm, "method": meth, "pair_id": E.pair_id, "plant_id": E.plant_id, "field": E.field,
                                      "horizon": E.cutoff if harvest else E.gap, "y_true": E.y, "y_pred": pt_}))
    # paired comparisons: does the learned model beat the matching baseline of its own arm?
    ref = {"image": "age_only", "meas": "last_measurements_linear" if harvest else "last_plus_growth"}
    for arm, b in ref.items():
        for meth in ("ridge", "cnn_lstm"):
            ea = np.abs(np.asarray(preds[(arm, meth)][1], float)[tm_] - E.y.values)
            eb = np.abs(np.asarray(preds[(arm, b)][1], float)[tm_] - E.y.values)
            for sub, msk in [("all", np.ones(len(E), bool))] + [(f, E.field.values == f) for f in sorted(E.field.unique())]:
                if msk.sum() < 2:
                    continue
                lo, hi = boot((ea - eb)[msk], E.plant_id.values[msk])
                paired.append({"task": task, "arm": arm, "method": meth, "baseline": b, "subset": sub, "n": int(msk.sum()),
                               "n_plants": int(E.plant_id[msk].nunique()), "diff": float((ea - eb)[msk].mean()), "ci_lo": lo, "ci_hi": hi})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", required=True)
    ap.add_argument("--image-pairs", required=True, help="only for the plant -> split map (739 plants)")
    ap.add_argument("--exclude-list", required=True)
    ap.add_argument("--embeddings-dir")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", nargs="+", default=NEXT_TRAITS + ["harvest"])
    ap.add_argument("--counts-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    traits = pd.read_parquet(args.traits)
    split_of = pd.read_parquet(args.image_pairs)[["plant_id", "split"]].drop_duplicates().set_index("plant_id")["split"].to_dict()
    assert set(traits.plant_id) == set(split_of), "split map does not cover every plant"
    bad = set(pd.read_csv(args.exclude_list, sep=r"\s+", header=None, names=["md5", "fp"])["fp"])
    seqs = plant_sequences(traits, bad)
    data = {t: next_pairs(seqs, split_of, t) for t in args.tasks if t != "harvest"}
    if "harvest" in args.tasks:
        data["harvest"] = harvest_pairs(seqs, split_of)
    for t, d in data.items():
        ev = d if t == "harvest" else d[d.last_value.notna()]
        print(f"{t:14s} pairs {d.groupby('split').size().to_dict()}  test eval set {int((ev.split == 'test').sum())} pairs / "
              f"{ev[ev.split == 'test'].plant_id.nunique()} plants  by field {ev[ev.split == 'test'].groupby('field').size().to_dict()}", flush=True)
    if args.counts_only:
        return
    device = torch.device(args.device)
    needed = sorted({f for d in data.values() for fps in d.filepaths for f in fps})
    emb = load_embeddings(args.embeddings_dir, needed)
    rows, paired, per_pair, grid = [], [], [], []
    for t, d in data.items():
        run_task(t, d, emb, args, device, rows, paired, per_pair, grid)
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "trait_forecast_results.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(args.out_dir, "trait_forecast_paired.csv"), index=False)
    pd.concat(per_pair).to_csv(os.path.join(args.out_dir, "trait_forecast_per_pair.csv"), index=False)
    pd.DataFrame(grid).to_csv(os.path.join(args.out_dir, "trait_forecast_grid.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
