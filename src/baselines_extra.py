"""
Stronger / fairer baselines for the diameter track (added after a review pointed out that persistence is a weak reference). All fitted on TRAIN plants only, settings chosen on
VALIDATION only, test scored once, on the same shared test set as every earlier comparison (test pairs whose last input date has a valid label: 1,057 of 1,060).

  persistence            y_hat = last measured diameter                                         (uses the last MEASUREMENT)
  age_only               mean TRAIN target diameter per (field, target_day)                     (uses only the plant's age at the target date: no images, no measurements)
  persistence_plus_growth  last measured diameter + mean TRAIN growth for that (field, last_day -> target_day) step; unseen steps fall back to the
                         (field, target_day) mean growth, then to 0   (uses the last MEASUREMENT and the calendar, no images)
  ridge_image_lastday    ridge on [last-frame ResNet18 embedding, last input day]               (information of the CNN-LSTM, one frame; needs --embeddings-dir)
  ridge_image_both_days  ridge on [embedding, last input day, target day]                       (also gets the target day, which the CNN-LSTM does NOT)

Note: the CNN-LSTM sees image features and the days of its input photos only. It receives no earlier diameter MEASUREMENT and no target day, so methods that use the last
measurement (persistence, persistence_plus_growth) or the target day (age_only, ridge_image_both_days) are not given the same information; rows say which. Labels are in the
dataset's own units (the unit, mm or cm, is not stated in the dataset card or the paper; see the README).

Usage:
    python src/baselines_extra.py --pairs-split data/pairs_split.parquet --metadata data/metadata.parquet --out-dir outputs [--embeddings-dir data/embeddings]
"""
import argparse
import os

import numpy as np
import pandas as pd


def mae(y, h):
    return float(np.mean(np.abs(y - h)))


def rmse(y, h):
    return float(np.sqrt(np.mean((y - h) ** 2)))


def boot_mae(y, h, plants, n=2000, seed=0):
    d = pd.DataFrame({"e": np.abs(y - h), "p": plants}).groupby("p")["e"].agg(["sum", "count"])
    rng = np.random.RandomState(seed)
    i = rng.randint(0, len(d), (n, len(d)))
    m = d["sum"].values[i].sum(1) / d["count"].values[i].sum(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embeddings-dir", default=None)
    a = ap.parse_args()
    p = pd.read_parquet(a.pairs_split)
    m = pd.read_parquet(a.metadata)
    lab = m.set_index(["plant_id", "day_after_planting"]).diameter
    p["y_last"] = [lab.get((r.plant_id, int(r.last_input_day)), np.nan) for r in p.itertuples()]
    p["pair_id"] = p.plant_id + "::" + p.target_day.astype(str)
    tr, va, te = (p[p.split == s] for s in ("train", "val", "test"))
    sh = te[te.y_last.notna()].copy()
    vs = va[va.y_last.notna()].copy()
    print(f"shared test set: {len(sh)} of {len(te)} test pairs (last input date has a valid label); val {len(vs)}")
    preds = {}

    def add(name, f):
        preds[name] = (f(vs), f(sh))

    add("persistence", lambda d: d.y_last.values)
    g = tr.groupby(["field", "target_day"]).target_diameter.mean()

    def near(table, field, day):
        t = table.loc[field]
        return t.iloc[np.abs(t.index.to_numpy() - day).argmin()]
    add("age_only", lambda d: np.array([g[(r.field, r.target_day)] if (r.field, r.target_day) in g.index else near(g, r.field, r.target_day) for r in d.itertuples()]))
    trl = tr[tr.y_last.notna()].assign(d=lambda x: x.target_diameter - x.y_last)
    g_step = trl.groupby(["field", "last_input_day", "target_day"]).d.mean()
    g_tgt = trl.groupby(["field", "target_day"]).d.mean()
    n_fb = {"step": 0, "target_day": 0, "zero": 0}

    def pg(d):
        o = []
        for r in d.itertuples():
            k = (r.field, r.last_input_day, r.target_day)
            if k in g_step.index:
                o.append(r.y_last + g_step[k])
            elif (r.field, r.target_day) in g_tgt.index:
                o.append(r.y_last + g_tgt[(r.field, r.target_day)]); n_fb["target_day"] += 1
            else:
                o.append(r.y_last); n_fb["zero"] += 1
        return np.array(o)
    add("persistence_plus_growth", pg)

    if a.embeddings_dir:
        import torch
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        idx = pd.read_parquet(os.path.join(a.embeddings_dir, "embedding_index.parquet")).set_index("image_id")[["shard_file", "offset"]]
        cache = {}

        def emb(d):
            out = []
            for r in d.itertuples():
                fp = list(r.input_filepaths)[-1]
                sf, off = idx.loc[fp, ["shard_file", "offset"]]
                if sf not in cache:
                    cache[sf] = torch.load(os.path.join(a.embeddings_dir, sf))
                out.append(cache[sf]["embeddings"][off].numpy())
            return np.stack(out)
        E = {k: emb(d) for k, d in (("tr", tr), ("va", vs), ("te", sh))}
        for name, cols in (("ridge_image_lastday", ["last_input_day"]), ("ridge_image_both_days", ["last_input_day", "target_day"])):
            X = {k: np.hstack([E[k], d[cols].values.astype(float)]) for k, d in (("tr", tr), ("va", vs), ("te", sh))}
            sc = StandardScaler().fit(X["tr"])
            best = None
            for alpha in (1, 10, 100, 1000, 10000):
                rg = Ridge(alpha=alpha).fit(sc.transform(X["tr"]), tr.target_diameter.values)
                v = mae(vs.target_diameter.values, rg.predict(sc.transform(X["va"])))
                if best is None or v < best[0]:
                    best = (v, alpha, rg)
            print(f"{name}: alpha {best[1]} chosen on validation (val MAE {best[0]:.3f})")
            preds[name] = (best[2].predict(sc.transform(X["va"])), best[2].predict(sc.transform(X["te"])))

    y_v, y_t = vs.target_diameter.values, sh.target_diameter.values
    info = {"persistence": "last measurement", "age_only": "target day only", "persistence_plus_growth": "last measurement + calendar",
            "ridge_image_lastday": "image + last input day", "ridge_image_both_days": "image + input day + target day"}
    print(f"\npersistence_plus_growth fallbacks used on test pairs: {n_fb}")
    print("\n| Method | Uses | Val MAE | Test MAE [95% plant-bootstrap CI] | Test RMSE | Field1 MAE | Field2 MAE |\n|---|---|---|---|---|---|---|")
    rows = []
    for n, (pv, pt) in preds.items():
        lo, hi = boot_mae(y_t, pt, sh.plant_id.values)
        f1, f2 = (sh.field == "Field1").values, (sh.field == "Field2").values
        print(f"| {n} | {info[n]} | {mae(y_v, pv):.3f} | {mae(y_t, pt):.3f} [{lo:.3f}, {hi:.3f}] | {rmse(y_t, pt):.3f} | {mae(y_t[f1], pt[f1]):.3f} | {mae(y_t[f2], pt[f2]):.3f} |")
        rows.append(pd.DataFrame({"pair_id": sh.pair_id.values, "method": n, "pred": pt, "y": y_t}))
    pd.concat(rows).to_csv(os.path.join(a.out_dir, "step_extra_baselines_test_predictions.csv"), index=False)
    print("\nReference (from the README, same shared test set): single-frame ridge (image only) 5.914, CNN-Transformer 5.528, CNN-LSTM sweep winner 5.134")


if __name__ == "__main__":
    main()
