"""
Harvest lead time: how early can the harvest date be forecast? Summarises the leave-one-plot-out harvest run (outputs/<prefix>_per_pair.csv) by cutoff day.

Per cutoff: pairs, plants, mean days to harvest; test MAE (days) of every method; image-arm LSTM minus age-only (paired, plant-bootstrap 95% interval) and the
number of test plots where the LSTM is better; share of plants whose forecast is within 4 days; coverage and width of the 80% range, calibrated per cutoff on
the calibration plot (pooled-calibration coverage in brackets).

Usage:  python src/harvest_leadtime_summary.py --per-pair outputs/harvest_leadtime_per_pair.csv --out outputs/harvest_leadtime_by_cutoff.csv
"""
import argparse

import numpy as np
import pandas as pd

from trait_forecast import boot

METHODS = ["image/age_only", "image/ridge", "image/cnn_lstm", "meas/last_measurements_linear", "meas/ridge", "meas/cnn_lstm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.per_pair)
    d = d[(d.task == "harvest") & (d.role == "test")].rename(columns={"plot": "plt"})
    d["k"] = d.arm + "/" + d.method
    d["ae"] = (d.y_pred - d.y_true).abs()
    rows = []
    for c, g in d.groupby("horizon"):
        w = g.pivot_table(index=["pair_id", "plant_id", "plt"], columns="k", values="ae").reset_index()
        r = {"cutoff": int(c), "n": len(w), "n_plants": w.plant_id.nunique(), "mean_days_to_harvest": g[g.k == "image/age_only"].y_true.mean()}
        for m in METHODS:
            r[f"mae_{m}"] = w[m].mean()
        x = (w["image/cnn_lstm"] - w["image/age_only"]).values
        r["lstm_minus_age"], (r["ci_lo"], r["ci_hi"]) = x.mean(), boot(x, w.plant_id.values)
        r["plots_lstm_better"] = f"{int((w.assign(x=x).groupby('plt').x.mean() < 0).sum())}/{w.plt.nunique()}"
        l = g[g.k == "image/cnn_lstm"]
        r["lstm_within_4_days"] = (l.ae <= 4).mean()
        r["lstm_cov80_per_cutoff"], r["lstm_width80_per_cutoff"] = (l.ae <= l.q80_h).mean(), (2 * l.q80_h).mean()
        r["lstm_cov80_pooled"] = (l.ae <= l.q80).mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
