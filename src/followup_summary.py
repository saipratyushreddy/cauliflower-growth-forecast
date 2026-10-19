"""
Collect the review follow-up experiments into three tables (prints Markdown; use after the Swan jobs finish and outputs/ is copied back):
  1. diameter clean-input rerun   (outputs/diameter_clean_rerun_results.csv)   mean +/- SD over seeds, original vs cleaned inputs
  2. robustness, latest photo kept (outputs/robustness_older_only_summary.csv) next to the original protocol (outputs/step_p2_robustness_summary.csv)
  3. sharpness follow-ups         (outputs/img_sharp_fu_*_per_pair.csv)        structure SSIM, sharpness ratio, VGG distance, selected epoch
Usage:  python src/followup_summary.py [--out-dir outputs]
"""
import argparse
import json
import os

import pandas as pd


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    o = ap.parse_args().out_dir
    f = os.path.join(o, "diameter_clean_rerun_results.csv")
    if os.path.exists(f):
        r = pd.read_csv(f)
        g = r.groupby(["model", "variant"]).agg(n_seeds=("seed", "count"), val_mae=("val_mae", "mean"), test_mae=("test_mae", "mean"), test_mae_sd=("test_mae", "std"),
                                                 test_rmse=("test_rmse", "mean"), affected_mae=("affected_mae", "mean")).round(3).reset_index()
        print("### 1. Diameter: original vs black-frame-cleaned inputs (fixed winner configs, test MAE, label units)\n\n" + md(g) + "\n")
    a, b = os.path.join(o, "step_p2_robustness_summary.csv"), os.path.join(o, "robustness_older_only_summary.csv")
    if os.path.exists(a) and os.path.exists(b):
        x, y = pd.read_csv(a), pd.read_csv(b)
        t = x[["drop_frac", "lstm_mae_mean", "transformer_mae_mean"]].merge(y[["drop_frac", "lstm_mae_mean", "transformer_mae_mean"]], on="drop_frac", suffixes=("_any", "_older_only")).round(3)
        print("### 2. Robustness MAE: any photo may be dropped (original) vs latest photo always kept (N = 1,060)\n\n" + md(t) + "\n")
    rows = []
    for name in ("fu_base", "fu_perc1", "fu_perc01", "fu_perc003", "fu_anygap_keep"):
        pp, rj = os.path.join(o, f"img_sharp_{name}_per_pair.csv"), os.path.join(o, f"img_sharp_{name}_run.json")
        if not (os.path.exists(pp) and os.path.exists(rj)):
            continue
        d, run = pd.read_csv(pp), json.load(open(rj))
        rows.append({"run": name, "best_epoch": run["best_epoch"], "structure_ssim": round(d.structure_ssim.mean(), 4), "rgb_ssim": round(d.ssim.mean(), 4),
                     "grad_ratio": round(d.grad_pred.mean() / d.grad_target.mean(), 3), "lap_ratio": round(d.lap_pred.mean() / d.lap_target.mean(), 3),
                     "vgg_dist": round(d.vgg_dist.mean(), 3)})
    if rows:
        print("### 3. Sharpness follow-ups (single seed; longer budget, selection on val structure SSIM; test pairs = 1,233)\n\n" + md(pd.DataFrame(rows)) + "\n")


if __name__ == "__main__":
    main()
