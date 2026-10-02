"""
Visual check of the blur-scan clusters. Writes ONE contact-sheet PNG. For each
cluster (rows of 4):
  row 1  FLAGGED/lowest-VoL images of the cluster (4 lowest VoL in the cluster)
  row 2  same plants (1-3) at the PREVIOUS acquisition date, + 1 same-date,
         same-plot image with median VoL among the cluster's non-flagged
  row 3  same plants (1-3) at the NEXT acquisition date, + 1 same-date,
         different-plot image with median VoL
The 4th images in rows 2/3 separate "flight-level" from "plot-level" problems.
Titles give plant, date(day), variance of Laplacian, mean luma and luma std so
"dark but sharp" can be told apart from "blurry". Images use the same 256x256
resize as the rest of the image track. Purely descriptive: nothing is excluded.

Usage:
    python src/make_contact_sheet.py --scan-csv outputs/img_blur_scan_all.csv \
        --images-root $WORK/cauliflower-growth-forecast/data/images --out outputs/img_contact_sheet.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SIZE = 256
# name, field, acquisition_date, plot (None = every plot)
CLUSTERS = [
    ("Field1 day 28 (2020-08-25), all plots", "Field1", "2020-08-25", None),
    ("Field2 Plot5 2021-08-11 (day 57)", "Field2", "2021-08-11", 5),
    ("Field2 Plot1 2021-08-30 (day 76)", "Field2", "2021-08-30", 1),
    ("Field2 Plot1 2021-06-16 (day 1)", "Field2", "2021-06-16", 1),
]


def load_img(root, fp):
    im = Image.open(os.path.join(root, fp)).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im)


def median_pick(df):
    """Row whose VoL is closest to the group's median."""
    return df.iloc[(df["vol"] - df["vol"].median()).abs().argsort().iloc[0]]


def neighbor(allimg, row, step):
    g = allimg[allimg.plant_id == row.plant_id].sort_values("day_after_planting").reset_index(drop=True)
    i = int(np.where(g.filepath == row.filepath)[0][0]) + step
    return g.iloc[i] if 0 <= i < len(g) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-csv", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scan = pd.read_csv(args.scan_csv)
    scan = scan[~scan["known_degenerate"]].copy()
    scan["plot"] = scan["plot"] if "plot" in scan else scan["plant_id"].str.extract(r"Plot(\d+)")[0].astype(int)
    scan["flagged"] = scan["flagged"].fillna(False).astype(bool)

    cells = []  # (cluster_idx, row_in_block, col, row_obj or None, label)
    for ci, (name, field, date, plot) in enumerate(CLUSTERS):
        day = scan[(scan.field == field) & (scan.acquisition_date == date)]
        grp = day if plot is None else day[day["plot"] == plot]
        if grp.empty:
            print(f"cluster '{name}': no images found, skipped")
            continue
        low = grp.sort_values("vol").head(4)
        print(f"{name}: {len(grp)} images on that date/plot, {int(grp.flagged.sum())} flagged; "
              f"VoL cluster median {grp.vol.median():.0f}, field-day median {day.vol.median():.0f}")
        for c, r in enumerate(low.itertuples()):
            cells.append((ci, 0, c, r, "FLAGGED" if r.flagged else "lowest-VoL (not flagged)"))
        for c, r in enumerate(low.head(3).itertuples()):
            for rr, step, lab in [(1, -1, "prev date"), (2, +1, "next date")]:
                nb = neighbor(scan, r, step)
                cells.append((ci, rr, c, nb, lab))
        same_plot = grp[~grp.flagged & ~grp.filepath.isin(low.filepath)]
        cells.append((ci, 1, 3, median_pick(same_plot) if len(same_plot) else None, "same date+plot, median"))
        other = day[(day["plot"] != low.iloc[0]["plot"]) & ~day.flagged]
        cells.append((ci, 2, 3, median_pick(other) if len(other) else None, "same date, other plot, median"))

    n_cl = len(CLUSTERS)
    fig, axes = plt.subplots(3 * n_cl, 4, figsize=(4 * 3.0, 3 * n_cl * 3.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ci, rr, c, r, lab in cells:
        ax = axes[ci * 3 + rr][c]
        if r is None:
            ax.text(0.5, 0.5, f"{lab}\n(none)", ha="center", va="center", fontsize=7)
            continue
        fp = r.filepath
        img = load_img(args.images_root, fp)
        luma = np.asarray(Image.fromarray(img).convert("L"), dtype=float)
        ax.imshow(img)
        ax.set_title(f"[{lab}]\n{r.plant_id.replace('_Ref_', '_')}\n{r.acquisition_date} (d{int(r.day_after_planting)})\n"
                     f"VoL={r.vol:.0f} luma={luma.mean():.0f} sd={luma.std():.0f}", fontsize=6)
    for ci, (name, *_rest) in enumerate(CLUSTERS):
        axes[ci * 3][0].annotate(f"CLUSTER {ci + 1}: {name}", xy=(0, 1.55), xycoords="axes fraction",
                                 fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005, wspace=0.03, hspace=0.55)
    fig.savefig(args.out, dpi=85)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
