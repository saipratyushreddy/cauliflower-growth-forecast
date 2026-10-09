"""
Demo UI for the image-to-image growth-forecasting track (Streamlit, local, CPU only).

    streamlit run app.py                         # expects ./demo_bundle (built on Swan by scripts/export_demo_bundle.py)
    DEMO_BUNDLE=/path/to/demo_bundle streamlit run app.py

Pick a held-out TEST plant and a cutoff (how many of its real images are fed in; every cutoff is a real, validated test pair, so the target is the
plant's next real image). Shown: the input sequence, the real target, and the frozen CNN-Transformer and ConvLSTM predictions with structure SSIM /
RGB SSIM / PSNR against the real target, optionally with the naive copy-forward prediction. All logic is in src/demo_core.py and reuses the project's code.
"""
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from demo_core import DROP_LEVELS, DemoBackend  # noqa: E402

BUNDLE = os.environ.get("DEMO_BUNDLE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_bundle"))
st.set_page_config(page_title="Cauliflower growth forecast demo", layout="wide")


@st.cache_resource(show_spinner="Loading checkpoints and images (CPU)...")
def backend():
    return DemoBackend(BUNDLE, device="cpu")


@st.cache_data(show_spinner=False, max_entries=64)
def run_state(plant_id, n_frames, drop_pct):
    return backend().run(plant_id, n_frames, drop_pct)


def greyed(img):
    g = Image.fromarray(img).convert("L").convert("RGB")
    g = Image.blend(g, Image.new("RGB", g.size, (255, 255, 255)), 0.55)
    d = ImageDraw.Draw(g)
    w, h = g.size
    d.line([(0, 0), (w, h)], fill=(200, 40, 40), width=6)
    d.line([(0, h), (w, 0)], fill=(200, 40, 40), width=6)
    return np.asarray(g)


if not os.path.isdir(BUNDLE):
    st.error(f"Demo bundle not found at `{BUNDLE}`. Build it on Swan with `python scripts/export_demo_bundle.py --out demo_bundle` and copy it here "
             "(see DEMO.md), or set DEMO_BUNDLE.")
    st.stop()

be = backend()
plants = be.plants()
label = {p: f"{p}  ({f})" for p, f in plants}

st.title("Cauliflower growth forecast: predicting a plant's next image")
with st.sidebar:
    st.header("Held-out test plant")
    plant = st.selectbox(f"Plant ({len(plants)} test plants)", [p for p, _ in plants], format_func=lambda p: label[p], index=0, key="plant")
    cuts = be.cutoffs(plant)
    ns = [n for n, _ in cuts]
    tday = dict(cuts)
    if len(ns) > 1:
        n_frames = st.select_slider("Cutoff: number of real input images fed in", options=ns, value=ns[-1],
                                    format_func=lambda n: f"{n} image{'s' if n > 1 else ''} -> predict day {tday[n]}", key="cutoff")
    else:
        n_frames = ns[0]
        st.write(f"Only one cutoff available: {n_frames} image -> predict day {tday[n_frames]}")
    drop_pct = st.select_slider("Missing-observation drop level", options=DROP_LEVELS, value=0, format_func=lambda v: f"{v}%", key="drop")
    show_cf = st.checkbox("Also show the naive copy-forward prediction", value=True, key="cf")
    st.caption("Same sidebar state -> same result: the random drop uses a fixed seed derived from (plant, target day, drop level).")

with st.spinner("Running both models on CPU..."):
    r = run_state(plant, n_frames, drop_pct)

st.subheader(f"{r['plant_id']} ({r['field']}): predicting day {r['target_day']} from {r['n_available']} real image(s)")
if drop_pct > 0:
    st.warning("**Exploratory view, not validated.** The drop levels reuse the project's missing-observation dropping function, which was validated in the "
               "earlier *diameter-regression* robustness test. The image models (Transformer, ConvLSTM) were never evaluated with dropped inputs, so nothing "
               "shown at drop > 0 is backed by a README result.")
    st.caption(f"{len(r['dropped'])} of {r['n_available']} input images dropped (a drop level rounds to a whole number of images, never below 1 kept). "
               f"Seed {r['seed']}: kept input positions {r['kept']}, dropped {r['dropped']}. Last kept image is day {r['last_kept_day']}, "
               f"{r['offset']} day(s) before the target.")

st.markdown("**Input image sequence** (day after planting; offset = days before the target). Dropped observations are greyed out and crossed.")
imgs, days, kept = r["input_images"], r["input_days"], set(r["kept"])
per_row = 7
for start in range(0, len(imgs), per_row):
    cols = st.columns(per_row)
    for j, col in enumerate(cols):
        i = start + j
        if i >= len(imgs):
            break
        cap = f"day {days[i]} (-{r['target_day'] - days[i]} d)" + ("" if i in kept else "  DROPPED")
        col.image(imgs[i] if i in kept else greyed(imgs[i]), caption=cap, use_container_width=True)

st.markdown("**Real target vs predictions** (metrics are against the real target)")
names = ["Transformer", "ConvLSTM"] + (["Copy-forward"] if show_cf else [])
cols = st.columns(1 + len(names))
cols[0].image(r["target"], caption=f"Real target (day {r['target_day']})", use_container_width=True)
titles = {"Transformer": "CNN-Transformer (full history)", "ConvLSTM": "ConvLSTM (last 4 images)", "Copy-forward": "Copy-forward (last image unchanged)"}
for col, name in zip(cols[1:], names):
    m = r["metrics"][name]
    col.image(r["pred"][name], caption=titles[name], use_container_width=True)
    col.markdown(f"structure SSIM **{m['structure_ssim']:.3f}**  \nRGB SSIM **{m['ssim']:.3f}**  \nPSNR **{m['psnr']:.2f} dB**")

with st.expander("About this demo"):
    st.markdown(
        "- **What is shown.** For a held-out test plant, the models receive the plant's real earlier images (the cutoff) and generate its next image; "
        "the real next image is shown for comparison. At drop 0% each state is exactly one of the 1,233 cleaned test pairs used in the README.\n"
        "- **Models (frozen checkpoints, CPU inference, no retraining).** CNN-Transformer over the full available history; ConvLSTM K=4 over the last "
        "4 images (seed 42); copy-forward repeats the last input image. Same preprocessing everywhere: 256x256, Field1 downsampled from 490x490.\n"
        "- **Metrics.** Structure SSIM (luma, local contrast normalization), RGB SSIM and PSNR, as defined in the README evaluation protocol.\n"
        "- **How to read the scores.** Pooled over the 1,233 test pairs (README): structure SSIM 0.044 copy-forward, 0.087 ConvLSTM K=4, 0.088 Transformer. "
        "The predictions are smooth, blurry colour fields, not sharp images, and absolute scores are low; the models' advantage over copy-forward is mainly "
        "in smooth/low-frequency agreement, with a small structure SSIM gain. Individual plants vary widely around these means.\n"
        "- **Not shown on purpose.** Nothing that was not validated in the README (for example, attention weights or any result at drop > 0).")
