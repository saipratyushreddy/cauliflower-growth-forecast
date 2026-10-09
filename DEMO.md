# Demo UI (Streamlit, laptop, CPU only)

Pick a held-out **test** plant, a **cutoff** (how many of its real images are fed in, up to its available history) and a **drop level**. The app shows the
input image sequence (day offsets; dropped images greyed and crossed), the real next image, and the frozen CNN-Transformer and ConvLSTM predictions with
structure SSIM / RGB SSIM / PSNR against the real target, optionally with the naive copy-forward prediction.

```bash
pip install -r requirements-demo.txt
streamlit run app.py                       # uses ./demo_bundle ; or:  DEMO_BUNDLE=/path/to/demo_bundle streamlit run app.py
```

## What to copy off Swan (everything else stays on the cluster)

Run **on Swan, in the original project directory** (the one whose checkpoints produced the README numbers; not the `cauliflower-repro` re-run):

```bash
cd $WORK/cauliflower-growth-forecast && source .venv/bin/activate
python scripts/export_demo_bundle.py --out demo_bundle      # reads only; ~350 MB; verifies the image subset byte-for-byte
```

then on the laptop, inside the repository folder:

```bash
rsync -av swan:/lustre/work/cseguo/skasara2/cauliflower-growth-forecast/demo_bundle/ demo_bundle/
```

`demo_bundle/` contains exactly (and the app needs nothing else from Swan):

| File | What | Source on Swan |
|---|---|---|
| `image_pairs_test.parquet` | the 1,233 cleaned **test** pairs (110 plants), the cleaned image-pair metadata | `data/image_pairs.parquet`, test rows |
| `cache/images256.npy`, `cache/images256_index.parquet` | only the 256x256 uint8 images those pairs use (inputs incl. history, targets) | subset of `data/images256.npy` (checked byte-for-byte) |
| `checkpoints/img_transformer_best.pt` | frozen full-history CNN-Transformer (README numbers) | `checkpoints/img_transformer_best.pt` |
| `checkpoints/img_convlstm_k4_best.pt` | frozen ConvLSTM K=4, seed 42 (README numbers) | `checkpoints/img_convlstm_k4_best.pt` |
| `MANIFEST.json` | sha256 of each file, counts, code commit | generated |

Not needed: the 9,377 raw jpgs, `data/images256.npy` in full (1.8 GB), train/validation images, Phase 0 files, embeddings, per-pair CSVs.
The repository code (`app.py`, `src/`) comes from git as usual.

## What is and is not backed by the README

* **Backed:** at drop 0% each (plant, cutoff) is exactly one of the 1,233 test pairs of the README, scored with the README's metrics. The ConvLSTM here is the
  K=4 model (it uses the last 4 images), the history-using ConvLSTM of the README.
* **Not backed (shown with an on-screen warning):** drop levels above 0%. They reuse `drop_indices` from `src/robustness_missing_obs.py` unchanged, which was
  validated only in the earlier diameter-regression robustness test. The image models were never evaluated with dropped inputs.
* The predictions are smooth, blurry colour fields and the absolute scores are low (pooled test structure SSIM 0.044 copy-forward, 0.087 ConvLSTM, 0.088 Transformer).

## Tests

`python scripts/demo_selftest.py --bundle demo_bundle --ui --reference Transformer=outputs/img_transformer_per_pair.csv ConvLSTM=outputs/img_convlstm_k4_per_pair.csv Copy-forward=outputs/img_stepB_copy_forward_per_pair.csv`
checks determinism, drop invariants, finite metrics, that the drop-0 scores equal the README's per-pair scores (CPU vs GPU float tolerance), and renders `app.py`
headlessly for several plants. `scripts/demo_dev_bundle.py` fabricates a SYNTHETIC bundle (random weights, noise images) for developing without the real files; nothing
from it means anything.
