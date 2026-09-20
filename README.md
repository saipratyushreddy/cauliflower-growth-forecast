# Cauliflower Growth Forecasting (GrowliFlower CNN-LSTM Baseline)

Phase 1 goal: a working, evaluated CNN-LSTM baseline that predicts one
cauliflower growth trait at the next measured timepoint, from a UAV image
time series, using the [GrowliFlower dataset](https://huggingface.co/datasets/Voxel51/GrowliFlower)
(curated FiftyOne release, `Voxel51/GrowliFlower`).

Runs on UNL Holland Computing Center's Swan cluster (SLURM). Local machine
is used only for lightweight metadata inspection/pipeline construction (no
GPU, no full image downloads) — anything touching the actual images or a
GPU runs on Swan.

## Task definition

Given all images and timestamps for a plant through observation `t`,
predict its trait value at `t+1`, where **t+1 means the next chronological
date with a valid measurement of the selected trait** (not necessarily the
next UAV flight date — flight dates and measurement dates aren't the same
calendar). The input sequence for a sample is every image from planting
through the last acquisition date **strictly before** the target date,
whether or not that image's own date has a valid label. No image or label
from the target date or later is used anywhere, including in
normalization/preprocessing statistics.

## Target trait: plant diameter (`in_situ_diameter`)

Chosen after auditing label coverage across five candidate traits (see
`outputs/step2_inspection_report.txt` for the full report). Plant diameter
has by far the best coverage: 738/739 reference plants have ≥2 valid
measurements (vs. 267/739 for height, 512/739 for head diameter), it's a
genuinely continuous regression target, and it's a core growth-curve trait.

## Repo structure

```
data/            Metadata/pairs/embeddings artifacts (gitignored; regenerate via scripts below)
src/             Pipeline scripts
scripts/         SLURM job scripts
notebooks/       Exploration only, not part of the pipeline
outputs/         Reports, plots, saved figures
checkpoints/     Trained model weights
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Swan, install a CUDA-matched `torch`/`torchvision` build (check
`module avail cuda` and pick the matching PyTorch wheel) rather than relying
on the generic `pip install torch` resolution used for local CPU-only work.

## Pipeline (run in order)

### Step 2 — Inspect the dataset
```bash
python src/inspect_growliflower.py --samples-json <path to samples.json from Voxel51/GrowliFlower>
```
Downloads/parses `samples.json` (the FiftyOne export) and reports per-task
sample counts, field inventory, and trait coverage across all 5 candidate
traits. See `outputs/step2_inspection_report.txt` for the full run
(includes an addendum auditing the gap distribution between consecutive
valid diameter measurements, and all 165 non-numeric raw diameter values).

### Step 3a — Build the metadata table
```bash
python src/build_metadata.py --samples-json <path> --out data/metadata.parquet
```
One row per (plant_id, acquisition_date) for `task=='reference'` samples,
with a cleaned `diameter` column. Cleaning rule for the 165 non-numeric
`in_situ_diameter` values (2 literal `"?"`, 163 `"X (Y)"` paired format):
**use X only, discard Y** — see the DECISION LOG section appended to
`outputs/step2_inspection_report.txt` for why (no dual-reading protocol is
documented in the source paper, and `in_situ_comment` is empty on all 163
paired records, ruling out a "two plants sharing one ID" explanation).

### Step 3b — Construct forecasting pairs
```bash
python src/build_pairs.py --metadata data/metadata.parquet --out data/pairs.parquet
```
Builds (input_sequence, target) pairs per the t→t+1 definition above.
**Result: 7,031 pairs from 738 distinct plants** (1 of 739 reference plants
dropped for having fewer than 2 valid diameter measurements).

### Step 3c — Plant-wise train/val/test split
```bash
python src/split_data.py --pairs data/pairs.parquet --out-dir data/ --seed 42
```
Grouped 70/15/15 split at the **plant** level (never the pair level), fixed
seed. Asserts zero `plant_id` overlap across splits (both group-level and
row-level checks).

**Result:** train 517 plants / 4,860 pairs, val 111 plants / 1,111 pairs,
test 110 plants / 1,060 pairs.

### Step 3d — Fit normalization
```bash
python src/fit_normalization.py --pairs-split data/pairs_split.parquet --out data/norm_stats.json
```
Fits target-diameter and day-after-planting z-score stats **on training
plants only**; val/test use these fixed values unchanged. CNN input
normalization uses fixed ImageNet mean/std (not fit on this dataset, so no
leakage risk there).

### Step 4 — Download images + cache CNN embeddings (Swan, GPU)
```bash
# 1. Download the 9,377 reference-task images from HF onto $WORK
sbatch scripts/download_images.slurm

# 2. Cache frozen ResNet18 (ImageNet) embeddings, sharded by plant_id
sbatch scripts/cache_embeddings.slurm
```
Both jobs are resumable — rerun the same `sbatch` command if a job times
out or is killed; already-completed work is skipped (checked against the
existing index/shard files, not re-downloaded or recomputed).

Embeddings are cached at `data/embeddings/shards/<plant_id>.pt` (each a
dict of `{"filepaths": [...], "embeddings": FloatTensor[N, 512]}`, ordered
by `day_after_planting`), indexed by `data/embeddings/embedding_index.parquet`
(`image_id -> (shard_file, offset, plant_id, embedding_dim)`).

Validated locally on 18 real images (2 plants) before running on Swan,
including the resumability logic under a simulated partial/interrupted
shard — see conversation history / commit log for the smoke-test procedure.

## Data characteristics worth carrying into limitations

- Reference-task population is **739 plants**, not ~14,000 (that figure
  didn't match the actual `Voxel51/GrowliFlower` release; only the
  `reference` task subset carries the in-situ trait fields needed for
  regression — `defoliation`, the other plant-time-series task, has no
  trait labels).
- Acquisition cadence is regular but sparse relative to the full growing
  season: 6–15 dates per plant (median 15), of which diameter is measured
  on ~83% (7,769/9,377 records).
- Forecast gap between consecutive valid diameter measurements: median 7
  days, p90 9 days, max 42 days. ~27% of plants (203/737) have one moderate
  outlier jump (typically ~30 days, likely a mid-season monitoring gap);
  none exceed 60 days. Forecast horizon is therefore not constant across
  samples.
- Input sequence length across all 7,031 pairs: min 1, median 5, max 14
  frames.

## Results (Step 5 — baselines)

Evaluated on held-out test plants (110 plants, plant-wise split, seed=42).
Each method's own full-N test result is reported first (its honest,
independently-eligible sample set); N differs slightly between methods
because persistence additionally requires the last input date to itself
have a valid label, a stricter filter than single-frame's (which only
needs the last input image to exist, always true here given Step 4's
0-failure embedding run).

| Baseline | Test N (pairs / plants) | MAE | RMSE |
|---|---|---|---|
| Persistence (ŷ_{t+1} = y_t) | 1,057 / 110 | 9.068 | 10.914 |
| Single-frame (ridge on frozen ResNet18 embedding, α=100, selected on val) | 1,060 / 110 | 5.962 | 8.736 |

**Shared-evaluation-set convention:** since methods can have different
eligible test sets, head-to-head comparisons are computed on the
*intersection* of pair_ids (`plant_id::target_day`, asserted unique) every
compared method can predict, not on each method's own full-N set. On the
shared 1,057-pair intersection: persistence MAE=9.068/RMSE=10.914,
single-frame MAE=5.914/RMSE=8.657 — **single-frame improves MAE by 34.8%
and RMSE by 20.7%** (close to the naive full-N deltas, confirming the 3
single-frame-only pairs weren't meaningfully skewing the comparison, but
this is now the correct number to cite and the convention Step 6 will
also follow). Full per-pair predictions/residuals are dumped to
`outputs/step5_<method>_test_predictions.csv`.

**Why MAE improved more (34.8%) than RMSE (20.7%) for single-frame —
residual analysis:** correlation(true diameter, residual) = **-0.355** on
single-frame's test predictions — a moderate regression-to-the-mean
effect: the model over-predicts small plants and under-predicts large
ones (mean residual +2.6 in the bottom quartile of true diameter, -5.7 in
the top quartile). MAE actually rises monotonically with true diameter
(bottom quartile 4.74 → mid 6.15 → top quartile 6.79), and the 32 worst
misses (>20mm absolute error) are concentrated in just a few plants: 9 of
32 come from a single plant (`2021_Ref_Plot2_A1`), 7 from another
(`2021_Ref_Plot2_E18`) — over half the worst misses from 2 of 110 test
plants. This is a concrete, testable hypothesis for Step 6: a temporal
sequence should let the LSTM recognize "this plant has consistently been
small/large across prior frames" rather than guessing from one ambiguous
image, which should specifically reduce the top/bottom-quartile bias —
worth re-checking this same quartile breakdown on the LSTM's results.

```bash
python src/baselines.py \
    --pairs-split data/pairs_split.parquet \
    --metadata data/metadata.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --out-dir outputs
```

## Results (Step 6 — CNN-LSTM)

Unidirectional (causal) LSTM over the ordered cached ResNet18 embeddings +
normalized day-after-planting per timestep, trained with early stopping on
val MAE (raw units). All runs on an NVIDIA A30 GPU on Swan, clean `.err`
logs (no warnings).

**First, a single manually-chosen config** (hidden_dim=128, 1 layer,
lr=1e-3) was trained as an initial check (stopped at epoch 29, best val
MAE=5.138). **This was later superseded by a proper 18-config
hyperparameter sweep** (`src/sweep_cnn_lstm.py`) — the manual run is kept
below only as a documented comparison point, not the final result.

### Hyperparameter sweep (the actual Step 6 result)

Grid: `hidden_dim ∈ {64,128,256} × lr ∈ {1e-3,3e-4} × num_layers ∈ {1,2}`,
with `dropout ∈ {0.0,0.2}` swept only for `num_layers=2` (dropout is a
no-op on a 1-layer `nn.LSTM`) — 18 configs total, targeting overfitting
risk from deeper models given only 517 training plants.

**Model-selection discipline:** every config is compared by **validation
MAE only**; test is evaluated **exactly once**, for the single config with
the best val MAE, after the full grid finishes (verified in the run log:
`grep -c "TEST EVALUATION"` = 1, not 18). This mirrors the same principle
already used for the ridge alpha sweep in Step 5's baseline — the test
split is never used for model selection, only for final reporting.

```bash
python src/sweep_cnn_lstm.py \
    --pairs-split data/pairs_split.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --baseline-results outputs/step5_baseline_results.json \
    --out-dir outputs \
    --checkpoint-dir checkpoints \
    --device cuda --batch-size 64 --max-epochs 200 --patience 15 --seed 42
```

**Winning config: `h256_lr0.001_L2_d0.2`** (hidden_dim=256, lr=1e-3,
2 layers, dropout=0.2), val MAE=5.007 — full 18-config table in
`outputs/step6_sweep_results.csv`.

The full grid spans val MAE **5.007–5.448**, under 9% spread top-to-bottom
— a fairly flat hyperparameter landscape. The original manual config
(val MAE=5.138) landed 4th of 18, meaning the manual choice was already
reasonable; the sweep found a modest, not dramatic, improvement.

**Shared-evaluation-set comparison** (same convention as Step 5 — all
methods restricted to the 1,057 test pairs every one of them can predict):

| Method | MAE | RMSE |
|---|---|---|
| Persistence | 9.068 | 10.914 |
| Single-frame | 5.914 | 8.657 |
| CNN-LSTM (manual config) | 5.373 | 8.350 |
| **CNN-LSTM (sweep winner)** | **5.134** | **8.055** |

Sweep winner improves MAE by **13.2%** and RMSE by **7.0%** over
single-frame, and **43.4%** MAE over persistence.

**Size-quartile bias — a consistent trend across three modeling stages:**
correlation(true diameter, residual) improved monotonically:
**-0.355** (single-frame) → **-0.265** (manual CNN-LSTM) →
**-0.215** (sweep-tuned CNN-LSTM). Architecture capacity and light
regularization measurably chip away at the regression-to-the-mean bias
at each stage, though it is not eliminated (top-quartile mean residual
is still -2.95mm for the tuned model, down from -5.66mm for single-frame).

### Structural limitation: late-season non-monotonic segments

A spot-check of growth-curve plots (`outputs/step6_growth_curve_<plant_id>.png`,
6 test plants) found one case (`2020_Ref_Plot1_A93`) where the model
over-predicted through a real late-season plateau/decline in the actual
measurements (54→48→44mm). A full quantitative follow-up
(`src/analyze_monotonicity.py`) confirmed this is a systematic pattern,
not a one-off:

- **Threshold used (exact, no noise band):** a pair is "flat-or-decline"
  when `target_diameter <= y_t` (delta ≤ 0mm, where y_t is the diameter at
  the pair's own last input date); "strictly declining" is `target_diameter
  < y_t`.
- **17.0%** of test pairs (180/1,057) are flat-or-decline; **79.1%** of
  test plants (87/110) have at least one such segment.
- **Magnitude — mostly real, not measurement jitter:** only 12.8% of
  flat-or-decline pairs are exact ties and 25.6% fall in a jitter-plausible
  (-2, 0)mm band; **47.8% are ≥5mm declines** (median delta -4mm, mean
  -6.5mm, min -44mm).
- **Strongly concentrated late-season, not scattered evenly** — this was
  checked explicitly since the two explanations imply very different
  limitations. Using each pair's normalized position within its own
  plant's trajectory (0=earliest measurement, 1=latest): flat-or-decline
  segments have median position **0.81** vs **0.41** for growth segments.
  By season quartile, only 2.5–10% of early/mid-season pairs are
  flat-or-decline vs **41.3%** in the last quartile. Across the full
  738-plant population (not just test), **83.8%** of all declines fall in
  the late half of a plant's own trajectory.
- **CNN-LSTM error concentrates specifically on these segments:** MAE is
  4.243 on growth segments vs **9.477** on flat-or-decline segments
  (**2.23x worse**), and 10.372 on strictly-declining segments. Overall
  test MAE also rises steadily by season quartile (2.61→4.26→5.93→7.62),
  so late season is harder in general, but declines within it are
  disproportionately harder still.

**Honest framing for limitations:** this is a real, late-season biological
transition (consistent with senescence, head-formation dynamics, or
harvest-related handling affecting the visible canopy) that the model saw
comparatively little training signal for — not generic measurement noise
scattered through the season. The model is trained predominantly on
monotonic growth segments (83% of pairs) and measurably underperforms on
the non-monotonic, late-season minority.

## Status

Steps 1–6 complete and verified on Swan:
- All pipeline artifacts (`data/metadata.parquet`, `data/pairs.parquet`,
  `data/pairs_split.parquet`, `data/norm_stats.json`) regenerated on Swan
  and confirmed to exactly match local runs (9,377 reference rows / 739
  plants, 7,031 pairs / 738 plants, 517/111/110 plant-wise split).
- Step 4 full run completed on an NVIDIA A30 GPU (`gpu` partition):
  9,377/9,377 images downloaded (0 failures) and embedded (0 failures),
  739/739 plants have a shard file, index verified structurally correct.
- Step 5 baselines run on Swan (CPU, login node — no GPU needed for ridge
  regression on 512-dim vectors); results above.
- Step 6 CNN-LSTM trained and evaluated on Swan (NVIDIA A30 GPU); results
  above, including a shared-eval-set comparison against both Step 5
  baselines and a re-check of Step 5's residual-bias hypothesis.

**Not yet run:** Step 7 (final SLURM job wrap-up — the CNN-LSTM SLURM
script already exists at `scripts/train_cnn_lstm.slurm` and was used for
the run above; Step 7 is mainly about consolidating this README's
scattered results/limitations into a final summary). Do not start the
Transformer model, multi-trait regression, or missing-data robustness
experiments — out of scope for this phase.
