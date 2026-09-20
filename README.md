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

## Status

Steps 1–4 complete (Step 4 written and locally smoke-tested; full run
pending on Swan — see command blocks above). **Not yet run:** Step 5
(persistence + single-frame baselines), Step 6 (CNN-LSTM), Step 7 (final
SLURM job + results in this README). Do not start the Transformer model,
multi-trait regression, or missing-data robustness experiments — out of
scope for this phase.
