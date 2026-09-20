# Cauliflower Growth Forecasting (GrowliFlower CNN-LSTM Baseline)

**Phase 1 of a two-phase project.** The full project goal, as scoped by
the course/research prompt, is to compare **three** approaches for
forecasting cauliflower growth traits from UAV image time series:
(1) simple baselines, (2) a CNN-LSTM, and (3) a CNN-Transformer. This
repository implements and rigorously evaluates the first two — baselines
and CNN-LSTM — against real data on UNL's Swan HPC cluster. **The
CNN-Transformer, multi-trait regression, and a missing-observation
robustness test are Phase 2, deliberately not started here** (see
[Phase 2 — planned, not started](#phase-2--planned-not-started) below).
This is a checkpoint, not the finished project.

## Dataset

[GrowliFlower](https://huggingface.co/datasets/Voxel51/GrowliFlower)
(curated FiftyOne release, `Voxel51/GrowliFlower`), consolidating the
`GrowliFlowerL`/`R`/`D` subsets from the original release:

> Kierdorf, J., Junker-Frohn, L. V., Delaney, M., Olave, M. D., Burkart,
> A., Jaenicke, H., Muller, O., Rascher, U., & Roscher, R. (2022).
> GrowliFlower: An image time series dataset for GROWth analysis of
> cauLIFLOWER. *arXiv preprint arXiv:2204.00294*.

**License caveat (unresolved):** no license is stated in the PhenoRoam
catalog record, the shipped dataset card, or the source paper. This
should be confirmed with the corresponding author (Jana Kierdorf,
`jkierdorf@uni-bonn.de`) before any wider sharing, publication, or reuse
beyond this internal research checkpoint.

Runs on UNL Holland Computing Center's Swan cluster (SLURM). Local
machine is used only for lightweight metadata inspection/pipeline
construction (no GPU, no full image downloads) — anything touching the
actual images or a GPU runs on Swan.

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
outputs/         Reports, plots, saved figures (gitignored; regenerate via scripts below)
checkpoints/     Trained model weights (gitignored; regenerate via scripts below)
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Swan, `pip install torch` pulls a CUDA-enabled build automatically
(bundled `nvidia-cu12*` runtime packages — confirmed via
`torch.__version__` showing `+cu128`); no separate `module load cuda` is
needed on this cluster.

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

Validated locally on real images before running on Swan, including the
resumability logic under a simulated partial/interrupted shard.

### Step 5 — Persistence and single-frame baselines
```bash
python src/baselines.py \
    --pairs-split data/pairs_split.parquet \
    --metadata data/metadata.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --out-dir outputs
```
(a) Persistence: ŷ_{t+1} = y_t, only for pairs where the last input date
itself has a valid label. (b) Single-frame: ridge regression on the last
cached ResNet18 embedding only, alpha selected on val. Introduces the
`pair_id` (`plant_id::target_day`) and **shared-evaluation-set**
convention used for every cross-method comparison from here on (see
Results below).

### Step 6 — CNN-LSTM (single config, then hyperparameter sweep)
```bash
# Single manually-chosen config (kept only as a documented comparison point)
python src/train_cnn_lstm.py \
    --pairs-split data/pairs_split.parquet --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json --baseline-results outputs/step5_baseline_results.json \
    --out-dir outputs --checkpoint-dir checkpoints \
    --device cuda --hidden-dim 128 --num-layers 1 --batch-size 64 --lr 1e-3 \
    --max-epochs 200 --patience 15 --seed 42

# 18-config hyperparameter sweep (the reported Step 6 result) -- SLURM job:
sbatch scripts/sweep_cnn_lstm.slurm
```
Unidirectional (causal) LSTM over ordered cached embeddings + normalized
day-after-planting, early-stopped on val MAE. The sweep selects its
winning config by **validation MAE only** and touches test **exactly
once**, for that single winner — see Results below and
`src/sweep_cnn_lstm.py`'s docstring for the full model-selection
discipline and why it matters.

### Monotonicity/structural-limitation analysis
```bash
python src/analyze_monotonicity.py \
    --pairs-split data/pairs_split.parquet --metadata data/metadata.parquet \
    --cnn-lstm-predictions outputs/step6_cnn_lstm_sweep_winner_test_predictions.csv
```
Quantifies the late-season non-monotonic structural limitation described
below.

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

## Results

All held-out evaluation is on **110 test plants** (plant-wise 70/15/15
split, seed=42, zero plant overlap across splits — see Step 3c). Because
different methods have slightly different eligible sample sets (e.g.
persistence additionally needs the last input date to itself carry a
valid label), **every cross-method comparison below uses the
shared-evaluation-set convention**: metrics are recomputed on the
*intersection* of `pair_id`s (`plant_id::target_day`, asserted unique)
every compared method can predict — currently **1,057 of 1,060** test
pairs. Each method's own full-N result is also reported where relevant.

| Method | Shared-set N | MAE | RMSE |
|---|---|---|---|
| Persistence (ŷ_{t+1} = y_t) | 1,057 | 9.068 | 10.914 |
| Single-frame (ridge on frozen ResNet18 embedding, α=100, selected on val) | 1,057 | 5.914 | 8.657 |
| CNN-LSTM (single manual config: hidden=128, 1 layer, lr=1e-3) | 1,057 | 5.373 | 8.350 |
| **CNN-LSTM (18-config sweep winner: hidden=256, 2 layers, lr=1e-3, dropout=0.2)** | **1,057** | **5.134** | **8.055** |

Sweep-winner CNN-LSTM improves MAE by **13.2%** and RMSE by **7.0%** over
single-frame, and **43.4%** MAE over persistence — temporal modeling adds
real, measured value on top of a purely non-temporal visual signal.

Own full-N test results (each method's independently-eligible set, not
used for the comparisons above): persistence N=1,057/110 plants,
single-frame N=1,060/110 plants, CNN-LSTM N=1,060/110 plants.

Full commands for each result:
```bash
python src/baselines.py --pairs-split data/pairs_split.parquet --metadata data/metadata.parquet \
    --embeddings-dir data/embeddings --norm-stats data/norm_stats.json --out-dir outputs

python src/sweep_cnn_lstm.py --pairs-split data/pairs_split.parquet --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json --baseline-results outputs/step5_baseline_results.json \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda \
    --batch-size 64 --max-epochs 200 --patience 15 --seed 42
```

### Hyperparameter sweep detail

Grid: `hidden_dim ∈ {64,128,256} × lr ∈ {1e-3,3e-4} × num_layers ∈ {1,2}`,
with `dropout ∈ {0.0,0.2}` swept only for `num_layers=2` (dropout is a
no-op on a 1-layer `nn.LSTM`) — 18 configs total, targeting overfitting
risk from deeper models given only 517 training plants.

**Model-selection discipline:** every config is compared by **validation
MAE only**; test is evaluated **exactly once**, for the single config with
the best val MAE, after the full grid finishes (verified in the run log:
`grep -c "TEST EVALUATION"` = 1, not 18). This mirrors the same principle
already used for the ridge alpha sweep in Step 5 — the test split is
never used for model selection, only for final reporting.

The full grid spans val MAE **5.007–5.448**, under 9% spread top-to-bottom
— a fairly flat hyperparameter landscape. The manually-chosen config
(val MAE=5.138) landed 4th of 18, meaning the manual starting point was
already reasonable; the sweep found a modest, not dramatic, improvement.
Full 18-config table: `outputs/step6_sweep_results.csv`.

### Residual bias: a consistent trend across three modeling stages

correlation(true diameter, residual) improved monotonically across every
stage of modeling sophistication:

**-0.355** (single-frame) → **-0.265** (manual CNN-LSTM) →
**-0.215** (sweep-tuned CNN-LSTM)

More temporal context and light regularization measurably chip away at a
regression-to-the-mean bias (over-predicting small plants, under-predicting
large ones) at each stage, though it is not eliminated (top-quartile mean
residual is still -2.95mm for the tuned model, down from -5.66mm for
single-frame). Bottom-quartile MAE improved the most across stages.

### Growth curve plots

6 test plants plotted (predicted vs. actual diameter over
day-after-planting), both for the manual config
(`outputs/step6_growth_curve_<plant_id>.png`) and the sweep winner
(`outputs/step6_sweep_growth_curve_<plant_id>.png`). Two are called out
explicitly, deliberately including the unflattering one:

- `2020_Ref_Plot1_A10` — predicted tracks actual closely across the full
  trajectory (good fit).
- `2020_Ref_Plot1_A93` — the model over-predicts through a real
  late-season plateau/decline in the actual measurements (54→48→44mm) —
  a concrete instance of the structural limitation quantified below, kept
  here rather than cherry-picked out.

![Growth curve: good fit example](outputs/step6_growth_curve_2020_Ref_Plot1_A10.png)
![Growth curve: late-season miss example](outputs/step6_growth_curve_2020_Ref_Plot1_A93.png)

## Limitations

- **Late-season non-monotonic structural bias (quantified, not
  anecdotal).** A pair is "flat-or-decline" when `target_diameter <= y_t`
  (delta ≤ 0mm vs. the plant's own last input measurement; exact
  threshold, no noise band). 17.0% of test pairs (180/1,057) are
  flat-or-decline; 79.1% of test plants (87/110) have at least one such
  segment. This is mostly real, not measurement jitter: only 12.8% are
  exact ties, while 47.8% are ≥5mm true declines (median delta -4mm, min
  -44mm). It is also **strongly concentrated late-season**: flat-or-decline
  segments have median normalized trajectory position 0.81 (vs. 0.41 for
  growth segments), and the last season quartile is 41.3% flat-or-decline
  vs. 2.5–10% elsewhere. Across the full 738-plant population, **83.8% of
  all declines fall in the late half of a plant's own trajectory**. The
  CNN-LSTM's error concentrates specifically on these segments: MAE 4.243
  on growth segments vs. **9.477 on flat-or-decline segments (2.23x
  worse)**. This reads as a real, late-season biological transition
  (consistent with senescence, head-formation dynamics, or
  harvest-related handling affecting the visible canopy) that the model
  saw comparatively little training signal for (83% of pairs are
  monotonic growth) — not generic scattered measurement noise.
- **Small phenotyped-plant population.** Only 739 plants carry the
  in-situ trait labels needed for regression (`task=='reference'` in the
  HF release), not the ~14,000 originally assumed — this constrains both
  training data volume (517 train plants) and how confidently results
  generalize.
- **Single target field/trait.** Results are for plant diameter only, on
  two fields (Field1/2020, Field2/2021) with different growing seasons
  and equipment (490×490px vs 256×256px crops). Not yet tested on other
  traits (height, head diameter, BBCH stage) or pooled across a larger
  set of fields/seasons.
- **Variable forecast horizon.** Forecast gap between consecutive valid
  measurements ranges from 2–42 days (median 7); the model is not
  horizon-conditioned beyond the day-after-planting feature, so a 7-day
  and a 40-day forecast are treated identically at the architecture level.
- **Dataset license unresolved** (see Dataset section above) — confirm
  with the corresponding author before any wider sharing or publication.

## Phase 2 — planned, not started

Explicitly out of scope for this checkpoint; not begun in this
repository:

1. **CNN-Transformer** model (in place of, or alongside, the CNN-LSTM),
   using a **continuous positional encoding** (e.g. keyed on
   day-after-planting rather than sequence index) to reflect the
   irregular acquisition cadence documented above, evaluated with the
   same shared-evaluation-set convention against the Step 5/6 results.
2. **Multi-trait regression** — extending beyond plant diameter to
   height, head diameter, and/or BBCH developmental stage, jointly or
   per-trait.
3. **Missing-observation robustness test** — evaluating how gracefully
   each model degrades as input frames are synthetically dropped, given
   the real-world irregularity already characterized above (6–15 dates
   per plant, ~83% trait coverage per acquisition date).

## Status

Steps 1–6 (baselines + CNN-LSTM, Phase 1 in full) complete and verified
on Swan:
- All pipeline artifacts (`data/metadata.parquet`, `data/pairs.parquet`,
  `data/pairs_split.parquet`, `data/norm_stats.json`) regenerated on Swan
  and confirmed to exactly match local runs (9,377 reference rows / 739
  plants, 7,031 pairs / 738 plants, 517/111/110 plant-wise split).
- Step 4 full run completed on an NVIDIA A30 GPU (`gpu` partition):
  9,377/9,377 images downloaded (0 failures) and embedded (0 failures),
  739/739 plants have a shard file, index verified structurally correct.
- Step 5 baselines run on Swan (CPU, login node — no GPU needed for ridge
  regression on 512-dim vectors).
- Step 6 CNN-LSTM: single config then 18-config sweep, both trained and
  evaluated on Swan (NVIDIA A30 GPU), with the shared-eval-set comparison
  against Step 5 and the quantified late-season structural limitation.

**This is the end of Phase 1.** Do not start the CNN-Transformer,
multi-trait regression, or missing-observation robustness experiments
without explicit approval — see [Phase 2](#phase-2--planned-not-started)
above for what's next.
