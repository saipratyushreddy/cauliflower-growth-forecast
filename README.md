# Cauliflower Growth Forecasting (GrowliFlower)

The project goal, as scoped by the course/research prompt, is to compare
**three** approaches for forecasting cauliflower growth traits from UAV
image time series: (1) simple baselines, (2) a CNN-LSTM, and (3) a
CNN-Transformer, against real data on UNL's Swan HPC cluster.

**Phase 1** (baselines + CNN-LSTM) is complete, sweep-tuned, and
rigorously evaluated. **Phase 2** is in progress: the CNN-Transformer has
been implemented, swept, and compared (see
[Phase 2 results](#phase-2-results--cnn-transformer) below), and a
missing-observation robustness test comparing both frozen models has
also been completed (see
[Phase 2 results — missing-observation robustness](#phase-2-results--missing-observation-robustness)).
Multi-trait regression is still planned, not started (see
[Phase 2 — remaining work](#phase-2--remaining-work)).

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

### Phase 2, part 1 — CNN-Transformer (hyperparameter sweep)
```bash
sbatch scripts/sweep_cnn_transformer.slurm
```
Same architecture-swap approach as Step 6 (reuses the cached embeddings),
same val-only model-selection discipline, plus a 4-way shared-eval-set
comparison against all three Phase 1 methods. See
[Phase 2 results](#phase-2-results--cnn-transformer) below.

### Follow-up — positional encoding d_model sensitivity check
```bash
sbatch scripts/check_pe_dmodel_sensitivity.slurm
```
Narrow, hypothesis-driven check (not a full re-sweep): retrains
`d_model ∈ {64,128,256,512}` with other hyperparameters fixed near the
sweep winner's, to test whether the aliasing symptom found in the
positional encoding sanity check is actually causing the Transformer's
underperformance vs. the CNN-LSTM. Same val-only selection, single test
evaluation. See [Phase 2 results](#phase-2-results--cnn-transformer).

### Phase 2, part 2 — missing-observation robustness test
```bash
sbatch scripts/robustness_missing_obs.slurm
```
Inference-only (frozen checkpoints, no retraining). Two follow-up
analysis scripts, run after the main job:
```bash
python src/analyze_robustness_results.py \
    --per-pair-csv outputs/step_p2_robustness_per_pair.csv --pairs-split data/pairs_split.parquet
python src/dig_into_n_kept_2.py --per-pair-csv outputs/step_p2_robustness_per_pair.csv
```
See [Phase 2 results — missing-observation robustness](#phase-2-results--missing-observation-robustness).

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
- **Two individual test plants with outsized error, traced to a mix of
  causes — not fully explained by the late-season pattern alone.**
  Step 5 found 2 of 110 test plants (`2021_Ref_Plot2_A1`,
  `2021_Ref_Plot2_E18`) accounted for over half the single-frame
  baseline's worst misses. Both have flat-or-decline rates well above
  average (42.9% and 41.7% vs. 17.0% test-wide), so the late-season
  pattern above explains a real part of it. But a follow-up check found
  each also has a distinct, traceable issue: (1) `2021_Ref_Plot2_A1`'s
  single largest error (50.2mm) stems from one upstream raw measurement,
  `"58 (7)"` at day 35, cleaned to 58 under the Step 2 "use X, discard Y"
  rule — the surrounding trajectory (`33→58→14→16→30→39...`) suggests the
  discarded value (7) was plausibly closer to the true reading, meaning
  this specific data point may be an artifact of that cleaning decision
  rather than real growth; (2) `2021_Ref_Plot2_E18` has a day-44
  measurement explicitly marked missing with the field comment *"not
  measureable due to grass-overgrowth"* — a genuine, documented
  data-quality event distinct from the late-season biological story. Not
  corrected here (would require re-deciding the Step 2 cleaning rule for
  a single record), but worth flagging as a concrete example of how a
  reasonable global cleaning rule can occasionally produce a local
  outlier.
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

## Phase 2 results — CNN-Transformer

Encoder-only Transformer over the same cached ResNet18 embeddings as the
CNN-LSTM, using a **continuous sinusoidal positional encoding keyed on
each timestep's real `day_after_planting` value** (raw days, not
normalized) in place of integer sequence position — directly encoding
the irregular 2–42 day acquisition gaps documented above, parameter-free,
and generalizes to gap lengths not seen in training (see
`src/cnn_transformer_model.py` for the exact formula). Causal + padding
masking ensures no future timestep or padded position is attended to.

**24-config hyperparameter sweep** (`d_model ∈ {64,128} × nhead ∈ {4,8}
× num_layers ∈ {1,2} × lr ∈ {1e-3,3e-4}`, dropout ∈ {0.0,0.2} for
`num_layers=2` only), same model-selection discipline as the CNN-LSTM
sweep: every config compared by **validation MAE only**, test evaluated
**exactly once** for the winning config (verified: `grep -c "TEST
EVALUATION"` = 1). Ran on an NVIDIA A30 GPU on Swan, 1h07m total, clean
`.err` log.

```bash
sbatch scripts/sweep_cnn_transformer.slurm
```

**Winning config: `d64_h4_lr0.0003_L2_do0.2`** (d_model=64, 4 heads,
2 layers, lr=3e-4, dropout=0.2), val MAE=5.097 — full 24-config table in
`outputs/step_p2_transformer_sweep_results.csv`.

**4-way shared-evaluation-set comparison** (1,057 test pairs every method
can predict):

| Method | MAE | RMSE |
|---|---|---|
| Persistence | 9.068 | 10.914 |
| Single-frame | 5.914 | 8.657 |
| **CNN-Transformer (sweep winner)** | **5.528** | **8.668** |
| **CNN-LSTM (Phase 1 sweep winner)** | **5.134** | **8.055** |

**Honest result: the Transformer beats single-frame but loses to the
CNN-LSTM.** It improves MAE by ~6.5% over single-frame, but its RMSE
(8.668) is essentially tied with single-frame's (8.657) — no real gain
there. Against the CNN-LSTM, it is clearly worse: **+7.7% MAE, +7.6%
RMSE** (more error, not less). The size-quartile bias correlation
(-0.229) falls between the two CNN-LSTM results (-0.265 manual, -0.215
sweep-tuned) rather than continuing that trend's improvement — it
interrupts, rather than extends, the Phase 1 bias-reduction pattern.

This is reported as a genuine result, not downplayed.

**Before concluding the architecture itself underperforms, two
alternative explanations were explicitly tested and ruled out, rather
than assumed away:**

1. **Was this an unfair comparison (untuned Transformer vs. tuned
   LSTM)?** No — both are hyperparameter-sweep winners under the
   identical val-only selection discipline (24 configs vs. 18 configs),
   with test touched exactly once for each (verified:
   `grep -c "TEST EVALUATION"` = 1 for both sweep logs).
2. **Is the continuous positional encoding broken or aliased at the
   winning config's small `d_model=64`?** This was checked directly, not
   assumed. A cosine-similarity sanity check on real
   `day_after_planting` values (15, 22, 42, 83, 91) confirmed the
   encoding correctly orders gap sizes (2-day gap similarity 0.884 >
   7-day 0.727 > 14-day 0.631 > 42-day 0.559 — larger gaps are
   correctly less similar) — **but also found a genuine aliasing
   symptom**: `day=91` was more cosine-similar to the distant `day=15`
   (0.582) than to the much closer `day=22` (0.450), consistent with
   sinusoidal PE's known periodicity behavior when `d_model` is small
   relative to the position range (here, days up to ~93 encoded in only
   64 dimensions). This was a real, plausible concern, not a strawman —
   so it was tested directly: a follow-up grid retrained `d_model ∈
   {64,128,256,512}` (other hyperparameters fixed near the original
   winner) using the same val-only selection and single test evaluation.
   **Result: larger `d_model` made validation performance monotonically
   *worse*** (val MAE 5.097→5.217→5.231→5.282 as `d_model` rises from 64
   to 512), the opposite of what the aliasing-is-the-bottleneck
   hypothesis predicts (a larger `d_model` reduces aliasing by spreading
   the same position range across more frequency dimensions). This
   pattern — larger capacity hurting, not helping — is the standard
   signature of overparameterization on a small dataset, not an
   encoding-quality problem. **The aliasing symptom is real but was
   directly ruled out as the cause of the Transformer's underperformance.**

With that alternative explanation tested and excluded, the standing
explanation is now evidenced rather than speculative: with only 517
training plants and short sequences (1–14 frames), the Transformer's
larger parameter space and weaker sequential inductive bias (versus the
LSTM's recurrence, which enforces processing order structurally) needs
more data than is available here to outperform an architecture that
already assumes temporal order — consistent with Phase 1's own finding
that the CNN-LSTM's hyperparameter landscape was already fairly flat
(5.007–5.448 val MAE across 18 configs), suggestive of a data-limited
regime.

Growth curve plots for the Transformer's winning config:
`outputs/step_p2_transformer_growth_curve_<plant_id>.png` (6 test
plants, same set as the Phase 1 CNN-LSTM plots, for direct comparison).

Spot-checking the same plant used for the Phase 1 CNN-LSTM comparison
(`2020_Ref_Plot1_A93`) makes the Transformer's weakness concrete: on the
final measurement (day 93), the actual diameter **drops** to 44mm (from
48mm at day 91), but the Transformer predicts a sharp **jump to ~92mm** —
a dramatically larger miss than the CNN-LSTM's own error on this same
plant's same late-season segment. This is a direct visual instance of why
the Transformer loses to the CNN-LSTM in aggregate: it appears to
struggle more, not less, on exactly the late-season non-monotonic
segments already identified as the model family's shared weak point.

![CNN-Transformer growth curve: late-season miss, same plant as the CNN-LSTM comparison](outputs/step_p2_transformer_growth_curve_2020_Ref_Plot1_A93.png)

## Phase 2 results — missing-observation robustness

**Inference-time evaluation only — no retraining, no hyperparameter
changes.** Loads the two already-trained, frozen sweep-winner checkpoints
(CNN-LSTM: `h256_lr0.001_L2_d0.2`; CNN-Transformer:
`d64_h4_lr0.0003_L2_do0.2`) exactly as saved. For each of the 1,057
shared-eval-set test pairs, at drop levels {0%, 25%, 50%, 75%} of that
pair's available input observations, a random subset is removed
(chronological order preserved among what remains, never dropping below
1 observation), and both models run inference on the **identical**
dropped-observation set per trial — the drop mask is computed once per
`(pair, drop_frac, seed)` and reused for both models' own input
representations, verified directly in the code (`src/robustness_missing_obs.py`),
not just by design intent. 5 random seeds per non-zero drop level.

```bash
sbatch scripts/robustness_missing_obs.slurm
```

**Aggregate table** (mean ± std MAE/RMSE over 5 seeds; 0% is
deterministic):

| Drop level | LSTM MAE | LSTM RMSE | Transformer MAE | Transformer RMSE |
|---|---|---|---|---|
| 0% | 5.184 ± 0.000 | 8.146 ± 0.000 | 5.587 ± 0.000 | 8.775 ± 0.000 |
| 25% | 7.023 ± 0.105 | 9.842 ± 0.162 | 6.677 ± 0.121 | 9.750 ± 0.155 |
| 50% | 11.447 ± 0.121 | 14.797 ± 0.204 | 9.288 ± 0.166 | 12.643 ± 0.166 |
| 75% | 18.978 ± 0.264 | 24.382 ± 0.434 | 13.841 ± 0.238 | 19.037 ± 0.351 |

At 0% drop the LSTM has lower error (matches the full-eval Phase 2
result above). At every non-zero drop level, the ranking flips: the
Transformer has lower MAE and RMSE, and the gap widens as more
observations are dropped.

**Before accepting this as "attention-based temporal modeling is more
robust to missing observations" in the general sense, three follow-up
checks were run** — the aggregate table alone doesn't distinguish a real
general effect from an artifact of a few plants or a length-specific
quirk:

1. **Fairness (same dropped set hits both models)**: confirmed by direct
   code inspection — `keep_idx` is computed once per trial and reused for
   both `predict_lstm()` and `predict_transformer()`; no independent
   second RNG draw exists in the loop.
2. **Broad or concentrated across plants?** Broad. At 50% drop, 94/110
   test plants (85.5%) favor the Transformer, and the top 5 plants
   account for only 11.1% of the total advantage; at 75% drop, 99/110
   (90.0%) favor the Transformer, top 5 = 8.5% of total. Notably, the
   plants where the LSTM does relatively *better* include the same three
   plants flagged earlier for outsized error (`2021_Ref_Plot2_A1`,
   `2021_Ref_Plot2_E18`, `2020_Ref_Plot3_A95` — see Limitations, the
   traced cleaning-rule artifact and documented "grass-overgrowth"
   measurement gap).
3. **Smooth degradation, or a cliff at short lengths?** A cliff — and
   this changes the correct framing substantially. Bucketing error by
   REMAINING observation count (`n_kept`, not drop %) shows both models
   are statistically indistinguishable at `n_kept=1` (median abs error
   4.92 LSTM vs. 5.53 Transformer — LSTM is marginally *better* here),
   but the gap opens sharply at `n_kept=2` (median 14.53 vs. 7.34) and
   stays wide at `n_kept=3` (median 12.68 vs. 6.62), before narrowing
   again at longer remaining lengths. This is not a mean skewed by a few
   pathological predictions: at `n_kept=2`, 40.1% of all 3,739 LSTM
   trials exceed 20mm absolute error, spread across 77 of 110 test
   plants (median stays close to the mean throughout) — a genuine,
   broad-based degradation specific to very short remaining sequences,
   not an outlier artifact.

**Conclusion, stated at the precision the evidence supports:** this is
**not** general evidence that attention-based temporal modeling is more
robust to missing observations. It is more specific: **the CNN-LSTM's
error is roughly 2x larger than the CNN-Transformer's specifically at
very short remaining-observation counts (2–3 frames)**, broad-based
across most of the test population rather than a few plants, and
narrowing at longer remaining lengths. A plausible (not proven)
explanation is a training/inference length mismatch: Step 6 trained the
LSTM only on the natural, denser sequences as they occur (median length
5), with no observation-dropout augmentation, so a synthetically-created
2–3-frame sequence is a different, thinner-sampled distribution than
anything the LSTM's recurrence was calibrated on — whereas the
Transformer's attention mechanism may degrade more gracefully to small
input sets somewhat independent of *why* they are small. This was not
directly isolated (would require training an LSTM variant with
observation-dropout augmentation to test in the "does training on
shortened sequences fix this" sense, which is out of scope for an
inference-only robustness test) — it is offered as the most consistent
explanation for the observed length-specific pattern, not a proven
mechanism.

## Phase 2 — remaining work

1. **Multi-trait regression** — extending beyond plant diameter to
   height, head diameter, and/or BBCH developmental stage, jointly or
   per-trait.

## Image-to-image future growth prediction (primary track)

This track supersedes diameter regression (documented above as Phase 0 and
left unchanged) as the headline deliverable: given a plant's image history,
predict its next acquired image.

**Pairs** (`src/build_image_pairs.py` -> `data/image_pairs.parquet`): for a
plant's k-th image (k>=1) the target is that image and the input is all
earlier images. Every acquisition counts; no diameter label is required.
8,638 pairs before cleaning, **8,386 after** (below).

**Split.** All 738 plants retain their Phase 0 split assignment, preserving
an identical test set across tasks. The 739th plant (`2021_Ref_Plot2_B13`,
excluded from Phase 0 for lacking 2+ diameter measurements) is assigned via
`RandomState(42)` applied to that single plant, which deterministically
yields train (`round(0.7*1)=1`). Re-running the Phase 0 scheme on the full
739-plant list was rejected because it reshuffles the entire permutation (125
plants changed split, 29 Phase 0 train plants became test).

**Resolution: all images are resized to 256x256.** Field1/2020 images
(natively 490x490) are **downsampled** (Lanczos); Field2/2021 images are
natively 256x256 and untouched. Upsampling Field2 instead would mean
generating detail that never existed in the source, making SSIM/PSNR
comparisons misleading. Because Field1 is resized and Field2 is not, results
are reported **pooled and per-field as co-equal results** (test is ~80%
Field2, so the pooled number alone is dominated by it).

### Image quality exclusions (applied) and checks (not excluded)

| Step | Train | Val | Test | Total |
|---|---|---|---|---|
| Unfiltered pairs | 6,015 | 1,346 | 1,277 | 8,638 |
| Black/placeholder frames removed | -178 | -16 | -40 | -234 |
| Blurry frames removed | -14 | 0 | -4 | -18 |
| **Final cleaned pairs** | **5,823** | **1,330** | **1,233** | **8,386** |

Final composition: train 518 plants (Field1 1,021 / Field2 4,802 pairs), val
111 plants (168 / 1,162), test 110 plants (215 / 1,018). Plant split
assignments are unchanged by any exclusion; only pairs are removed.

1. **Black/placeholder frames (excluded, 234 images).** All Field1, day
   91/93 (last 1-2 acquisitions of 119 plants): 232 byte-identical
   placeholders (md5 `1fe4b7f0...`) and 2 near-black frames (<10KB). They
   produced 113 pixel-identical input/target pairs (SSIM=1.0) that inflated
   an earlier Field1 SSIM. Every pair touching one is dropped
   (`--exclude-list`).
2. **Blurry frames (excluded, 9 images)**, from the blur scan plus a visual
   check on a contact sheet (same-plant previous/next dates and same-date
   same-plot / other-plot references alongside): `Plot5_{A16,A17,A18,B17}`
   on 2021-08-11 and `Plot1_{A4,A5,A6,B6,B7}` on 2021-08-30. Leaf edges are
   smeared (a resampled/stitching look, not darkness) while the same
   plants a week earlier and same-date references are crisp; the affected
   plant IDs are adjacent, suggesting a local region of the plot (a
   hypothesis; the source orthomosaics were not inspected). Each is removed
   as target (9 pairs) and as last input (9 pairs); no pair is created
   across a removed frame. In 25 retained pairs such a frame appears only in
   older history and is scrubbed from it (`BLURRY_EXCLUDE` in
   `src/build_image_pairs.py`). Dropping every pair with one anywhere in its
   history instead would remove 43 pairs (33/0/10); not applied.
   *Explicitly checked:* in all 25 pairs where a blurry frame was scrubbed
   from older history (19 train, 6 test), the gap from the last input to the
   target is identical before and after scrubbing (0 of 25 differ, as expected:
   a pair whose last input is a blurry frame is dropped, never scrubbed), so the
   lenient rule is kept. Side effect, for later sequence models: the scrub
   leaves an internal hole in the history of the 20 Plot5 pairs (max gap
   between consecutive history frames 9 -> 15 days; Plot1 pairs unchanged at 9).
3. **Field1 day 28 (2020-08-25): NOT excluded, stratified.** Dark but sharp
   (mean luma 37-40 vs ~108-112 on adjacent dates; leaf veins in focus; the
   whole flight, other plots even darker), i.e. valid data with a large
   exposure change. It touches 478 of 1,404 Field1 pairs (74 of 215 test).
   Reported with and without it (see evaluation protocol).
4. **Plot1 2021-06-16 (day 1): checked and cleared.** Bare soil with tiny
   seedlings; flagged only because low-texture soil gives low variance of
   Laplacian. As sharp as same-date references.

**Limitation of the blur scan (stated, not open).** Variance of Laplacian
flags the bottom 1% within each (field, day) group *by construction*, so it
cannot say how many other images are soft; 88 images were flagged and only
the four clusters above were inspected. It also confuses blur with
darkness/low contrast. Residual softness outside the 9 excluded frames may
exist in both inputs and targets and is not quantified. Only a visual check
at ~200px thumbnails was done.

### Evaluation protocol (final)

Reported on the **test split of the cleaned pair set** (1,233 pairs), for:
pooled; pooled excluding Field1 day-28 pairs; Field1; Field1 excluding day
28; Field1 day 28 only (and its two directions); Field2. Each reports **mean
and median** of RGB SSIM, RGB PSNR and structure SSIM (luma, local contrast
normalization with Gaussian sigma=7, eps=0.05, clip +-3, SSIM data_range=6).
A "day-28 pair" is a Field1 pair whose last input or target is day 28.
Medians are always reported because SSIM is heavy-tailed. Gap-length buckets
are not used (confounded with growth stage and field; see below). Scores
come from `src/image_copy_forward_baseline.py`, aggregated by
`src/summarize_baseline.py` (`outputs/img_stepB_final_protocol.json`).

### Step B: copy-forward baseline (honest) -- results

Prediction = the plant's last input image, unchanged. Test split:

| Subset | Pairs | RGB SSIM mean (median) | RGB PSNR dB mean (median) | Structure SSIM mean (median) |
|---|---|---|---|---|
| Pooled | 1,233 | 0.122 (0.072) | 10.04 (9.06) | 0.044 (0.028) |
| Pooled excl. Field1 day 28 | 1,159 | 0.124 (0.070) | 10.10 (9.08) | 0.046 (0.029) |
| Field1 (downsampled) | 215 | 0.096 (0.085) | 11.32 (10.53) | 0.018 (0.016) |
| Field1 excl. day 28 | 141 | 0.096 (0.082) | 12.53 (11.59) | 0.019 (0.014) |
| Field1 day 28 only | 74 | 0.095 (0.089) | 9.03 (8.98) | 0.017 (0.017) |
| &nbsp;&nbsp;target is day 28 (bright->dark) | 37 | 0.095 (0.083) | 8.77 (8.75) | 0.018 (0.018) |
| &nbsp;&nbsp;input is day 28 (dark->bright) | 37 | 0.094 (0.098) | 9.30 (9.35) | 0.016 (0.016) |
| Field2 (native) | 1,018 | 0.128 (0.067) | 9.76 (8.71) | 0.050 (0.032) |

- **Day 28 is an exposure effect that only PSNR sees.** Day-28 pairs score
  3.5 dB lower PSNR than the rest of Field1 (9.03 vs 12.53) but the same
  RGB SSIM (0.095 vs 0.096) and structure SSIM (0.017 vs 0.019). Holding
  them out moves the pooled numbers very little (PSNR 10.04 -> 10.10, SSIM
  0.122 -> 0.124) because they are 6% of test pairs.
- **Field1 vs Field2 is not attributable to the field.** The metrics
  disagree on direction (Field1 higher PSNR, lower SSIM and structure
  SSIM) and Field1 is the downsampled field.
- **Structure SSIM is very low everywhere (0.044 pooled)** even with
  exposure and local contrast removed, so most copy-forward error is spatial
  (canopy/leaf layout change, sub-plant misregistration between flights,
  real growth), not lighting.
- **SSIM is heavy-tailed** (Field2 mean 0.128, median 0.067), driven by a
  few static, dim early-season pairs. The earlier gap-bucket breakdown
  (SSIM did not fall with gap) is confounded with growth stage and field and
  is not evidence about forecast horizon.

### Oracle colour-match -- reference only; NOT a baseline and NOT an upper bound

**ORACLE: it USES THE TARGET IMAGE'S per-channel mean/std, so it cannot be
run at inference, and it is not comparable to the baseline above or to any
trained model.** *Correction (supersedes the earlier wording that it
"bounds" the exposure error):* it is **not a ceiling**. Control 2 below
uses only training-set statistics and reaches within ~0.003 SSIM / ~0.24
dB of it (pooled: SSIM 0.155 vs 0.158, PSNR 12.44 vs 12.68 dB), so
exposure is almost entirely predictable from (field, input_day,
target_day) alone, and the trained model exceeds the oracle on the
day-28-target pairs (28.2 vs 24.4 dB).

| Subset (test) | Pairs | ORACLE SSIM mean (median) | ORACLE PSNR dB mean (median) |
|---|---|---|---|
| Pooled | 1,233 | 0.158 (0.075) | 12.68 (10.29) |
| Pooled excl. Field1 day 28 | 1,159 | 0.151 (0.074) | 12.34 (9.80) |
| Field1 | 215 | 0.160 (0.103) | 14.87 (13.90) |
| Field1 excl. day 28 | 141 | 0.103 (0.103) | 13.28 (13.90) |
| Field1 day 28 only | 74 | 0.268 (0.247) | 17.91 (17.78) |
| &nbsp;&nbsp;target is day 28 (bright->dark) | 37 | 0.467 (0.455) | 24.35 (24.16) |
| &nbsp;&nbsp;input is day 28 (dark->bright) | 37 | 0.069 (0.070) | 11.47 (11.44) |
| Field2 | 1,018 | 0.158 (0.070) | 12.21 (9.51) |

**Day-28 cross-check (Field1 test).** Colour matching gains +8.9 dB on
day-28 pairs versus +0.75 dB on the rest of Field1, taking day-28 PSNR from
3.5 dB *below* the rest (9.03 vs 12.53) to 4.6 dB *above* it (17.91 vs
13.28): it closes the whole gap and overshoots. This confirms the day-28
effect is global exposure, consistent with day-28 pairs having unchanged
SSIM and structure SSIM relative to the rest of Field1. (The "fraction of
gap closed" values in the JSON, 2.5 for PSNR and 145 for SSIM, are not
meaningful for SSIM because the honest SSIM gap is ~0.001; use the gains.)
Always read the oracle stratified: its Field1 headline (0.160 SSIM) is
driven by day-28 pairs.

### Step C: single-frame learned model (first trained image model)

Input per pair: only the most recent image plus the scalar day offset
(`target_day - input_last_day`); earlier history is never read.
`src/img_film_unet.py` (9.3M parameters): ResNet-style encoder to an 8x8
bottleneck, FiLM (scale/shift from an MLP of the offset, zero-initialised)
at the bottleneck, U-Net decoder with skips, sigmoid output generated
directly (no residual to the input). Loss = 1.0 L1 + 1.0 (1 - SSIM)
(differentiable Gaussian-window SSIM; no adversarial term, no colour-match
logic), untuned weights. Train-only augmentation: joint flips / 90-degree
rotations. `src/train_img_single_frame.py`, `sbatch
scripts/train_img_single_frame.slurm`. Pairs: 5,823 train / 1,330 val / 1,233
test. Early-stopped at epoch 73, best epoch 61 chosen on validation combined
loss only (0.902), ~33 s/epoch, 40 min total; test evaluated exactly once.
Validation plateaus around epochs 40-60 while train keeps improving (SSIM
0.288 train vs 0.273 val at the selected epoch): mild overfitting. Training
SSIM uses a Gaussian window; evaluation uses the same skimage metrics as every
other row below.

### Step C controls (closed-form, no training)

Test touched exactly once per control. All use the copy-forward base.

- **Control 1, blurred copy-forward:** Gaussian blur (sigma = 16), chosen as
  the grid value (0.5-64) maximizing mean *validation* RGB SSIM. (Chosen over
  "matching the model's validation SSIM" because that figure comes from a
  different SSIM implementation, and the maximum gives the control its best
  shot.) Validation structure SSIM peaks at sigma = 2 (0.0547 vs 0.0455
  unblurred) and falls to 0.0483 at sigma 16: no blur sigma comes near the
  model's 0.082 on this metric (validation-only evidence; a second sigma was not
  evaluated on test).
- **Control 2, flight-aware colour copy-forward:** per (field, input_day,
  target_day) bucket, from training pairs only, the mean of the target
  images' per-channel mean/std; each test input is standardized by its own
  per-channel mean/std and rescaled to the bucket's values (no test-target
  statistics). 22 buckets; training pairs per bucket used for a test pair: min
  7, median 344, max 344. Buckets with <10 training pairs fall back to
  (field, target_day).
- **Control 3:** Control 2's colour transform, then Control 1's blur (same
  sigma).

**Caveat on control 2 / the oracle's apparent success:** exposure is
predictable here because every plant in a field is photographed on the same
dates. This is an in-dataset property and will not transfer to unseen
flights or seasons.

#### Results (test, 1,233 pairs, mean (median))

**Structure SSIM -- the headline comparison**

| Subset | Copy-fwd | C1 blur s=16 | C2 flight-colour | C3 colour+blur | **Model (Step C)** |
|---|---|---|---|---|---|
| Pooled | 0.044 (0.028) | 0.049 (0.032) | 0.048 (0.030) | 0.049 (0.032) | **0.082 (0.054)** |
| Pooled excl. Field1 day 28 | 0.046 (0.029) | 0.047 (0.032) | 0.049 (0.030) | 0.047 (0.032) | **0.080 (0.054)** |
| Field1 | 0.018 (0.016) | 0.047 (0.030) | 0.022 (0.016) | 0.052 (0.031) | **0.071 (0.050)** |
| Field1 excl. day 28 | 0.019 (0.014) | 0.029 (0.030) | 0.020 (0.015) | 0.030 (0.031) | **0.050 (0.048)** |
| Field1 day 28 only | 0.017 (0.017) | 0.081 (0.065) | 0.026 (0.025) | 0.094 (0.082) | **0.111 (0.102)** |
| &nbsp;&nbsp;target is day 28 | 0.018 (0.018) | 0.138 (0.133) | 0.042 (0.041) | 0.168 (0.161) | **0.184 (0.175)** |
| &nbsp;&nbsp;input is day 28 | 0.016 (0.016) | 0.023 (0.024) | 0.011 (0.010) | 0.020 (0.020) | **0.037 (0.038)** |
| Field2 | 0.050 (0.032) | 0.049 (0.032) | 0.054 (0.034) | 0.049 (0.032) | **0.085 (0.056)** |

Paired difference, model minus control, structure SSIM, mean (median), and
share of pairs where the model is better:

| Subset | Pairs | vs copy-fwd | vs C1 | vs C2 | vs C3 |
|---|---|---|---|---|---|
| Pooled | 1,233 | +0.038 (+0.027), 87% | +0.034 (+0.022), 88% | +0.034 (+0.026), 86% | +0.033 (+0.021), 88% |
| Pooled excl. day 28 | 1,159 | +0.035 (+0.026), 86% | +0.034 (+0.022), 88% | +0.031 (+0.025), 85% | +0.034 (+0.021), 88% |
| Field1 | 215 | +0.052 (+0.033), 99% | +0.024 (+0.021), 91% | +0.049 (+0.033), 99% | +0.019 (+0.016), 92% |
| Field1 excl. day 28 | 141 | +0.031 (+0.030), 99% | +0.021 (+0.021), 90% | +0.030 (+0.030), 99% | +0.020 (+0.017), 89% |
| Field1 day 28 only | 74 | +0.094 (+0.084), 100% | +0.030 (+0.031), 92% | +0.084 (+0.075), 100% | +0.016 (+0.016), 99% |
| &nbsp;&nbsp;target is day 28 | 37 | +0.166 (+0.166), 100% | +0.046 (+0.045), 100% | +0.142 (+0.142), 100% | +0.016 (+0.015), 100% |
| &nbsp;&nbsp;input is day 28 | 37 | +0.021 (+0.020), 100% | +0.015 (+0.014), 84% | +0.027 (+0.028), 100% | +0.017 (+0.018), 97% |
| Field2 | 1,018 | +0.035 (+0.024), 84% | +0.036 (+0.022), 88% | +0.031 (+0.023), 84% | +0.036 (+0.022), 88% |

**Structure SSIM verdict.** Pooled, copy-forward 0.044 -> best control 0.049
-> model 0.082. The model is ahead of every control in every subset; the
smallest margin against Control 3 is +0.016 (day-28 pairs, including the
target-is-day-28 direction), and the smallest margin anywhere is +0.015
(against Control 1, input-is-day-28 pairs). Blur and per-flight exposure
therefore do **not** account for the model's structure SSIM gain. This is a
statement about structure SSIM only; absolute structural fidelity remains low
(median 0.054), and all outputs are blurry.

**RGB SSIM**

| Subset | Copy-fwd | C1 | C2 | C3 | Model |
|---|---|---|---|---|---|
| Pooled | 0.122 (0.072) | 0.189 (0.116) | 0.155 (0.072) | 0.217 (0.121) | 0.254 (0.159) |
| Pooled excl. Field1 day 28 | 0.124 (0.070) | 0.188 (0.114) | 0.149 (0.070) | 0.206 (0.117) | 0.243 (0.156) |
| Field1 | 0.096 (0.085) | 0.204 (0.224) | 0.156 (0.105) | 0.276 (0.231) | 0.304 (0.255) |
| Field1 excl. day 28 | 0.096 (0.082) | 0.207 (0.224) | 0.102 (0.105) | 0.212 (0.231) | 0.238 (0.255) |
| Field1 day 28 only | 0.095 (0.089) | 0.198 (0.184) | 0.260 (0.263) | 0.399 (0.385) | 0.430 (0.423) |
| &nbsp;&nbsp;target is day 28 | 0.095 (0.083) | 0.289 (0.261) | 0.451 (0.445) | 0.660 (0.648) | 0.703 (0.701) |
| &nbsp;&nbsp;input is day 28 | 0.094 (0.098) | 0.107 (0.111) | 0.068 (0.069) | 0.138 (0.149) | 0.157 (0.167) |
| Field2 | 0.128 (0.067) | 0.185 (0.107) | 0.155 (0.067) | 0.205 (0.110) | 0.243 (0.146) |

**RGB PSNR (dB)**

| Subset | Copy-fwd | C1 | C2 | C3 | Model |
|---|---|---|---|---|---|
| Pooled | 10.04 (9.06) | 11.61 (10.97) | 12.44 (10.01) | 14.40 (12.04) | 14.84 (12.40) |
| Pooled excl. Field1 day 28 | 10.10 (9.08) | 11.76 (11.15) | 12.12 (9.56) | 14.06 (11.74) | 14.43 (12.11) |
| Field1 | 11.32 (10.53) | 12.91 (12.18) | 14.73 (13.93) | 16.97 (16.09) | 17.80 (16.66) |
| Field1 excl. day 28 | 12.53 (11.59) | 14.83 (14.38) | 13.23 (13.93) | 15.54 (16.09) | 15.94 (16.66) |
| Field1 day 28 only | 9.03 (8.98) | 9.25 (9.25) | 17.57 (17.27) | 19.70 (19.32) | 21.35 (21.12) |
| &nbsp;&nbsp;target is day 28 | 8.77 (8.75) | 9.15 (9.13) | 23.75 (23.83) | 25.81 (25.86) | 28.23 (28.27) |
| &nbsp;&nbsp;input is day 28 | 9.30 (9.35) | 9.35 (9.39) | 11.40 (11.40) | 13.59 (13.56) | 14.47 (14.67) |
| Field2 | 9.76 (8.71) | 11.33 (10.77) | 11.96 (9.26) | 13.85 (11.56) | 14.22 (11.79) |

**RGB SSIM and PSNR gains overstate the model's real progress.** Control 3
alone, a closed-form blur plus per-flight colour transform with no learning,
reproduces **72% of the model's SSIM gain** over copy-forward (+0.095 of
+0.132) and **91% of its PSNR gain** (+4.36 of +4.81 dB), but only **13% of
its structure SSIM gain** (+0.005 of +0.038). These two metrics must not be
used alone to judge any future model stage.

**Thin-bucket caveat.** Three Field1 test pairs (day 83 -> 97:
`Plot3_A9`, `Plot3_C10`, `Plot3_C7`) have only 7 training pairs even in
Control 2's (field, target_day) fallback bucket (below the 10-pair
threshold). That bucket was still used (flagged `thin_bucket` in
`outputs/img_ctrl2_per_pair.csv`; the pairs were kept so every method is
scored on the identical 1,233 pairs). Control 2/3 results on that handful of
pairs are less reliable; it is 3 of 1,233 test pairs.

**The day-28 asymmetry: partly explained, partly still open.** Every method
that corrects exposure or smooths scores far higher when the target is day
28 than when the input is day 28: oracle SSIM 0.467 vs 0.069, Control 2
0.451 vs 0.068, Control 1 0.289 vs 0.107, Control 3 0.660 vs 0.138, model
0.703 vs 0.157. Plain copy-forward shows **no** such asymmetry (0.095 vs
0.094), so it appears only once a method alters the input, across methods
that differ in what they do (colour standardization, pure blur, learned
model). Three candidate explanations, and where they stand:

- **(c) SSIM rewards smooth output on a low-contrast target: premise
  confirmed, mechanism supported but not isolated.**
  `src/target_variance_check.py` on the 1,233 test pairs: day-28 *target*
  images have median luma std 11.0 vs 63.7 for all other targets (ratio
  0.17; vs 43.2 for other Field1 targets, ratio 0.25), median local std 0.020
  vs 0.146 (Mann-Whitney p ~1e-19 to 1e-21), mean luma 26 vs 101, and 93% of
  their pixels have local std below the structure-SSIM stabiliser eps = 0.05
  (14% for other targets, 7% for other Field1). Across *all* test pairs,
  per-pair SSIM falls with target contrast for every method (Spearman, target
  luma std: copy-forward -0.67, Control 1 -0.87, model -0.83); structure SSIM
  shows the same sign but weaker (-0.15, -0.32, -0.48). Caveats: contrast is
  confounded with darkness and with growth stage (low-contrast early-season
  targets are also the static, easy ones), no within-stage control was run,
  and because 93% of day-28 target pixels sit below eps, structure SSIM is
  partly degenerate on these 37 pairs. Consequence: day-28 target-pair scores
  (e.g. the model's 0.703 SSIM / 0.184 structure SSIM) overstate skill, which
  is why results are always also reported excluding day 28.
  The same 74 day-28 images are the inputs of the "input is day 28" pairs, so
  those inputs are equally low-contrast (median luma std 11.0); the
  asymmetry is about which side of the pair has the low contrast.
- **(a) Noise amplification from stretching a dark, low-contrast 8-bit input:
  does not explain Control 1** (pure blur, no stretching), **but is not
  excluded for the oracle and Control 2.** Both standardize an input whose
  std is ~11 up to a target std of ~60, a ~5x gain, so amplifying noise is
  plausible there. Untested.
- **(b) A genuinely larger growth step from day 28 to 36 than from 22 to 28:
  open, untested.**

**Qualitative grid (illustration only).** `outputs/img_ctrl_grid.png` shows
four test pairs (one day-28 target pair, one other Field1 pair, one
early-season and one late-season Field2 pair; fixed seed) as input, target,
copy-forward, C1, C2, C3 and model. It is an illustration only and is not
evidence on its own; the conclusions above rest on the metrics.
In the grid, all methods are blurry, none recovers leaf-level detail, and
the model retains coarse layout (for example the plant against soil in the
day 22 -> 28 pair) that the colour-and-blur controls lose.

### Evaluation protocol for all future model stages (temporal, Transformer)

Report **both** of the following, **together in the same table, always**,
never structure SSIM alone or RGB metrics alone:

1. **Structure SSIM against the best prior model stage** (for the temporal
   model: the Step C single-frame model, 0.082 pooled test), and
2. **Pooled RGB SSIM and PSNR against Control 3** (0.217 SSIM, 14.40 dB
   pooled test),

on the cleaned pair set, with the stratification above (pooled, per-field,
with/without day 28; mean and median). This is what catches a future model
finding a different shortcut.

## Status

**Phase 1 (baselines + CNN-LSTM) complete and verified on Swan:**
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

**Phase 2, part 1 (CNN-Transformer) complete:** 24-config sweep run on
Swan (NVIDIA A30 GPU, 1h07m, clean exit), same val-only model-selection
discipline as Phase 1, 4-way shared-eval-set comparison against all three
Phase 1 methods. Result: beats single-frame on MAE only (not RMSE), loses
to the CNN-LSTM on both metrics — reported honestly above, not as a win.
A follow-up positional-encoding aliasing symptom was found, tested via a
targeted d_model sensitivity check, and ruled out as the cause (larger
d_model made results monotonically worse, the opposite of what the
aliasing-is-the-bottleneck hypothesis predicts).

**Phase 2, part 2 (missing-observation robustness test) complete:**
inference-only evaluation of both frozen sweep-winner checkpoints at
input-drop levels {0%, 25%, 50%, 75%}, 5 seeds each, on the shared
1,057-pair eval set. Headline aggregate result (Transformer degrades more
slowly overall) was verified rather than taken at face value: the
same-dropped-set fairness constraint was confirmed by direct code
inspection, the effect was confirmed broad-based across ~85-90% of test
plants (not a few outliers), and bucketing by remaining-observation count
revealed the true mechanism is a ~2x LSTM-specific error spike
concentrated at very short remaining lengths (2-3 frames), not a general
degradation-rate difference — reported at that more precise, narrower
level rather than the broader claim the aggregate table alone would
suggest.

**Remaining:** multi-trait regression (see
[Phase 2 — remaining work](#phase-2--remaining-work) above).
Repository is public: https://github.com/saipratyushreddy/cauliflower-growth-forecast
