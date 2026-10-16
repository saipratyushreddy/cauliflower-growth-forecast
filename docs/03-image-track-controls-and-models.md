# Track 2: next-photo prediction, part 2 (controls, ConvLSTM, Transformer)

[Back to the project overview](../README.md)

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

### Step D: history-aware ConvLSTM (K = 4 frames, with K = 1 reference)

`src/img_convlstm.py`, `src/train_img_convlstm.py`, `sbatch scripts/train_img_convlstm.slurm`
(K and an optional SEED are passed via `--export`). Step C's shared encoder runs on the last K input
frames; a ConvLSTM runs over the 8x8x256 bottleneck features, each step FiLM-conditioned on the day
gap to the *next* frame (for the last frame, the target offset); the final hidden state gets Step C's
target-offset FiLM; the decoder uses skips from the **last frame only**. Sequences shorter than K are
left-padded and masked (verified: padded frames have no effect on the output). Same loss, pair set,
optimizer, augmentation (applied to every frame and the target) and validation-only model selection as
Step C; test evaluated exactly once per run. 14.1M parameters (Step C: 9.3M). History comes from the
post-scrub `input_filepaths`.

**Sequence-length pre-check (cleaned pairs).** Median 6 history frames (min 1, max 14); 74% of pairs
have >= 4 frames, 26% would need padding under K=4 (Field1: 51%; Field1 median is 3 frames, Field2 7).
**K=4 truncates 65% of pairs** (older history discarded; 3.5 of 6.8 available frames used on average).

| Run | Epochs run | Selected epoch | Best val loss | Val SSIM / PSNR (torch, Gaussian window) | Test structure SSIM / RGB SSIM / PSNR |
|---|---|---|---|---|---|
| Step C (single frame) | 73 | 61 | 0.9021 | 0.2734 / 14.66 | 0.0823 / 0.2540 / 14.84 |
| ConvLSTM K=1 (seed 42) | 75 | 63 | 0.8999 | 0.2748 / 14.65 | 0.0843 / 0.2554 / 14.85 |
| ConvLSTM K=4 (seed 42) | 77 | 65 | 0.8954 | 0.2773 / 14.69 | 0.0866 / 0.2580 / 14.87 |
| ConvLSTM K=4 (seed 43, confirmatory) | 62 | 50 | 0.8964 | 0.2765 / 14.75 | 0.0851 / 0.2571 / 14.94 |

All runs overfit mildly (train SSIM ~0.29-0.30 vs val ~0.275-0.277) and plateau on validation around
epochs 40-60. K=1 and K=4 took ~40 and ~88 minutes.

#### Headline table (test, mean (median)): structure SSIM against Step C, RGB SSIM / PSNR against Control 3

(K=4 = the seed-42 run. Per the protocol, both comparisons always appear together.)

| Subset | Pairs | structure SSIM: Step C | structure SSIM: K=1 | structure SSIM: K=4 | RGB SSIM: Control 3 | RGB SSIM: K=1 | RGB SSIM: K=4 | RGB PSNR dB: Control 3 | RGB PSNR dB: K=1 | RGB PSNR dB: K=4 |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 1,233 | 0.082 (0.054) | 0.084 (0.055) | 0.087 (0.057) | 0.217 (0.121) | 0.255 (0.161) | 0.258 (0.165) | 14.40 (12.04) | 14.85 (12.42) | 14.87 (12.41) |
| pooled_excl_f1_day28 | 1,159 | 0.080 (0.054) | 0.083 (0.054) | 0.085 (0.056) | 0.206 (0.117) | 0.244 (0.158) | 0.247 (0.160) | 14.06 (11.74) | 14.44 (12.17) | 14.45 (12.10) |
| Field1 | 215 | 0.071 (0.050) | 0.070 (0.048) | 0.071 (0.051) | 0.276 (0.231) | 0.304 (0.257) | 0.306 (0.261) | 16.97 (16.09) | 17.76 (16.73) | 17.84 (16.64) |
| Field1_excl_day28 | 141 | 0.050 (0.048) | 0.048 (0.045) | 0.049 (0.048) | 0.212 (0.231) | 0.238 (0.257) | 0.238 (0.261) | 15.54 (16.09) | 15.91 (16.73) | 15.94 (16.64) |
| Field1_day28_only | 74 | 0.111 (0.102) | 0.111 (0.105) | 0.114 (0.108) | 0.399 (0.385) | 0.430 (0.424) | 0.434 (0.425) | 19.70 (19.32) | 21.28 (21.17) | 21.46 (20.93) |
| Field1 target is day28 (bright->dark) | 37 | 0.184 (0.175) | 0.184 (0.178) | 0.184 (0.179) | 0.660 (0.648) | 0.703 (0.701) | 0.703 (0.701) | 25.81 (25.86) | 28.21 (28.18) | 28.20 (28.28) |
| Field1 input is day28 (dark->bright) | 37 | 0.037 (0.038) | 0.038 (0.043) | 0.044 (0.046) | 0.138 (0.149) | 0.157 (0.171) | 0.165 (0.179) | 13.59 (13.56) | 14.34 (14.54) | 14.73 (14.84) |
| Field2 | 1,018 | 0.085 (0.056) | 0.087 (0.058) | 0.090 (0.060) | 0.205 (0.110) | 0.245 (0.148) | 0.248 (0.152) | 13.85 (11.56) | 14.24 (11.86) | 14.24 (11.82) |

#### The three explicit comparisons (paired, mean (median) [95% plant-bootstrap CI of the mean], % of pairs where the first is better)

**(1) K=1 vs Step C: architecture alone.**

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | +0.002 (+0.001) [+0.001, +0.003], 56% | +0.001 (+0.001) [+0.001, +0.002], 59% | +0.01 (+0.01) [-0.01, +0.03], 51% |
| pooled_excl_f1_day28 | 1,159 | +0.002 (+0.002) [+0.002, +0.003], 57% | +0.002 (+0.001) [+0.001, +0.002], 60% | +0.02 (+0.01) [-0.00, +0.03], 52% |
| Field1 | 215 | -0.001 (-0.001) [-0.002, +0.000], 45% | +0.000 (+0.000) [-0.001, +0.001], 52% | -0.04 (-0.04) [-0.08, -0.01], 43% |
| Field1_excl_day28 | 141 | -0.002 (-0.001) [-0.003, -0.000], 42% | +0.000 (+0.000) [-0.001, +0.001], 52% | -0.02 (-0.01) [-0.06, +0.01], 48% |
| Field1_day28_only | 74 | +0.001 (+0.000) [-0.001, +0.002], 50% | -0.000 (+0.000) [-0.001, +0.001], 51% | -0.07 (-0.06) [-0.15, +0.00], 32% |
| Field1 target is day28 (bright->dark) | 37 | +0.000 (+0.001) [-0.001, +0.002], 59% | +0.000 (+0.000) [-0.000, +0.001], 65% | -0.01 (-0.04) [-0.06, +0.03], 38% |
| Field1 input is day28 (dark->bright) | 37 | +0.001 (-0.000) [-0.002, +0.003], 41% | -0.000 (-0.002) [-0.003, +0.003], 38% | -0.13 (-0.13) [-0.28, +0.02], 27% |
| Field2 | 1,018 | +0.003 (+0.002) [+0.002, +0.003], 59% | +0.002 (+0.001) [+0.001, +0.002], 61% | +0.02 (+0.01) [-0.00, +0.04], 52% |

**(2) K=4 vs K=1: history alone, same architecture.**

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | +0.002 (+0.002) [+0.002, +0.003], 61% | +0.003 (+0.002) [+0.002, +0.003], 61% | +0.02 (+0.01) [-0.00, +0.04], 52% |
| pooled_excl_f1_day28 | 1,159 | +0.002 (+0.002) [+0.002, +0.003], 61% | +0.002 (+0.002) [+0.002, +0.003], 61% | +0.01 (+0.01) [-0.01, +0.03], 51% |
| Field1 | 215 | +0.002 (+0.001) [+0.001, +0.003], 60% | +0.001 (+0.000) [+0.000, +0.002], 50% | +0.08 (+0.04) [+0.03, +0.13], 57% |
| Field1_excl_day28 | 141 | +0.001 (+0.001) [-0.000, +0.002], 57% | -0.000 (-0.000) [-0.001, +0.001], 45% | +0.03 (+0.02) [-0.01, +0.07], 52% |
| Field1_day28_only | 74 | +0.002 (+0.002) [+0.001, +0.004], 65% | +0.004 (+0.001) [+0.002, +0.006], 61% | +0.18 (+0.09) [+0.07, +0.31], 65% |
| Field1 target is day28 (bright->dark) | 37 | -0.000 (+0.001) [-0.003, +0.002], 59% | -0.000 (-0.001) [-0.001, +0.001], 43% | -0.02 (+0.00) [-0.11, +0.07], 51% |
| Field1 input is day28 (dark->bright) | 37 | +0.005 (+0.004) [+0.003, +0.008], 70% | +0.008 (+0.004) [+0.005, +0.011], 78% | +0.38 (+0.20) [+0.20, +0.61], 78% |
| Field2 | 1,018 | +0.003 (+0.002) [+0.002, +0.003], 62% | +0.003 (+0.002) [+0.002, +0.003], 63% | +0.01 (+0.00) [-0.02, +0.03], 51% |

**(3) K=4 vs Step C: combined (headline).**

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | +0.004 (+0.004) [+0.004, +0.005], 65% | +0.004 (+0.003) [+0.003, +0.005], 64% | +0.03 (+0.01) [+0.01, +0.05], 51% |
| pooled_excl_f1_day28 | 1,159 | +0.004 (+0.004) [+0.004, +0.005], 65% | +0.004 (+0.003) [+0.003, +0.005], 64% | +0.02 (+0.00) [-0.00, +0.05], 51% |
| Field1 | 215 | +0.001 (+0.001) [-0.000, +0.002], 55% | +0.001 (+0.000) [+0.000, +0.002], 52% | +0.04 (+0.00) [-0.01, +0.09], 51% |
| Field1_excl_day28 | 141 | -0.000 (-0.001) [-0.002, +0.001], 48% | -0.000 (-0.001) [-0.001, +0.001], 45% | +0.00 (-0.02) [-0.04, +0.05], 47% |
| Field1_day28_only | 74 | +0.003 (+0.003) [+0.001, +0.005], 69% | +0.004 (+0.002) [+0.002, +0.006], 65% | +0.11 (+0.03) [+0.01, +0.21], 59% |
| Field1 target is day28 (bright->dark) | 37 | -0.000 (+0.002) [-0.002, +0.002], 59% | +0.000 (-0.000) [-0.001, +0.002], 46% | -0.03 (+0.00) [-0.13, +0.05], 54% |
| Field1 input is day28 (dark->bright) | 37 | +0.006 (+0.006) [+0.003, +0.009], 78% | +0.007 (+0.005) [+0.004, +0.011], 84% | +0.25 (+0.11) [+0.09, +0.43], 65% |
| Field2 | 1,018 | +0.005 (+0.005) [+0.004, +0.006], 67% | +0.005 (+0.003) [+0.004, +0.005], 67% | +0.03 (+0.01) [-0.00, +0.05], 51% |

Pooled: structure SSIM 0.082 (Step C) -> 0.084 (K=1) -> 0.087 (K=4): +0.002 from the architecture alone,
+0.002 from history, +0.004 combined (about 5% relative).

#### K=4 minus K=1 by number of history frames the pair has

(1 frame = identical information for K=1 and K=4, so that row is a noise reference.)

| History frames | Pairs | d Structure SSIM mean (median) [95% CI], better | d RGB SSIM | d PSNR dB |
|---|---|---|---|---|
| 1 | 110 | +0.004 (+0.004) [+0.003, +0.006], 75% | +0.001 (+0.001) [-0.000, +0.002], 62% | +0.00 (+0.01) [-0.03, +0.03], 52% |
| 2 | 110 | +0.001 (+0.001) [+0.000, +0.002], 62% | +0.000 (+0.000) [-0.000, +0.001], 55% | +0.01 (-0.00) [-0.03, +0.04], 48% |
| 3 | 110 | +0.006 (+0.005) [+0.004, +0.007], 75% | +0.009 (+0.007) [+0.007, +0.010], 83% | +0.12 (+0.05) [+0.03, +0.22], 59% |
| 4-5 | 220 | +0.002 (+0.002) [+0.001, +0.003], 63% | +0.002 (+0.002) [+0.001, +0.003], 62% | +0.05 (+0.04) [+0.00, +0.09], 55% |
| 6+ | 683 | +0.002 (+0.001) [+0.001, +0.003], 57% | +0.002 (+0.002) [+0.002, +0.003], 58% | -0.00 (+0.00) [-0.03, +0.03], 50% |

No increase with more history (+0.004, +0.001, +0.006, +0.002, +0.002 structure SSIM for 1, 2, 3, 4-5, 6+
frames).

#### Confirmatory second seed (K=4, seed 43; same architecture and hyperparameters)

(In the tables below, "seed 2" means seed 43; "seed 42" is the original K=4 run. Output files for the
second seed carry a `_seed43` suffix.)

Seed-to-seed difference (K=4 seed 43 minus seed 42):

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | -0.002 (-0.002) [-0.002, -0.001], 42% | -0.001 (-0.001) [-0.001, -0.000], 45% | +0.07 (+0.05) [+0.05, +0.09], 58% |
| pooled_excl_f1_day28 | 1,159 | -0.002 (-0.002) [-0.002, -0.001], 42% | -0.001 (-0.001) [-0.001, -0.000], 46% | +0.08 (+0.06) [+0.05, +0.10], 60% |
| Field1 | 215 | +0.000 (-0.001) [-0.001, +0.001], 46% | -0.001 (-0.001) [-0.002, +0.000], 40% | -0.01 (-0.04) [-0.05, +0.02], 43% |
| Field1_excl_day28 | 141 | -0.000 (-0.001) [-0.002, +0.001], 44% | -0.001 (-0.001) [-0.002, +0.001], 45% | -0.02 (-0.03) [-0.06, +0.03], 45% |
| Field1_day28_only | 74 | +0.000 (+0.000) [-0.001, +0.002], 50% | -0.001 (-0.001) [-0.002, -0.001], 32% | -0.01 (-0.04) [-0.06, +0.05], 38% |
| Field1 target is day28 (bright->dark) | 37 | +0.002 (+0.001) [-0.000, +0.004], 57% | -0.000 (-0.000) [-0.001, +0.001], 46% | -0.05 (-0.05) [-0.12, +0.02], 30% |
| Field1 input is day28 (dark->bright) | 37 | -0.001 (-0.002) [-0.003, +0.001], 43% | -0.002 (-0.002) [-0.004, -0.001], 19% | +0.04 (-0.04) [-0.02, +0.11], 46% |
| Field2 | 1,018 | -0.002 (-0.002) [-0.003, -0.001], 41% | -0.001 (-0.001) [-0.002, -0.000], 46% | +0.09 (+0.07) [+0.07, +0.11], 62% |
| (pairs with 1 history frame) | 110 | -0.000 (+0.001) [-0.002, +0.002] | +0.001 (+0.000) [-0.001, +0.002] | +0.03 (+0.01) [-0.01, +0.06] |

**Is the seed-to-seed difference within the ~0.002-0.004 band seen in the noise-floor check? Yes.** Pooled
structure SSIM differs by **-0.0016** between the two K=4 runs (0.0851 vs 0.0866); RGB SSIM by -0.001;
PSNR by +0.07 dB.

The history effect (K=4 minus K=1) for each seed and their average:

| K=4 run | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|
| seed 42 | +0.002 [+0.002, +0.003] | +0.003 [+0.002, +0.003] | +0.02 [-0.00, +0.04] |
| seed 2 | +0.001 [+0.000, +0.001] | +0.002 [+0.001, +0.002] | +0.09 [+0.07, +0.11] |
| average of the two seeds | +0.002 [+0.001, +0.002] | +0.002 [+0.002, +0.003] | +0.06 [+0.04, +0.07] |

Protocol check for both K=4 runs:

| K=4 run | d Structure SSIM vs Step C | d RGB SSIM vs Control 3 | d RGB PSNR dB vs Control 3 |
|---|---|---|---|
| seed 42 | +0.004 [+0.004, +0.005] | +0.041 [+0.039, +0.042] | +0.47 [+0.42, +0.53] |
| seed 2 | +0.003 [+0.002, +0.003] | +0.040 [+0.038, +0.041] | +0.54 [+0.50, +0.60] |

#### What this does and does not establish

- **Under this K=4 ConvLSTM design, additional history provides no benefit that can be told apart from
  training-run noise.** The pooled history effect is +0.002 structure SSIM (seed 42), +0.001 (seed 43),
  +0.002 on average; the directly measured model-to-model difference is of the same size (-0.0016
  between two K=4 seeds) and larger when comparing different configurations on pairs with identical
  input (+0.004, K=4 vs K=1, 1-history-frame pairs). The bootstrap intervals above reflect sampling of
  test plants only, not training randomness, and two runs cannot estimate seed variance reliably; the
  honest summary is "same order as the noise", not a proven zero. The effect, if real, is at most ~0.002
  structure SSIM (~2% relative).
  *Update from the reproducibility pass:* the noise references in this bullet (a two-seed difference, 110-pair subsets) were the best available
  then. A better estimate, the within-configuration run-to-run spread of the pooled mean (per-run SD ~0.0006, so ~0.0008 for a difference
  between two single runs; Reproducibility section), puts the ConvLSTM's history effect at about 3x (seed 42, +0.0023) and 1x (seed 43, +0.0008)
  that noise: "indistinguishable from noise" holds for seed 43 and understates seed 42 modestly; the effect remains small (<= ~0.002, ~2% relative).
- **This characterizes THIS encoder / K = 4 design, not history in general.** K=4 discards older history for
  65% of pairs, half of Field1 is padded, and the history only enters through a recurrent state over
  bottleneck features. The CNN-Transformer, which conditions on the full available sequence through
  attention instead of a fixed truncated window, is a materially different and more direct test of whether
  history helps at all. (Result: see Step E below; its single-frame control found a larger history effect for the
  Transformer, concentrated in early-season Field2 flights.)
- **Untested hypothesis, flagged under multiple-comparisons caution: history may help when the last frame is
  uninformative.** On the 37 Field1 pairs whose *input* is the dark, low-contrast day-28 image, K=4 beats K=1
  by +0.005 structure SSIM (+0.008 RGB SSIM, +0.38 dB; 70-78% of pairs) and Step C by about the same; the
  second seed shows the same direction (+0.004 structure SSIM, +0.005 RGB SSIM, +0.42 dB vs K=1). The proposed
  mechanism is that earlier frames (days 15/22) are well exposed while the last frame is dark. This is a
  post-hoc subset of 37 pairs picked after examining ~120 subset-by-metric cells, not a finding, and it
  sits close to the noise reference; it should be tested prospectively.
- **No new shortcut (Control 3 protocol, restated).** Against Control 3, pooled test: K=4 structure SSIM
  lead +0.037 (Step C: +0.033; K=1: +0.035), RGB SSIM +0.041 / +0.040 (seeds 42 / 43; Step C +0.037),
  PSNR +0.47 / +0.54 dB (Step C +0.44 dB). The ConvLSTM's gains over Control 3 are essentially Step C's;
  structure SSIM against Step C is +0.004 / +0.003. Nothing here suggests a different shortcut.
- **Qualitative (illustration only).** `outputs/img_convlstm_k4_examples.png` shows the same six test pairs as
  Step C's figure; the K=4 predictions are still smooth colour blobs, visually near-identical to Step C's.

### Step E: CNN-Transformer over the full history, with a single-frame control

`src/img_transformer.py`, `src/train_img_transformer.py`, `sbatch scripts/train_img_transformer.slurm`
(`--max-history 1` / `MAXHIST=1` gives the single-frame control with the identical architecture). **One token per
frame** (global average of the shared Step C / ConvLSTM bottleneck, LayerNorm, projection to d_model = 128; not
spatial-patch tokens). **Time encoding per token: a fixed continuous sinusoidal encoding of (target_day - input_day)**, not
day_after_planting, plus a learned output token at offset 0. Standard 2-layer, 4-head Transformer encoder (dropout 0.1, one
untuned configuration, 9.7M parameters in total) over the **full available history, variable length (up to 14 frames; the
real maximum is 14, not 13), with a padding mask; no K truncation**. An attention-pooling step (output token as query) gives
weights over frames, and the decoder's bottleneck **and all five skip connections are the attention-weighted combination of
every frame's feature maps** (Step C / ConvLSTM take skips from the last frame only, the one design element that had to
differ). Frames are encoded only where real, under activation checkpointing. Same L1 + SSIM loss, pair set, optimizer,
augmentation, validation-only selection and single test evaluation as before. Masking verified at both extremes in a mixed
batch of 1-, 2- and 14-frame samples (weights sum to 1, exactly 0 on padding, a 1-frame sample is identical alone or
batched, extra padding changes nothing).

**Time-encoding aliasing check (before any training; `src/pe_aliasing_check.py`).** Real range of target_day - input_day
over all (pair, frame): 2-84 days, 60 distinct values. The Phase 0 setting (standard sinusoid, 64 dims, base 10000) does
**not** transfer: 77 of 83 anchor values have a non-monotone similarity profile (max rise 0.078), and it compresses every
distance into a narrow band (cos(84, 15) = 0.45, cos(84, 22) = 0.46, cos(84, 64) = 0.59). Used instead: a geometric frequency
set, 128 dims, wavelengths 16-1000 days: worst non-monotone rise 0.0009 and 2 ordering violations of 0.0007-0.0022 cosine;
mean cosine falls smoothly from 0.99 (1 day apart) to 0.06 (80 days apart).

| Run | Epochs run | Selected epoch | Best val loss | Val SSIM / PSNR | Test structure SSIM / RGB SSIM / PSNR | Time |
|---|---|---|---|---|---|---|
| Transformer K=1 (single frame, same architecture) | 68 | 56 | 0.9013 | 0.2741 / 14.63 | 0.0827 / 0.2540 / 14.81 | 66 min |
| Transformer (full history) | 62 | 50 | 0.8935 | 0.2784 / 14.77 | 0.0879 / 0.2594 / 14.96 | 3 h 18 min |

(Reference validation losses: Step C 0.9021, ConvLSTM K=1 0.8999, ConvLSTM K=4 0.8954 / 0.8964.) Both overfit mildly like earlier
stages (full-history train SSIM 0.299 vs val 0.278).

#### Full comparison, structure SSIM as the primary metric (test, mean (median))

| Subset | Pairs | Step C | ConvLSTM K=1 | ConvLSTM K=4 (s42) | ConvLSTM K=4 (s43) | Transformer K=1 | Transformer (full) |
|---|---|---|---|---|---|---|---|
| pooled | 1,233 | 0.082 (0.054) | 0.084 (0.055) | 0.087 (0.057) | 0.085 (0.057) | 0.083 (0.053) | 0.088 (0.056) |
| pooled_excl_f1_day28 | 1,159 | 0.080 (0.054) | 0.083 (0.054) | 0.085 (0.056) | 0.083 (0.056) | 0.081 (0.053) | 0.086 (0.055) |
| Field1 | 215 | 0.071 (0.050) | 0.070 (0.048) | 0.071 (0.051) | 0.071 (0.050) | 0.068 (0.045) | 0.070 (0.048) |
| Field1_excl_day28 | 141 | 0.050 (0.048) | 0.048 (0.045) | 0.049 (0.048) | 0.049 (0.047) | 0.046 (0.042) | 0.047 (0.045) |
| Field1_day28_only | 74 | 0.111 (0.102) | 0.111 (0.105) | 0.114 (0.108) | 0.114 (0.105) | 0.110 (0.104) | 0.114 (0.105) |
| Field1 target is day28 (bright->dark) | 37 | 0.184 (0.175) | 0.184 (0.178) | 0.184 (0.179) | 0.186 (0.180) | 0.183 (0.176) | 0.185 (0.177) |
| Field1 input is day28 (dark->bright) | 37 | 0.037 (0.038) | 0.038 (0.043) | 0.044 (0.046) | 0.043 (0.044) | 0.036 (0.043) | 0.043 (0.046) |
| Field2 | 1,018 | 0.085 (0.056) | 0.087 (0.058) | 0.090 (0.060) | 0.088 (0.059) | 0.086 (0.056) | 0.092 (0.060) |

#### Protocol table (pooled test): structure SSIM against Step C, RGB SSIM and PSNR against Control 3, in one table

| Model | Structure SSIM | d vs Step C | RGB SSIM | d vs Control 3 | PSNR dB | d vs Control 3 |
|---|---|---|---|---|---|---|
| Control 3 | 0.0494 |  | 0.2173 |  | 14.40 |  |
| Step C | 0.0823 |  | 0.2540 | +0.0367 | 14.84 | +0.44 |
| ConvLSTM K=1 | 0.0843 | +0.0020 | 0.2554 | +0.0381 | 14.85 | +0.45 |
| ConvLSTM K=4 (s42) | 0.0866 | +0.0044 | 0.2580 | +0.0407 | 14.87 | +0.47 |
| ConvLSTM K=4 (s43) | 0.0851 | +0.0028 | 0.2571 | +0.0397 | 14.94 | +0.54 |
| Transformer K=1 | 0.0827 | +0.0004 | 0.2540 | +0.0367 | 14.81 | +0.41 |
| Transformer (full) | 0.0879 | +0.0056 | 0.2594 | +0.0420 | 14.96 | +0.56 |

#### The three-way decomposition for the Transformer (paired, mean (median) [95% plant-bootstrap CI], % of pairs where the first is better)

**(1) Transformer K=1 vs Step C: architecture alone.**

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | +0.000 (-0.000) [-0.000, +0.001], 50% | +0.000 (+0.000) [-0.000, +0.000], 53% | -0.03 (-0.03) [-0.05, -0.01], 46% |
| pooled_excl_f1_day28 | 1,159 | +0.000 (+0.000) [-0.000, +0.001], 50% | +0.000 (+0.000) [-0.000, +0.001], 54% | -0.02 (-0.02) [-0.04, -0.01], 47% |
| Field1 | 215 | -0.003 (-0.002) [-0.004, -0.002], 33% | -0.001 (-0.001) [-0.002, -0.000], 40% | -0.11 (-0.10) [-0.14, -0.08], 32% |
| Field1_excl_day28 | 141 | -0.004 (-0.003) [-0.005, -0.002], 30% | -0.002 (-0.002) [-0.003, -0.000], 38% | -0.12 (-0.10) [-0.15, -0.09], 30% |
| Field1_day28_only | 74 | -0.001 (-0.001) [-0.002, -0.000], 41% | -0.001 (-0.000) [-0.002, +0.000], 46% | -0.10 (-0.08) [-0.19, -0.03], 36% |
| Field1 target is day28 (bright->dark) | 37 | -0.001 (-0.001) [-0.002, +0.000], 46% | +0.000 (+0.000) [-0.001, +0.001], 57% | -0.07 (-0.05) [-0.17, +0.01], 38% |
| Field1 input is day28 (dark->bright) | 37 | -0.001 (-0.002) [-0.003, +0.001], 35% | -0.002 (-0.002) [-0.004, +0.001], 35% | -0.13 (-0.18) [-0.27, +0.00], 35% |
| Field2 | 1,018 | +0.001 (+0.001) [+0.000, +0.002], 53% | +0.000 (+0.001) [-0.000, +0.001], 56% | -0.01 (-0.01) [-0.03, +0.01], 49% |

**(2) Transformer (full) vs Transformer K=1: history alone, same architecture.**

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | +0.005 (+0.005) [+0.005, +0.006], 68% | +0.005 (+0.003) [+0.005, +0.006], 71% | +0.15 (+0.09) [+0.13, +0.17], 66% |
| pooled_excl_f1_day28 | 1,159 | +0.005 (+0.005) [+0.005, +0.006], 67% | +0.005 (+0.004) [+0.005, +0.006], 71% | +0.14 (+0.10) [+0.12, +0.17], 66% |
| Field1 | 215 | +0.002 (+0.002) [+0.001, +0.003], 61% | +0.002 (+0.001) [+0.001, +0.003], 60% | +0.13 (+0.05) [+0.08, +0.18], 62% |
| Field1_excl_day28 | 141 | +0.001 (+0.001) [-0.000, +0.003], 53% | +0.000 (+0.000) [-0.001, +0.001], 55% | +0.08 (+0.06) [+0.05, +0.11], 64% |
| Field1_day28_only | 74 | +0.004 (+0.003) [+0.003, +0.006], 76% | +0.004 (+0.001) [+0.003, +0.007], 70% | +0.22 (+0.05) [+0.10, +0.35], 59% |
| Field1 target is day28 (bright->dark) | 37 | +0.002 (+0.002) [+0.001, +0.004], 76% | +0.000 (+0.000) [-0.001, +0.001], 54% | +0.04 (-0.01) [-0.04, +0.14], 49% |
| Field1 input is day28 (dark->bright) | 37 | +0.006 (+0.004) [+0.004, +0.010], 76% | +0.009 (+0.005) [+0.005, +0.013], 86% | +0.40 (+0.19) [+0.18, +0.66], 70% |
| Field2 | 1,018 | +0.006 (+0.005) [+0.005, +0.007], 69% | +0.006 (+0.004) [+0.006, +0.007], 74% | +0.15 (+0.10) [+0.13, +0.18], 67% |

**(3) Transformer (full) vs Step C: combined.**

| Subset | Pairs | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|---|
| pooled | 1,233 | +0.006 (+0.005) [+0.005, +0.006], 69% | +0.005 (+0.004) [+0.005, +0.006], 71% | +0.12 (+0.09) [+0.10, +0.14], 62% |
| pooled_excl_f1_day28 | 1,159 | +0.006 (+0.005) [+0.005, +0.006], 69% | +0.005 (+0.004) [+0.005, +0.006], 72% | +0.12 (+0.09) [+0.10, +0.14], 62% |
| Field1 | 215 | -0.001 (-0.000) [-0.002, +0.001], 48% | +0.000 (+0.000) [-0.001, +0.001], 50% | +0.02 (-0.03) [-0.03, +0.06], 44% |
| Field1_excl_day28 | 141 | -0.003 (-0.003) [-0.004, -0.001], 38% | -0.002 (-0.001) [-0.003, -0.000], 43% | -0.04 (-0.04) [-0.08, +0.01], 40% |
| Field1_day28_only | 74 | +0.003 (+0.002) [+0.002, +0.005], 69% | +0.004 (+0.001) [+0.002, +0.005], 64% | +0.12 (+0.02) [+0.03, +0.21], 51% |
| Field1 target is day28 (bright->dark) | 37 | +0.001 (+0.001) [+0.000, +0.003], 59% | +0.000 (+0.000) [+0.000, +0.001], 57% | -0.04 (-0.07) [-0.09, +0.01], 41% |
| Field1 input is day28 (dark->bright) | 37 | +0.005 (+0.005) [+0.003, +0.007], 78% | +0.007 (+0.004) [+0.004, +0.010], 70% | +0.27 (+0.11) [+0.09, +0.46], 62% |
| Field2 | 1,018 | +0.007 (+0.006) [+0.006, +0.008], 73% | +0.006 (+0.005) [+0.006, +0.007], 76% | +0.14 (+0.11) [+0.12, +0.17], 66% |

Pooled: the architecture alone changes nothing (+0.000 structure SSIM; slightly worse on Field1, -0.003); history adds +0.005; combined
+0.006 (about 7% relative to Step C). For comparison, the ConvLSTM's history effect (K=4 minus K=1) was +0.002 and +0.001 for the two
seeds.

#### Transformer minus Transformer K=1 by number of history frames

(The 1-frame row has identical information and identical architecture, so it is a training-noise reference: +0.003 structure SSIM on 110
pairs.)

| History frames | Pairs | d Structure SSIM | d RGB SSIM | d PSNR dB |
|---|---|---|---|---|
| 1 | 110 | +0.003 (+0.003) [+0.001, +0.005], 64% | +0.000 (+0.001) [-0.001, +0.001], 61% | +0.08 (+0.08) [+0.05, +0.12], 69% |
| 2 | 110 | +0.007 (+0.006) [+0.006, +0.009], 89% | +0.002 (+0.002) [+0.001, +0.002], 80% | +0.12 (+0.09) [+0.08, +0.17], 75% |
| 3 | 110 | +0.017 (+0.016) [+0.014, +0.019], 91% | +0.022 (+0.023) [+0.019, +0.025], 95% | +0.27 (+0.12) [+0.17, +0.39], 67% |
| 4-5 | 220 | +0.004 (+0.004) [+0.003, +0.005], 64% | +0.005 (+0.005) [+0.004, +0.007], 74% | +0.16 (+0.14) [+0.11, +0.20], 71% |
| 6-9 | 321 | +0.002 (+0.002) [+0.001, +0.003], 56% | +0.002 (+0.002) [+0.001, +0.003], 61% | +0.01 (+0.02) [-0.03, +0.06], 52% |
| 10-14 | 362 | +0.005 (+0.005) [+0.004, +0.006], 68% | +0.006 (+0.006) [+0.005, +0.007], 73% | +0.25 (+0.18) [+0.20, +0.31], 71% |

#### Early-season Field2 targets (days 16, 22; 146 test pairs) vs all other test pairs: structure SSIM differences, mean [95% CI]

| Comparison | Early Field2 (n=146) | All other (n=1087) |
|---|---|---|
| Transformer - Transformer K=1 | +0.0160 [+0.0143, +0.0178] | +0.0038 [+0.0031, +0.0045] |
| Transformer - ConvLSTM K=4 (s42) | +0.0108 [+0.0094, +0.0121] | -0.0001 [-0.0007, +0.0006] |
| Transformer - ConvLSTM K=4 (s43) | +0.0100 [+0.0086, +0.0117] | +0.0018 [+0.0012, +0.0024] |
| ConvLSTM K=4 (s42) - ConvLSTM K=1 | +0.0038 [+0.0027, +0.0049] | +0.0022 [+0.0016, +0.0027] |
| Transformer K=1 - Step C | -0.0030 [-0.0039, -0.0021] | +0.0008 [+0.0002, +0.0014] |

#### Attention over history (test)

| History frames | Pairs | Weight on last frame | Uniform 1/n | Weight on oldest frame | Effective frames used | Largest weight on last frame |
|---|---|---|---|---|---|---|
| 1 | 110 | 1.000 | 1.000 | 1.000 | 1.00 | 100% |
| 2 | 110 | 0.773 | 0.500 | 0.227 | 1.66 | 100% |
| 3 | 110 | 0.370 | 0.333 | 0.326 | 2.98 | 70% |
| 4-5 | 220 | 0.462 | 0.225 | 0.190 | 3.67 | 74% |
| 6-9 | 321 | 0.453 | 0.139 | 0.159 | 4.25 | 97% |
| 10-14 | 362 | 0.583 | 0.085 | 0.093 | 4.01 | 100% |

#### Follow-up diagnostics on the early-Field2 history gain (three explanations tested in sequence)

The Transformer's history gain is concentrated in the 146 early-Field2 test pairs (targets on days 16 and 22; Table E above). Three
explanations were tested with controls and diagnostics that need no training, in this order; the third result is the strongest evidence
in the image-prediction track that the Transformer uses genuine per-plant information.

**1. Generic-date image control (`src/generic_date_control.py`, `src/compare_generic_date.py`): ruled out.** The prediction is the pixel-wise
mean of the TRAINING plants' real images at the pair's (field, target day) (344 images for Field2 days 16 and 22; it uses nothing about the
test pair except its field and date; scored once on all 1,233 test pairs; three Field1 day-97 pairs use a 7-image mean and are flagged
`thin_group`, none are in this subset). Early Field2 subset, test, mean (median):

| Method | Structure SSIM | RGB SSIM | RGB PSNR dB |
|---|---|---|---|
| Copy-forward | 0.1459 (0.0791) | 0.390 (0.327) | 13.83 (14.12) |
| C1 blur | 0.1133 (0.0917) | 0.418 (0.412) | 14.02 (14.26) |
| C2 flight-colour | 0.1588 (0.0921) | 0.440 (0.393) | 20.82 (19.89) |
| C3 colour+blur | 0.1103 (0.0874) | 0.480 (0.469) | 22.34 (21.95) |
| Generic-date mean image | 0.1004 (0.0896) | 0.482 (0.473) | 22.35 (22.49) |
| Step C | 0.2167 (0.1489) | 0.547 (0.511) | 23.86 (23.16) |
| ConvLSTM K=1 | 0.2151 (0.1498) | 0.547 (0.514) | 23.91 (23.12) |
| ConvLSTM K=4 (s42) | 0.2189 (0.1568) | 0.552 (0.519) | 23.91 (23.24) |
| ConvLSTM K=4 (s43) | 0.2196 (0.1554) | 0.549 (0.511) | 23.97 (23.11) |
| Transformer K=1 | 0.2137 (0.1466) | 0.545 (0.510) | 23.80 (23.12) |
| Transformer (full) | 0.2297 (0.1573) | 0.560 (0.522) | 23.99 (23.49) |

Structure SSIM differences on the subset, mean (median) [95% plant-bootstrap CI], % first better:

| Comparison | mean (median) [95% plant-bootstrap CI], % first better |
|---|---|
| Generic-date minus Transformer K=1 | -0.1133 (-0.1076) [-0.1187, -0.1081], 0% |
| Transformer minus Transformer K=1 | +0.0160 (+0.0141) [+0.0143, +0.0178], 97% |
| Transformer minus Generic-date | +0.1292 (+0.1228) [+0.1236, +0.1353], 100% |
| Generic-date minus ConvLSTM K=4 (s42) | -0.1185 (-0.1153) [-0.1241, -0.1131], 0% |
| Generic-date minus Step C | -0.1163 (-0.1174) [-0.1216, -0.1110], 0% |
| Generic-date minus C1 (blur) | -0.0129 (-0.0120) [-0.0149, -0.0108], 16% |
| Generic-date minus C3 (colour+blur) | -0.0099 (-0.0098) [-0.0119, -0.0079], 25% |
| Generic-date minus copy-forward | -0.0455 (-0.0290) [-0.0515, -0.0401], 8% |
| SHARE of the (Transformer - Transformer K=1) gap reproduced by the generic-date control | -7.09  [-8.04, -6.30]  (generic-K1 = -0.1133; Transformer-K1 = +0.0160) |

The generic-date image scores 0.100 structure SSIM here (0.040 on all test pairs), below copy-forward (0.146) and far below the
single-frame Transformer (0.214); it reproduces none of the +0.016 history gap (share reproduced -7.1 [-8.0, -6.3]). Its RGB SSIM and
PSNR on this subset equal Control 3's (0.482 vs 0.480; 22.35 vs 22.34 dB): the colour and exposure of these dates is fully predictable from
the date alone, but the Transformer's extra structure SSIM sits on top of that.

**2. Oracle colour-match diagnostic (`src/oracle_colormatch_transformer.py`): exposure-only explanation ruled out.** DIAGNOSTIC ONLY, NOT
achievable at inference: each prediction of the full and single-frame Transformer is shifted/scaled per channel to its OWN TARGET's mean and
std (the Step B oracle transform), then re-scored. Integrity check: the recomputed original scores match the saved per-pair CSVs (max
difference 1.7e-4 and 2.4e-4). Early Field2 subset (n=146):

| Metric | Model | before colour-matching | after colour-matching (mean+std; PRIMARY) | after mean-only match (supplementary) |
|---|---|---|---|---|
| structure_ssim | Transformer K=1 | 0.2137 (0.1466) | 0.2021 (0.1349) | 0.2137 (0.1470) |
| structure_ssim | Transformer (full) | 0.2297 (0.1573) | 0.2179 (0.1448) | 0.2297 (0.1576) |
| ssim | Transformer K=1 | 0.545 (0.510) | 0.538 (0.498) | 0.545 (0.510) |
| ssim | Transformer (full) | 0.560 (0.522) | 0.554 (0.509) | 0.560 (0.522) |
| psnr | Transformer K=1 | 23.80 (23.12) | 23.07 (22.11) | 23.95 (23.25) |
| psnr | Transformer (full) | 23.99 (23.49) | 23.38 (22.47) | 24.13 (23.51) |

| Gap (Transformer full minus Transformer K=1) | before | after mean+std (PRIMARY) | share remaining | after mean-only (supplementary) | share remaining |
|---|---|---|---|---|---|
| structure_ssim | +0.0160 [+0.0143, +0.0178], 97% | +0.0159 [+0.0141, +0.0178], 93% | 0.99 [0.97, 1.01] | +0.0161 [+0.0143, +0.0179], 97% | 1.01 [1.00, 1.01] |
| ssim | +0.0156 [+0.0141, +0.0170], 96% | +0.0156 [+0.0141, +0.0172], 89% | 1.01 [0.97, 1.04] | +0.0154 [+0.0140, +0.0169], 96% | 0.99 [0.98, 1.00] |
| psnr | +0.187 [+0.130, +0.247], 77% | +0.304 [+0.255, +0.355], 91% | 1.62 [1.31, 2.13] | +0.184 [+0.141, +0.229], 84% | 0.98 [0.81, 1.23] |

The structure SSIM gap is +0.0160 before and +0.0159 after the mean+std match (share remaining 0.99 [0.97, 1.01]; 1.01 with a mean-only match),
and the same on the other 1,087 pairs (+0.0038 -> +0.0037). Exposure does not explain the concentration. (The mean+std match itself lowers both
models' structure SSIM, 0.2137 -> 0.2021 for K=1 and 0.2297 -> 0.2179 for the full model, because it stretches smooth predictions' contrast; the
mean-only match, which removes brightness offsets without that stretch, leaves structure SSIM unchanged and raises PSNR ~0.15 dB for both, so both
models' brightness offsets are small and equal here.) The PSNR gap grows to +0.304 dB after the mean+std match, which I read as that
stretching artifact, not exposure; with the mean-only match it is +0.184 (0.98 remaining).

**3. Swapped-history test (`src/swapped_history_test.py`): the gain depends on the plant's own earlier frames.** Inference only, saved checkpoints.
For the 146 pairs, every EARLIER input frame (day 1 for target-16 pairs; days 1 and 8 for target-22 pairs) is replaced by the frame of a DIFFERENT
plant of the same field at the same acquisition date (time offsets and conditioning unchanged); the last input frame and the target stay the
plant's own. Donors are other TEST plants (never seen in training), one donor per (plant, draw), 5 random draws; 0 of 146 pairs excluded (72 donor
candidates per plant). Integrity: original-history scores match the saved CSVs (max difference 1.2e-4 and 8.3e-5); K=1 reads only the last frame and
its scores are identical under the swap (difference exactly 0).

| Arm | Structure SSIM | RGB SSIM | PSNR |
|---|---|---|---|
| Transformer K=1 (single frame) | 0.2137 | 0.5447 | 23.804 |
| Transformer, original history | 0.2297 | 0.5602 | 23.991 |
| Transformer, SWAPPED history (mean over draws) | 0.1831 | 0.4984 | 22.429 |
| (supplementary) Transformer, earlier frames = copies of own last frame | 0.1865 | 0.5004 | 21.444 |

Gap (Transformer minus its own K=1 control) on the subset; mean [95% plant-bootstrap CI], % pairs better:

| Arm | Structure SSIM | RGB SSIM | PSNR dB | share of the original structure-SSIM gap |
|---|---|---|---|---|
| (a) original history | +0.0160 [+0.0143, +0.0178], 97% | +0.0156 [+0.0141, +0.0170], 96% | +0.187 [+0.130, +0.247], 77% | 1.00 [1.00, 1.00] |
| (b) swapped history, draw 1 | -0.0315 [-0.0348, -0.0283], 8% | -0.0475 [-0.0521, -0.0429], 11% | -1.382 [-1.545, -1.215], 10% | -1.97 [-2.32, -1.68] |
| (b) swapped history, draw 2 | -0.0312 [-0.0347, -0.0281], 3% | -0.0478 [-0.0525, -0.0434], 7% | -1.449 [-1.640, -1.262], 14% | -1.95 [-2.30, -1.66] |
| (b) swapped history, draw 3 | -0.0289 [-0.0317, -0.0260], 5% | -0.0434 [-0.0471, -0.0395], 8% | -1.286 [-1.451, -1.120], 14% | -1.81 [-2.11, -1.53] |
| (b) swapped history, draw 4 | -0.0304 [-0.0336, -0.0272], 6% | -0.0445 [-0.0489, -0.0402], 7% | -1.256 [-1.453, -1.065], 16% | -1.90 [-2.25, -1.60] |
| (b) swapped history, draw 5 | -0.0312 [-0.0338, -0.0287], 4% | -0.0482 [-0.0517, -0.0447], 6% | -1.500 [-1.661, -1.326], 14% | -1.95 [-2.26, -1.68] |
| (b) swapped history, mean over draws | -0.0306 [-0.0329, -0.0285], 1% | -0.0463 [-0.0490, -0.0435], 3% | -1.375 [-1.484, -1.260], 8% | -1.92 [-2.21, -1.66] |
| (c, supplementary) earlier frames = own last frame | -0.0272 [-0.0294, -0.0250], 9% | -0.0443 [-0.0469, -0.0418], 5% | -2.360 [-2.515, -2.207], 1% | -1.70 [-1.97, -1.45] |

Swapped minus original history, same pairs:

| Metric | swapped (mean over draws) - original |
|---|---|
| structure_ssim | -0.0466 [-0.0491, -0.0442], 0% |
| ssim | -0.0618 [-0.0647, -0.0590], 0% |
| psnr | -1.562 [-1.668, -1.457], 1% |

Plainly: with another plant's earlier frames the Transformer's gap over the single-frame control goes from **+0.0160 to -0.0306 structure SSIM**
[-0.0329, -0.0285] (the five draws agree, -0.029 to -0.032): the full-history model is then **worse than the single-frame model**, not merely no better,
losing 0.047 structure SSIM and 1.6 dB against its own original-history score. Replacing the earlier frames by copies of the plant's own last frame
(supplementary arm; offsets unchanged, but out of distribution) gives -0.0272. The mean attention weight on the last frame is identical under
the swap (0.530 vs 0.530), so attention follows offsets and history length, not frame content.

**What this establishes, stated explicitly.** This is the strongest evidence in the image-prediction track that the Transformer uses genuine
per-plant information. Three explanations of the early-Field2 history gain have now been tested and ruled out in sequence: a generic "plant at this
date" image (step 1), an exposure-only effect (step 2), and offset / date identification (step 3: offsets are identical in both manipulations, yet
the gain collapses). The per-plant-content explanation is what remains standing after those eliminations, not merely the last untested option. The
test also bears on the earlier history claims: the early-Field2 history benefit is plant-specific, not a date or offset shortcut.

**Fragility, stated plainly.** The model is not robust to inconsistent history: given another plant's frames at the right offsets it falls 0.031
structure SSIM below the single-frame model. Any use of the history benefit has to treat the history as load-bearing and fragile.

**What it does not establish.** (i) That the dependence is informative GROWTH content: the gap going negative shows the decoder's pooled maps are
corrupted by misaligned wrong-plant content, i.e. dependence, not that the information used is growth-relevant. Plant-specific growth cues and aligned
multi-frame averaging / early-season layout (plant position is stable between days 1 and 16) both remain possible and untested. (ii) That the
supplementary arm isolates offsets: it is out of distribution for the trained model. (iii) Anything outside the 146 pairs (below).

**Named follow-up (not just a caveat): repeat the swapped-history test beyond early-season Field2.** The swap test was run only on the 146-pair
early-Field2 subset. Whether the pooled +0.005 history effect elsewhere in the dataset (the 1,087 other test pairs, where the Transformer's history
effect is +0.0038, close to the +0.003 identical-input training-noise reference) shows the same plant-specific dependence is untested. It is the
natural next check if this finding needs to generalize beyond early-season Field2.

#### What the Transformer stage does and does not establish

- **Noise-floor framing, stated plainly: each adjacent step in the ordering is within or near measured training noise.** Pooled
  structure SSIM: Step C 0.0823, Transformer K=1 0.0827, ConvLSTM K=1 0.0843, ConvLSTM K=4 0.0851 / 0.0866 (two seeds), Transformer
  0.0879. The seed-to-seed difference between two identical K=4 runs was 0.0016 and the identical-input reference above is +0.003, so
  the individual steps (+0.0004, +0.002, +0.002, +0.001 to +0.003) are of that size. Only the cumulative separation is larger:
  the three runs that use history (ConvLSTM K=4 x2, Transformer) all sit above the three that do not (Step C, ConvLSTM K=1, Transformer
  K=1), by +0.0034 on average (0.0865 vs 0.0831, ~4% relative), although the closest pair differs by only 0.0008 and the runs are single
  seeds (two for K=4). Bootstrap intervals cover test-plant sampling only, not training randomness.
  *Update from the reproducibility pass:* this bullet was written against the looser references above. Against the within-configuration
  run-to-run noise estimate (a difference between two single runs ~0.0008), the Step C -> Transformer K=1 step (+0.0004) is within noise (0.5x); the
  other steps (+0.001 to +0.003) are about 1x to 4x it; and the cumulative history/no-history separation (+0.0034, a difference of three-run means,
  noise ~0.0005) is about 7x it.
- **The Transformer's own history effect (+0.005 structure SSIM, [+0.005, +0.006]) is the largest of the history effects measured, but it
  is not general.** It is larger than the run-to-run noise estimate for a difference between two single runs (about 0.0008, from within-configuration
  replicates, described in the Reproducibility section below; about 6x), but the earlier 110-pair identical-input reference is itself unstable across
  replicates (+0.0033 / +0.0043 originally, +0.0052 / +0.0028 in the fresh run) and should not be read as a reliable ceiling. For consistency, the
  ConvLSTM's history effect (+0.0023 for K=4 seed 42, +0.0008 for seed 43; fresh run +0.0020 / +0.0013) is about 3x and 1x the same estimate. Validation shows the same direction and a larger gap (val loss 0.9013 ->
  0.8935 vs 0.8999 -> 0.8954 for the ConvLSTM). By history length the effect is not monotone (+0.003, +0.007, +0.017, +0.004, +0.002,
  +0.005 for 1, 2, 3, 4-5, 6-9, 10-14 frames): it is concentrated in the 2- and 3-frame pairs, which are exactly the early-season Field2
  pairs below. Outside those, the Transformer's history effect (+0.0038) is about the identical-input reference (+0.003).
- **Early-season Field2 concentration: observed, and what has been ruled out.** On the 146 early-Field2 pairs (targets on days 16 and 22; 12% of test
  pairs) the Transformer beats its single-frame control by +0.016 [+0.014, +0.018] and the ConvLSTM K=4 by +0.011 / +0.010 (two seeds), versus +0.004 and
  -0.000 / +0.002 on the other 88%: they supply about 36% of the Transformer's pooled history effect. The ConvLSTM K=4 gains less there (+0.004 over K=1).
  Tested in sequence (see the follow-up diagnostics above): a generic-date image (ruled out), an exposure-only effect (ruled out) and offset / date
  identification (ruled out); the gain collapses when the plant's earlier frames are swapped for another plant's, so it depends on this plant's own history.
  Still untested: whether the information used is growth-relevant or aligned multi-frame averaging / early-season layout, and whether the pooled +0.005
  elsewhere shows the same dependence (the named follow-up). The concentration was first observed after looking at many subsets, which is why it was tested
  rather than assumed.
- **Attention is recency-dominant plus diffuse averaging, not selective retrieval.** For histories of 6+ frames the largest weight is on the
  last frame in 97-100% of pairs (weight 0.45-0.58 vs a uniform 0.09-0.14), while the oldest frame gets about its uniform share (0.159 vs
  0.139; 0.093 vs 0.085); the effective number of frames used is ~4 even for 10-14-frame histories. Caveat: only the first, last and
  maximum weights (and the effective number of frames) were saved, so the weights on intermediate frames, and hence any recency profile,
  are not available.
- **No new shortcut (Control 3 protocol, restated).** Against Control 3, pooled test: Transformer structure SSIM +0.038, RGB SSIM +0.042,
  PSNR +0.56 dB; the single-frame control +0.033, +0.037, +0.41 dB; Step C +0.033, +0.037, +0.44 dB. The Transformer's gains over Control 3
  are essentially Step C's; its structure SSIM against Step C is +0.006.
- **Overall conclusion.** At this data scale, more sophisticated temporal modelling yields small, mostly noise-scale gains: from Step C to the
  best model, structure SSIM rises from 0.082 to 0.088 (+0.006, ~7% relative). The gain from using history is real in direction (+0.002 for
  the ConvLSTM, +0.005 for the Transformer) but concentrated in specific flights (early-season Field2) rather than a general benefit of
  history; outside them the Transformer is indistinguishable from the ConvLSTM K=4 (-0.000 against seed 42, +0.002 against seed 43). All
  outputs remain smooth colour blobs (structure SSIM ~0.09, median ~0.056). The history benefit that does exist (early-season Field2) depends on the plant's own earlier frames (swapped-history
  test) and the model degrades below the single-frame baseline when history is inconsistent.
