# Track 2: next-photo prediction, part 1 (data, baselines, oracle, single-photo model)

[Back to the project overview](../README.md)

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
