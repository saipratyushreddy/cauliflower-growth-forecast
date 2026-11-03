# Next directions: multi-trait forecasting and the image-to-trait link

Branch `next-directions`. The designs were written **before any real run**; the results were added below each design afterwards.

## Why these two

- **Multi-trait forecasting** is the direction a grower would use: head size and harvest timing matter more than an image similarity score. It reuses the diameter
  pipeline and fixes its main limit: the diameter model never saw an earlier manual measurement, so a simple measurement-based rule beat it (4.74 vs 5.13).
- **The image-to-trait link** asks whether the next-photo track has practical value. Higher structure SSIM has never been shown to mean better trait estimates
  (README caveat). If a trait read from the generated photo is no better than one read from the last real photo plus average growth, the image track is a demonstration only.

## Data available (from `src/build_traits.py`, `data/traits.parquet`)

| Trait | Records | Plants | Where | Used for |
|---|---|---|---|---|
| Plant diameter | 7,769 | 738 | both fields, whole season | forecast (reference, already studied) |
| Head diameter | 3,392 | 689 | late season (Field2 days 44 to 85, Field1 days 83 to 93) | forecast |
| Height | 1,969 | 271 | 239 Field1 plants, 32 Field2 plants | forecast |
| Growth stage (BBCH) | 2,768 | 739 | Field2 never passes stage 19; Field1 reaches 33 to 49 on few dates | **not forecast** (too little progression to forecast) |
| Harvest date | first harvest mark | 470 Field2, 108 Field1 | Field2 days 65 to 85 (median 76); Field1 almost all day 91 | forecast in **Field2 only** (Field1 has almost no variation) |

Units are the dataset's own and are not stated anywhere. A side observation: head diameter runs 0 to 20 and height 2 to 87, which reads naturally as centimetres;
if so, plant diameter (1 to 118) is likely centimetres as well. This is circumstantial and does not verify the unit.

## Direction 2: multi-trait forecasting with prediction ranges (`src/trait_forecast.py`, `scripts/trait_forecast.slurm`)

**Split and inputs.** The frozen plant-wise split (Phase 0 assignments, the 739th plant in train). Black/placeholder photos are removed from the inputs.
Inputs are the frozen ResNet18 embeddings used by the diameter models.

**Tasks.**
- *Next measurement* (diameter, head diameter, height): predict the trait at the next acquisition date from all earlier photos.
- *Harvest timing* (Field2): from the photos up to a cutoff date (day 44, 50, 57 or 65, all before the first harvest), predict the days until the plant's first
  harvest mark. Plants already harvested by the cutoff, and the 30 Field2 plants never marked harvested, are excluded.

**Two arms, never mixed.**
- *Image arm*: photos and photo days (plus the forecast gap for next-measurement tasks). No manual measurement.
- *Measurement arm*: the image arm plus every earlier manual measurement of the trait (harvest: of head diameter and diameter), with a missing flag.

**Methods.**
- Baselines. Image arm: age-only (train mean per field and target day; harvest: train mean per cutoff). Measurement arm: persistence and last measurement + mean train
  growth for that step (harvest: a linear model on the last head diameter and diameter, per cutoff).
- Ridge on the last photo's features (+ last measurement in the measurement arm); alpha chosen on validation.
- CNN-LSTM (the diameter architecture with the extra inputs). Configuration chosen on validation MAE from hidden {64, 128, 256} x layers {1, 2} (dropout 0.2,
  lr 1e-3, seed 42); the chosen configuration is trained with seeds 42, 43, 44 and its test predictions averaged.

**Prediction ranges.** Split-conformal: the interval is prediction +/- the ceil((n+1)(1-a))-th smallest validation absolute residual, for 80% and 90% ranges.
Reported: test coverage and mean width, overall and by field. Pairs of one plant are correlated, so coverage is approximate.

**Evaluation.** Test pairs where every method is defined (next measurement: the last input date has a measurement of the trait). MAE with plant-bootstrap 95%
intervals; paired difference of each learned method against its own arm's best simple baseline (image arm vs age-only, measurement arm vs last + growth).

| Task | Test evaluation set |
|---|---|
| Diameter | 1,049 pairs, 110 plants (210 Field1, 839 Field2) |
| Head diameter | 409 pairs, 73 plants (all Field2) |
| Height | 270 pairs, 43 plants (210 Field1, 60 Field2) |
| Harvest timing | 280 pairs, 70 plants (Field2) |

The diameter set differs slightly from Phase 0's 1,057 because black photos are removed from inputs and the pairs are rebuilt from the trait table.

**Baseline results already available** (no embeddings needed, from the smoke run; final numbers come from the full run):
head diameter, last + growth MAE 1.12, age-only 1.74, persistence 2.15; harvest timing, per-cutoff mean 3.38 days, last-measurement linear model 2.80 days.

**Questions answered.**
1. With a measurement available, do the photos improve on last + growth? (the question the diameter work could not answer)
2. Without one, do photos beat age alone for head size, height and harvest timing?
3. Are the prediction ranges calibrated on new plants, and how wide are they?

**Results** (`outputs/trait_forecast_*.csv`; job 18783932; test MAE in the dataset's units, harvest in days; LSTM = mean of 3 seeds):

| Task (test pairs) | Age-only | **Image arm** ridge | **Image arm** LSTM | Persistence | Last + growth | **Meas. arm** ridge | **Meas. arm** LSTM |
|---|---|---|---|---|---|---|---|
| Diameter (1,049) | 6.57 | 5.70 | 5.09 | 9.05 | 4.70 | 4.75 | 4.60 |
| Head diameter (409) | 1.74 | 1.93 | 1.45 | 2.15 | 1.12 | 1.27 | 1.20 |
| Height (270) | 5.00 | 5.16 | 4.47 | 9.27 | 3.54 | 4.41 | 4.26 |
| Harvest, days (280) | 3.38 | 3.25 | 2.50 | | 2.80 (linear on last measurements) | 2.85 | 2.57 |

Paired differences (learned method minus its own arm's baseline, test MAE; negative = better; plant-bootstrap 95% interval):

| Task | Image LSTM vs age-only | Meas. LSTM vs last + growth |
|---|---|---|
| Diameter | -1.48 [-1.96, -1.04] | -0.10 [-0.43, +0.25] |
| Head diameter | -0.29 [-0.52, -0.06] | +0.08 [-0.08, +0.28] |
| Height | -0.53 [-1.00, -0.08] (Field1 -0.71; Field2, 6 plants, +0.09) | +0.71 [-0.40, +2.48] |
| Harvest (days) | -0.88 [-1.40, -0.39] | -0.22 [-0.62, +0.20] (vs the linear model) |

**Reading.**
1. *With a measurement available, the photos add nothing measurable.* For every trait the measurement-arm LSTM is within noise of last measurement + mean growth. This answers the question the diameter work left open: the 4.74 baseline was not beaten because, given the last measurement, the photos carry little extra information for the next measurement, not because the model lacked the measurement.
2. *Without a measurement, photos beat age alone for every trait* (LSTM; the single-photo ridge does not, for head diameter and height). The gain is largest for diameter and harvest timing.
3. *Harvest timing from photos alone* is about 2.5 days off (age-only 3.4), numerically better than the linear model on the last head diameter and diameter (2.8; paired -0.30 [-0.73, +0.13], not significant). Error is similar at every cutoff from day 44 to 65.
4. The ridge on the last photo is weak; the gains come from the photo history (or the recurrent model; this does not separate them).

**Prediction ranges** (split-conformal from validation residuals; nominal 80% / 90%):

| Task | LSTM image arm: coverage 80 / 90, width 80 | LSTM meas. arm: coverage 80 / 90, width 80 |
|---|---|---|
| Diameter | 0.79 / 0.90, 14.8 | 0.81 / 0.89, 14.0 |
| Head diameter | 0.74 / 0.84, 3.6 | 0.76 / 0.85, 3.3 |
| Height | 0.73 / 0.88, 9.1 | 0.76 / 0.86, 10.2 |
| Harvest (days) | 0.69 / 0.91, 7.0 | 0.68 / 0.92, 6.9 |

Coverage is close to nominal for diameter and 5 to 11 points short at 80% for the smaller tasks. Likely causes: the same validation pairs chose the configuration (so their residuals are optimistic; e.g. head diameter image arm, validation MAE 1.04 vs test 1.45) and validation pairs of one plant are correlated. A separate calibration split, or plant-grouped calibration, would be needed for reliable ranges.

**Not done:** multiple test-time seeds for the baselines (deterministic), growth-stage forecasting, horizon-specific ranges, held-out plots.

## Direction 3: does a generated photo carry trait information? (`src/trait_from_generated.py`, `scripts/trait_link.slurm`)

**Trait readers.** Ridge on the frozen ResNet18 features of one photo, predicting that same day's measured diameter or head diameter; fitted on train-plant photos,
alpha on validation-plant photos. All photos go through the same path (256 x 256 cache, centre crop 224, ImageNet normalisation).
- *Sharp reader*: fitted on real photos.
- *Blur reader*: fitted on real photos blurred with a Gaussian whose strength is chosen on train photos to match the models' sharpness (gradient ratio of the
  models' test outputs, about 0.2). This removes the mismatch between sharp training photos and blurry generated ones.

**Conditions** (test image-track pairs whose target date has a measurement):

| Condition | What it tells |
|---|---|
| Real target photo | ceiling: how well one photo shows that day's trait |
| Last real photo | photo-space persistence |
| Last real photo + growth | the reading of the last photo + mean train growth for that step; **the reference to beat** |
| Blurred last photo | how much of any difference is blur alone |
| Single-photo, ConvLSTM K=4, Transformer outputs | the generated next photo |

**Decision rule.** A generated photo carries forward-looking trait information only if, with the same reader, it beats "last real photo + growth" (paired MAE
difference with a plant-bootstrap interval excluding zero). The dedicated diameter forecasts give a second reference: if reading the generated photo is much
worse than forecasting the number directly, the image route adds nothing for that trait.

**Results** (`outputs/trait_link_*.csv`; job 18783933). The models' outputs have a gradient sharpness ratio of 0.19 to 0.22; a Gaussian blur of sigma 6 matches it (0.20). Test MAE of the trait read from each photo:

| Trait (test pairs) | Real target photo (ceiling) | Last real photo + growth | Single-photo output | ConvLSTM output | Transformer output |
|---|---|---|---|---|---|
| Diameter, sharp reader (1,049) | 5.79 | 5.92 | 13.11 | 12.90 | 12.66 |
| Diameter, blur reader | | 6.28 (blurred last photo + growth) | 12.86 | 12.08 | 12.56 |
| Head diameter, sharp reader (505) | 2.06 | 2.26 | 4.44 | 4.39 | 4.44 |
| Head diameter, blur reader | | 2.37 (blurred last photo + growth) | 3.84 | 3.88 | 3.65 |

Paired differences of each generated photo against the reference, all pairs: diameter +6.3 to +7.2 (sharp reader) and +5.8 to +6.6 (blur reader); head diameter
+2.1 to +2.2 and +1.3 to +1.5; every interval excludes zero, in both fields, except head diameter with the blur reader in Field1 (26 pairs; -0.25 to -0.42,
intervals crossing or touching zero).

**Correction to the design, made after seeing the results.** The design compared the blur reader with "last real photo + growth", which feeds a sharp photo to a
reader fitted on blurred photos and so handicaps the reference. The fair reference for the blur reader is "blurred last photo + growth", computed from the saved
per-pair predictions (same growth step). The conclusion is the same under either reference for the sharp reader, and holds with the fair reference for the blur reader.

**Reading.**
1. *The generated photos carry less trait information than the last real photo plus average growth*, by a wide margin, even when the reader is fitted on equally
   blurry photos. The image models' gain in structure SSIM does not translate into better trait estimates.
2. *There is little room for any image forecast here.* Reading the real future photo (5.79) is barely better than reading the last real photo and adding average
   growth (5.92) for diameter, and 2.06 vs 2.26 for head diameter. Even a perfect next-photo predictor would improve these single-photo trait readings only slightly.
3. Reading a trait from a generated photo (12.7 for diameter) is far worse than forecasting the number directly (5.09 image-only, 4.60 with measurements).

Caveats: the readers are linear on frozen ImageNet features; generated photos may also differ from real ones in colour and texture, not only blur; one reader
family was tried.

## Not in scope here

Held-out-plot evaluation, the alignment diagnostic, growth-stage forecasting, and adversarial or diffusion image generation.

## Held-out plots: do the trait forecasts work in a plot the model never saw? (design fixed before running)

**Why.** Every result above uses a plant-wise split: test plants grow in the same 9 plots as training plants, while the dataset paper recommends spatially
disjoint splits. The prediction ranges were also too narrow on the smaller tasks because validation plants both chose the model and calibrated the range.

**Design** (`src/trait_forecast_lopo.py`, `scripts/trait_forecast_lopo.slurm`).
- *Leave one plot out.* Each of the 9 plots (Field1: 4, Field2: 5) is the test plot once. Tasks are run only where their test field has data: diameter in all 9
  folds, head diameter and harvest timing in the 5 Field2 folds, height in the 4 Field1 folds (Field2 has height for only 32 plants).
- *Calibration plot.* In each fold, the next plot of the same field (cyclic order) is held out for calibrating the prediction ranges only.
- *Training and early stopping.* All other plots of both fields; 15% of their plants (fixed seed) are used only for epoch selection and the ridge alpha.
- *Models.* The same baselines, ridge and CNN-LSTM as above, in both arms. The LSTM configuration is fixed to the one chosen per task and arm in the plant-wise run
  (no new search); seeds 42, 43, 44, predictions averaged.
- *Ranges.* Split-conformal from the calibration plot's absolute residuals (80% and 90%), evaluated on the test plot.
- *Reported.* Pooled MAE over all folds with plant-bootstrap intervals; MAE per test plot (spread); coverage and width; and a like-for-like comparison on the 110
  plant-wise test plants: their plant-split prediction (above) vs their held-out-plot prediction, same pairs.

**Known limits.** The fixed LSTM configurations were chosen on plant-wise validation plants, some of which fall in a fold's test plot (small leakage of
configuration choice). Neighbouring plots share a field, a season and flights, so this is a stronger test than the plant split but not a new site or year.
