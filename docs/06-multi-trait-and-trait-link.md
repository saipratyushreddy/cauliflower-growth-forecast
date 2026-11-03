# Next directions: multi-trait forecasting and the image-to-trait link

Branch `next-directions`. This file was written **before any real run**; results are added below each design when the jobs finish.

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

**Results:** pending.

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

**Results:** pending.

## Not in scope here

Held-out-plot evaluation, the alignment diagnostic, growth-stage forecasting, and adversarial or diffusion image generation.
