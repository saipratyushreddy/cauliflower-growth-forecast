# Original introduction and status log (kept for the record)

[Back to the project overview](../README.md)

# Cauliflower Growth Forecasting (GrowliFlower)

The project goal, as scoped by the course/research prompt, is to compare
**three** approaches for forecasting cauliflower growth traits from UAV
image time series: (1) simple baselines, (2) a CNN-LSTM, and (3) a
CNN-Transformer, against real data on UNL's Swan HPC cluster.

**Phase 1** (baselines + CNN-LSTM) is complete, sweep-tuned, and
rigorously evaluated. **Phase 2** is in progress: the CNN-Transformer has
been implemented, swept, and compared (see
[Phase 2 results](01-diameter-forecasting.md#phase-2-results--cnn-transformer) below), and a
missing-observation robustness test comparing both frozen models has
also been completed (see
[Phase 2 results — missing-observation robustness](01-diameter-forecasting.md#phase-2-results--missing-observation-robustness)).
Multi-trait regression is still planned, not started (see
[Phase 2 — remaining work](01-diameter-forecasting.md#phase-2--remaining-work)).


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
[Phase 2 — remaining work](01-diameter-forecasting.md#phase-2--remaining-work) above).
Repository is public: https://github.com/saipratyushreddy/cauliflower-growth-forecast
