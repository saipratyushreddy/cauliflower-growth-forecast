# Track 2: next-photo prediction, part 3 (sharpening ablation, memorisation check)

[Back to the project overview](../README.md)

### Sharpness ablation: do residual output, a perceptual loss or any-gap training sharpen the predictions? (negative result)

**Question.** The models output smooth colour fields. Starting from the cheap single-frame model (Step C), each candidate fix was added one at a time (`src/img_sharp.py`,
`src/train_img_sharp.py`, `scripts/sharp_ablation.sh`; one seed each, same cleaned pairs, plant-wise split, L1 + (1-SSIM) base loss, augmentation and validation-only
selection as Step C; test scored once; the any-gap runs also score the test plants' any-gap pairs):
- **residual:** the network predicts the change from the last photo (zero-initialised output head, so training starts exactly at copy-forward);
- **perceptual:** + 1.0 x L1 distance between pretrained VGG16 feature maps (relu1_2, relu2_2, relu3_3, relu4_3);
- **any-gap:** trained on any earlier photo -> any later photo of the same plant (39,477 pairs from the 5,823 standard ones, gaps 2-84 days; 5,823 sampled per epoch),
  selected on the standard validation task (last photo -> next photo).

**New metrics (they do not reward blur).** *Sharpness ratio* = prediction / real target of the mean Sobel gradient magnitude and of the variance of the Laplacian of the
luma, pooled over pairs (1 = as sharp as the real photo, < 1 = blurrier); *VGG distance* = summed L1 distance of VGG16 features (a perceptual distance in the spirit of
LPIPS, but not the LPIPS package). Both are for this project only; no external comparison is implied. The existing models were re-scored with the same metrics
(`src/sharpness_existing.py`, frozen checkpoints). Test, 1,233 pairs, pooled (d = difference to Step C for structure SSIM, to Control 3 for RGB SSIM / PSNR):

| Method | Structure SSIM (d vs Step C) | RGB SSIM (d vs Control 3) | PSNR dB (d vs Control 3) | VGG distance (lower better) | Sharpness: gradient ratio | Sharpness: Laplacian ratio |
|---|---|---|---|---|---|---|
| Copy-forward | 0.0442 (-0.0381) | 0.1221 (-0.095) | 10.04 (-4.36) | 4.781 | 0.948 | 0.9318 |
| Control 3 (colour+blur) | 0.0494 | 0.2173 | 14.40 | n/a | n/a | n/a |
| Step C (baseline) | 0.0823 | 0.2540 (+0.037) | 14.84 (+0.44) | 3.656 | 0.193 | 0.0048 |
| ConvLSTM K=4 | 0.0866 (+0.0044) | 0.2580 (+0.041) | 14.87 (+0.47) | 3.683 | 0.215 | 0.0055 |
| Transformer | 0.0879 (+0.0056) | 0.2594 (+0.042) | 14.96 (+0.56) | 3.664 | 0.202 | 0.0051 |
| + perceptual | 0.0546 (-0.0277) | 0.2239 (+0.007) | 14.70 (+0.30) | 3.145 | 0.040 | 0.0096 |
| + residual | 0.0819 (-0.0004) | 0.2530 (+0.036) | 14.88 (+0.48) | 3.642 | 0.191 | 0.0055 |
| + residual + perceptual | 0.0535 (-0.0288) | 0.2225 (+0.005) | 14.69 (+0.29) | 3.150 | 0.037 | 0.0055 |
| + any-gap | 0.0722 (-0.0101) | 0.2444 (+0.027) | 14.77 (+0.37) | 3.563 | 0.126 | 0.0023 |
| + residual + perceptual + any-gap | 0.0498 (-0.0325) | 0.2198 (+0.002) | 14.49 (+0.09) | 3.156 | 0.036 | 0.0056 |

**Paired difference to Step C** (variant minus Step C; mean [95% plant-bootstrap CI]; % of pairs where the variant is better):

| Variant | Structure SSIM | RGB SSIM | PSNR dB | VGG distance (negative = closer) |
|---|---|---|---|---|
| ConvLSTM K=4 | +0.0044 [+0.0038, +0.0050], 65% | +0.0040 [+0.0034, +0.0046], 65% | +0.030 [+0.006, +0.055], 51% | +0.027 [+0.025, +0.029], 23% |
| Transformer | +0.0056 [+0.0049, +0.0063], 69% | +0.0054 [+0.0048, +0.0060], 71% | +0.120 [+0.095, +0.143], 62% | +0.008 [+0.005, +0.011], 37% |
| + perceptual | -0.0277 [-0.0292, -0.0260], 14% | -0.0301 [-0.0317, -0.0284], 10% | -0.142 [-0.177, -0.107], 37% | -0.511 [-0.524, -0.497], 100% |
| + residual | -0.0004 [-0.0008, +0.0000], 47% | -0.0009 [-0.0013, -0.0005], 44% | +0.035 [+0.015, +0.053], 55% | -0.014 [-0.017, -0.011], 55% |
| + residual + perceptual | -0.0288 [-0.0304, -0.0271], 14% | -0.0315 [-0.0331, -0.0297], 9% | -0.151 [-0.187, -0.117], 39% | -0.506 [-0.519, -0.491], 100% |
| + any-gap | -0.0101 [-0.0109, -0.0094], 22% | -0.0096 [-0.0104, -0.0088], 20% | -0.077 [-0.108, -0.048], 43% | -0.093 [-0.097, -0.089], 84% |
| + residual + perceptual + any-gap | -0.0325 [-0.0343, -0.0305], 12% | -0.0342 [-0.0359, -0.0324], 8% | -0.350 [-0.393, -0.307], 35% | -0.500 [-0.513, -0.485], 100% |

**Findings.**
1. **The existing models are very blurry by these metrics:** about 19-22% of the real photo's edge strength and about 0.5% of its Laplacian variance, while copy-forward
   (which repeats a real photo) is at 0.95 / 0.93.
2. **The perceptual loss made the output flatter, not sharper:** edge ratio 0.19 -> 0.04, structure SSIM -0.0277 [-0.0292, -0.0260] (86% of pairs worse), RGB SSIM -0.030.
   The saved example images (`outputs/img_sharp_perc_examples.png`) are near-uniform green fields. Its VGG distance is the best of all (3.15 vs 3.66), so **VGG feature
   distance is gamed by a flat output and is not a sharpness measure here**; it was dropped as a headline metric.
3. **Residual output changed nothing:** structure SSIM -0.0004 [-0.0008, +0.0000], edge ratio 0.191 vs 0.193. The model starts at copy-forward but drifts to the same blur
   during training (its examples are blobs with some large-scale shape). Residual + perceptual is the same as perceptual alone.
4. **Any-gap training hurt the standard next-photo task** (structure SSIM -0.0101 [-0.0109, -0.0094], 78% of pairs worse) and did not sharpen. It gives, for the first time,
   accuracy as a function of the gap (test plants' any-gap pairs, 8,355 pairs; no copy-forward comparison at these gaps was computed, so whether these beat copy-forward is untested):

| Model | Gap (days) | Pairs | Structure SSIM | RGB SSIM | PSNR dB | VGG distance | Edge sharpness ratio |
|---|---|---|---|---|---|---|---|
| + any-gap | <=7 | 983 | 0.0619 | 0.2230 | 13.9179 | 3.6434 | 0.1322 |
| + any-gap | 8-14 | 1275 | 0.0500 | 0.1894 | 13.4245 | 3.6170 | 0.0707 |
| + any-gap | 15-28 | 2073 | 0.0332 | 0.1498 | 12.1237 | 3.7219 | 0.0504 |
| + any-gap | 29-56 | 2653 | 0.0249 | 0.1075 | 10.7597 | 3.7809 | 0.0320 |
| + any-gap | 57+ | 1371 | 0.0223 | 0.1025 | 10.6018 | 3.7576 | 0.0236 |
| + residual + perceptual + any-gap | <=7 | 983 | 0.0389 | 0.1938 | 13.6926 | 3.2074 | 0.0350 |
| + residual + perceptual + any-gap | 8-14 | 1275 | 0.0415 | 0.1762 | 13.2513 | 3.3138 | 0.0319 |
| + residual + perceptual + any-gap | 15-28 | 2073 | 0.0278 | 0.1403 | 12.0094 | 3.4778 | 0.0295 |
| + residual + perceptual + any-gap | 29-56 | 2653 | 0.0211 | 0.1014 | 10.8188 | 3.6158 | 0.0274 |
| + residual + perceptual + any-gap | 57+ | 1371 | 0.0210 | 0.0970 | 10.6722 | 3.6124 | 0.0253 |

5. **The same pattern holds excluding Field1 day-28 pairs and within each field** (all rows are generated by `src/compare_sharp.py`; the other tables are in its output).

**Why cheap loss changes cannot fix this (interpretation, not tested separately).** Even real consecutive photos differ in leaf layout (copy-forward scores only 0.044
structure SSIM against the real next photo; in the example images the input at day 51 and the target at day 83 show different leaf arrangements), so a loss that compares
the output pixel- or feature-wise with the one real future photo is minimised by an average, and a perceptual loss minimised by a flat feature average. Sharp output would need a
model that is rewarded for plausible texture rather than for matching the one real photo (adversarial or diffusion training). Such output would be plausible, not
accurate; structure SSIM would probably fall, so sharpness and accuracy would have to be reported separately. **Not done:** adversarial training, multiple seeds
(the differences above are 10x-35x the ~0.0008 run-to-run noise on a pooled mean, so single seeds do not change the conclusions), applying any variant to the Transformer.

### Memorisation check: frozen models on training vs unseen plants

Question: do the models predict plants they were trained on better than unseen plants? `src/memorization_check.py` (inference only, original checkpoints behind
the README numbers, no augmentation, same scoring as every table) scores the Transformer and ConvLSTM K=4 on all train / validation / test pairs, with
copy-forward on each split as a difficulty control (it uses no training). Pooled, mean (median):

| Model | Split | Pairs | Structure SSIM | RGB SSIM | PSNR dB |
|---|---|---|---|---|---|
| Copy-forward | train | 5,823 | 0.0434 (0.0277) | 0.1204 (0.0700) | 10.02 (9.05) |
| Copy-forward | val | 1,330 | 0.0455 (0.0293) | 0.1229 (0.0687) | 9.96 (8.95) |
| Copy-forward | test | 1,233 | 0.0442 (0.0277) | 0.1221 (0.0719) | 10.04 (9.06) |
| ConvLSTM K=4 | train | 5,823 | 0.1016 (0.0769) | 0.2745 (0.1870) | 15.47 (13.05) |
| ConvLSTM K=4 | val | 1,330 | 0.0873 (0.0575) | 0.2536 (0.1582) | 14.69 (12.25) |
| ConvLSTM K=4 | test | 1,233 | 0.0866 (0.0567) | 0.2580 (0.1652) | 14.87 (12.41) |
| Transformer | train | 5,823 | 0.0978 (0.0703) | 0.2687 (0.1773) | 15.35 (12.96) |
| Transformer | val | 1,330 | 0.0884 (0.0555) | 0.2547 (0.1571) | 14.77 (12.39) |
| Transformer | test | 1,233 | 0.0879 (0.0560) | 0.2594 (0.1643) | 14.96 (12.59) |

Copy-forward scores the same on all three splits (structure SSIM 0.043 / 0.046 / 0.044), so the splits are equally hard and the train-test difference is not
split difficulty. Gain over copy-forward on the same pairs, with the train-minus-test difference [95% CI; plants resampled independently per group]:

| Model | Metric | gain on train | gain on test | train minus test |
|---|---|---|---|---|
| ConvLSTM K=4 | structure_ssim | +0.0583 | +0.0425 | +0.0158 [+0.0140, +0.0175] |
| ConvLSTM K=4 | ssim | +0.1541 | +0.1359 | +0.0182 [+0.0117, +0.0237] |
| ConvLSTM K=4 | psnr | +5.45 | +4.84 | +0.61 [+0.47, +0.74] |
| Transformer | structure_ssim | +0.0544 | +0.0437 | +0.0107 [+0.0090, +0.0123] |
| Transformer | ssim | +0.1483 | +0.1373 | +0.0110 [+0.0048, +0.0165] |
| Transformer | psnr | +5.32 | +4.93 | +0.40 [+0.26, +0.52] |

**Reading.** Both models do better on plants they were trained on, so there is some memorisation: the train-minus-test difference of the structure SSIM gain is
+0.0158 [+0.0140, +0.0175] (ConvLSTM K=4) and +0.0107 [+0.0090, +0.0123] (Transformer). But about 73% (ConvLSTM) and 80% (Transformer) of the training
structure gain is retained on unseen plants, and validation matches test (0.087 / 0.087 and 0.088 / 0.088), so the model selection did not leak into the test
split. The same pattern holds excluding Field1 day-28 pairs and in each field separately. The Transformer memorises less than the ConvLSTM (14.1M vs 9.7M
parameters; that this is the reason was not tested). Caveats: training plants and test plants are different plants, so plant-level difficulty cannot be fully
separated from memorisation (the copy-forward control argues against it, a retraining-with-held-out-plants test would settle it); one run per model; the
intervals cover plant sampling, not training randomness (run-to-run noise on a pooled mean is about 0.0008).
