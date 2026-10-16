# Reproducibility pass (image track)

[Back to the project overview](../README.md)

## Reproducibility pass (image-to-image track)

**Procedure (from a fresh clone, 2026-10-08).** Step A verified the underlying data before reuse; Step B re-ran every training run, control and
diagnostic of the image track from a fresh clone (commit `b02f299`) as parallel SLURM jobs with the original seeds (default 42; ConvLSTM K=4 second
run 43), pointing at the existing, just-verified images and pair data (shared read-only; the Python environment was also shared, so library versions
are identical by construction); Step C compared fresh against original at full precision and against the README. Diagnostics were run from the
*freshly trained* checkpoints. Tooling: `src/repro_verify_data.py`, `scripts/repro/`, `src/repro_compare.py`, `src/repro_claims.py`.

**Step A: 21 of 21 checks passed.** Raw images 9,377 (exact set equality with the metadata); the 234 black frames recomputed from file size and md5,
identical file by file (232 share md5 `1fe4b7f0...`, plus `Plot2_A9-8` and `Plot2_C9-8`); the 9 blurry frames identical by name and by content
(recomputed variance of Laplacian equal to the stored scan, difference 0.0); the pair table rebuilt with the fresh code equal to
`data/image_pairs.parquet` in every column and row (5,823 / 1,330 / 1,233 pairs; Field1/Field2 table; 234 black-frame pairs removed as 178/16/40;
18 blurry pairs removed as 14/0/4; 25 scrubbed); the Phase 0 split unchanged for the 738 plants; `images256.npy` rebuilt from the raw images
byte-identical (0 of 9,134 rows differ). Environment: Python 3.9.4, torch 2.8.0+cu128, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-image 0.24.0,
Pillow 11.3.0 (`requirements.txt` is not pinned; the full `pip freeze` and SHA-256 fingerprints of the data files are in `repro_step_a.json` of that run).

**Closed-form items reproduce bit-for-bit:** copy-forward, Controls 1, 2 and 3, the generic-date control and the Control 1 validation sigma curve are
identical (maximum per-pair difference 0), as are the Step B baseline and oracle tables.

**Trained models do not reproduce bit-for-bit, and diverge from epoch 1, despite identical seed, code and (for three of them) GPU model.**
For all six trained runs (Step C, ConvLSTM K=1, K=4 seed 42, K=4 seed 43, Transformer, Transformer K=1) the validation loss already differs by more
than 1e-6 at epoch 1; epochs run and the selected epoch differ (fresh vs original: Step C 64 / 52 vs 73 / 61; ConvLSTM K=1 68 / 56 vs 75 / 63; K=4
seed 42 71 / 59 vs 77 / 65; K=4 seed 43 69 / 57 vs 62 / 50; Transformer 67 / 55 vs 62 / 50; Transformer K=1 68 / 56 vs 68 / 56). The model and
training code files are unchanged since the original runs (only an output-tag option and `--max-history` were added later, with no change to default
behaviour). GPU models, original vs fresh: Step C L40S / L40S; ConvLSTM K=1, K=4 x2 L40S / A30; Transformer and Transformer K=1 A30 / A30.

| Pooled test structure SSIM | Original | Fresh | Fresh - original |
|---|---|---|---|
| Step C | 0.08230 | 0.08282 | +0.00052 |
| ConvLSTM K=1 | 0.08429 | 0.08397 | -0.00032 |
| ConvLSTM K=4 seed 42 | 0.08665 | 0.08602 | -0.00063 |
| ConvLSTM K=4 seed 43 | 0.08510 | 0.08525 | +0.00016 |
| Transformer K=1 | 0.08265 | 0.08271 | +0.00005 |
| Transformer (full) | 0.08788 | 0.08746 | -0.00042 |

**Primary noise reference: the within-configuration run-to-run spread of the pooled mean.** Pooling all replicate runs (four runs of Step C: the
original, the fresh run and two further same-environment reruns, see below; two runs each of the other five configurations), the per-run SD of the
pooled test structure SSIM is **0.0006** (8 degrees of freedom; 95% CI 0.0004-0.0011; Step C alone, four runs, 0.0009, maximum pairwise
difference 0.0021). The noise SD of a *difference between two single runs* is therefore about **0.0008** (CI 0.0006-0.0016), and of a difference
between two three-run means about 0.0005. This supersedes (i) an earlier estimate of 0.0004 (the RMS of the six original-vs-fresh differences),
which understated the spread (the two further Step C reruns differ from each other by up to 0.0021), and (ii) the earlier references (two-seed
difference 0.0016 / 0.0008 in the fresh run; 110-pair identical-input subsets +0.0033 / +0.0043 originally and +0.0052 / +0.0028 fresh), which are
consistent with this spread but unstable. It assumes similar run-to-run noise across configurations and rests on few degrees of freedom. Subset-level
quantities are noisier: gaps on the 146 early-Field2 pairs and the swapped-history arms move by ~0.001-0.002 between replicates (up to 0.004 in the
worst case, the copy-last-frame arm); individual pairs differ by up to 0.02-0.05 structure SSIM; win rates, medians and confidence-interval endpoints
change in the last digit or two (and win rates on 110-pair buckets by tens of percentage points when the underlying difference is tiny). All signs
and all CI-versus-zero statuses replicate.

**Cause of the non-reproducibility: intrinsic run-to-run training nondeterminism, isolated by a same-environment rerun test.** Two further Step C
runs were launched with identical settings (seed 42, default hyperparameters) from the same fresh clone, venv and data, pinned to an L40S GPU
(the GPU model of the original and the fresh Step C run), writing to separate folders (`scripts/repro/nondeterminism_stepc.slurm`,
`src/nondeterminism_compare.py`; decision rule fixed before the reruns). The runs of one configuration (original, fresh, nd1, nd2):

| Run | Epochs run | Selected epoch | Best val loss | Test structure SSIM | Test RGB SSIM | Test PSNR |
|---|---|---|---|---|---|---|
| original | 73 | 61 | 0.902081 | 0.08230 | 0.25398 | 14.842 |
| fresh | 64 | 52 | 0.900164 | 0.08282 | 0.25454 | 14.872 |
| nd1 (same environment) | 66 | 54 | 0.902262 | 0.08120 | 0.25362 | 14.820 |
| nd2 (same environment) | 66 | 54 | 0.900796 | 0.08328 | 0.25475 | 14.840 |

| Pair | First epoch val loss differs by >1e-6 | Pooled-mean diff, structure SSIM | RGB SSIM | PSNR | Max per-pair abs diff (structure SSIM) | Pairs differing >1e-6 |
|---|---|---|---|---|---|---|
| original vs fresh | 1 | +0.00052 | +0.00056 | +0.031 | 4.79e-02 | 100% |
| original vs nd1 | 1 | -0.00110 | -0.00037 | -0.022 | 3.17e-02 | 100% |
| original vs nd2 | 1 | +0.00098 | +0.00077 | -0.002 | 3.98e-02 | 100% |
| fresh vs nd1 | 1 | -0.00162 | -0.00093 | -0.052 | 3.95e-02 | 100% |
| fresh vs nd2 | 1 | +0.00046 | +0.00021 | -0.032 | 3.14e-02 | 100% |
| nd1 vs nd2 | 1 | +0.00208 | +0.00114 | +0.020 | 5.54e-02 | 100% |

The three same-environment runs (fresh, nd1, nd2) diverge from each other, all three pairs, from epoch 1, with pooled-mean differences of -0.0016, +0.0005
and +0.0021 (RMS 0.0015, the same order as the all-six-pairs RMS of 0.0013 and larger than original-vs-fresh, 0.0005) and per-pair differences up to
0.055. So identical code, seed, data, Python environment and GPU model are enough to produce divergence: **no environment change between the original
and fresh runs is needed to explain the gap**, and the original-vs-fresh difference is an ordinary draw from this run-to-run spread. The training code
sets seeds but not deterministic-kernel flags; which operations are nondeterministic was not investigated, and enabling deterministic mode would
change training behaviour (a new baseline) rather than reproduce the original runs. The job logs' node and driver lines were not reviewed.

**Wall-clock.** Step A 11.6 min; Step B 3.55 h from first submission to last job end (critical path: the full Transformer at 3.46 h, then the dependent
oracle and swapped-history jobs, which waited 208 min for it); Step C 15 s; about 3.8 h in total, with 10.56 h of summed job time over 11 jobs. The same-environment nondeterminism test added two further Step C reruns
(38 min each, run in parallel).

**Claim-by-claim comparison (42 quantities: original vs fresh).** Each README quantity of the image track recomputed from the fresh run with the
same plant-level bootstrap. "Original / README" is recomputed from the original per-pair scores, except the swap and oracle rows (S, O), whose original
values are the README's. Pooled structure SSIM unless stated.

| Claim | Original / README | Fresh run | Fresh - original | Same sign | CI vs 0 (orig / fresh) | |diff| < original CI half-width |
|---|---|---|---|---|---|---|
| A. structure SSIM, pooled: Step C | +0.0823 | +0.0828 | +0.0005 | yes |  /  |  |
| A. structure SSIM, pooled: ConvLSTM K=1 | +0.0843 | +0.0840 | -0.0003 | yes |  /  |  |
| A. structure SSIM, pooled: ConvLSTM K=4 s42 | +0.0866 | +0.0860 | -0.0006 | yes |  /  |  |
| A. structure SSIM, pooled: ConvLSTM K=4 s43 | +0.0851 | +0.0853 | +0.0002 | yes |  /  |  |
| A. structure SSIM, pooled: Transformer K=1 | +0.0827 | +0.0827 | +0.0001 | yes |  /  |  |
| A. structure SSIM, pooled: Transformer | +0.0879 | +0.0875 | -0.0004 | yes |  /  |  |
| B1. ConvLSTM K=1 - Step C (architecture alone) | +0.0020 | +0.0011 | -0.0008 | yes | excl 0 / excl 0 | no |
| B2. ConvLSTM K=4 s42 - K=1 (history alone) | +0.0024 | +0.0020 | -0.0003 | yes | excl 0 / excl 0 | yes |
| B3. ConvLSTM K=4 s43 - K=1 (history alone) | +0.0008 | +0.0013 | +0.0005 | yes | excl 0 / excl 0 | yes |
| B4. ConvLSTM K=4 s42 - Step C (combined) | +0.0044 | +0.0032 | -0.0012 | yes | excl 0 / excl 0 | no |
| C1. Transformer K=1 - Step C (architecture alone) | +0.0004 | -0.0001 | -0.0005 | **NO** | incl 0 / incl 0 | yes |
| C2. Transformer - Transformer K=1 (history alone) | +0.0052 | +0.0048 | -0.0005 | yes | excl 0 / excl 0 | yes |
| C3. Transformer - Step C (combined) | +0.0056 | +0.0046 | -0.0009 | yes | excl 0 / excl 0 | no |
| C4. Transformer - ConvLSTM K=4 s42 | +0.0012 | +0.0014 | +0.0002 | yes | excl 0 / excl 0 | yes |
| C5. Transformer - ConvLSTM K=4 s43 | +0.0028 | +0.0022 | -0.0006 | yes | excl 0 / excl 0 | yes |
| D1. seed-to-seed: K=4 s43 - K=4 s42 (noise reference) | -0.0016 | -0.0008 | +0.0008 | yes | excl 0 / excl 0 | no |
| D2. identical-input (1 history frame): K=4 s42 - K=1 | +0.0043 | +0.0028 | -0.0015 | yes | excl 0 / excl 0 | yes |
| D3. identical-input (1 history frame): Transformer - Transformer K=1 | +0.0033 | +0.0052 | +0.0020 | yes | excl 0 / excl 0 | no |
| E1. early Field2 (n=146): Transformer - Transformer K=1 | +0.0160 | +0.0173 | +0.0013 | yes | excl 0 / excl 0 | yes |
| E2. early Field2: Transformer - ConvLSTM K=4 s42 | +0.0108 | +0.0130 | +0.0022 | yes | excl 0 / excl 0 | no |
| E3. early Field2: Transformer - ConvLSTM K=4 s43 | +0.0100 | +0.0126 | +0.0026 | yes | excl 0 / excl 0 | no |
| E4. OTHER pairs: Transformer - Transformer K=1 | +0.0038 | +0.0031 | -0.0007 | yes | excl 0 / excl 0 | no |
| E5. OTHER pairs: Transformer - ConvLSTM K=4 s42 | -0.0001 | -0.0001 | -0.0001 | yes | incl 0 / incl 0 | yes |
| F1. early Field2: generic-date - Transformer K=1 | -0.1133 | -0.1142 | -0.0009 | yes | excl 0 / excl 0 | yes |
| F2. share of the early-Field2 Transformer gap reproduced by generic-date | -7.09 | -6.60 | +0.49 | yes | excl 0 / excl 0 | yes |
| E6. early-Field2 pairs' share of the Transformer's pooled history effect | +0.36 | +0.43 | +0.07 | yes |  /  |  |
| G1. day-28-input pairs (n=37): ConvLSTM K=4 s42 - K=1 (untested hypothesis) | +0.0054 | +0.0047 | -0.0007 | yes | excl 0 / excl 0 | yes |
| G2. day-28-input pairs: ConvLSTM K=4 s43 - K=1 | +0.0043 | +0.0053 | +0.0010 | yes | excl 0 / excl 0 | yes |
| H1. protocol: Transformer - Control 3 (structure SSIM) | +0.0385 | +0.0381 | -0.0004 | yes | excl 0 / excl 0 | yes |
| H2. protocol: Step C - Control 3 (structure SSIM) | +0.0329 | +0.0334 | +0.0005 | yes | excl 0 / excl 0 | yes |
| H3. protocol: Transformer - Control 3, RGB SSIM | +0.0420 | +0.0414 | -0.0006 | yes | excl 0 / excl 0 | yes |
| H4. protocol: Transformer - Control 3, PSNR dB | +0.564 | +0.572 | +0.008 | yes | excl 0 / excl 0 | yes |
| I1. every history run above every no-history run: min(history) - max(no-history) | +0.0008 | +0.0013 | +0.0005 | yes |  /  |  |
| I2. mean(history runs) - mean(no-history runs) | +0.0035 | +0.0031 | -0.0004 | yes |  /  |  |
| S1. swap test (146 pairs): original-history gap vs K=1 [README +0.0160] | +0.0160 | +0.0173 | +0.0013 | yes | excl 0 / excl 0 | yes |
| S2. swap test: swapped-history gap vs K=1, mean of 5 draws [README -0.0306] | -0.0306 | -0.0313 | -0.0007 | yes | excl 0 / excl 0 | yes |
| S3. swap test: earlier frames = own last frame, gap vs K=1 [README -0.0272] | -0.0272 | -0.0234 | +0.0038 | yes | excl 0 / excl 0 | no |
| S4. swap test: swapped minus original history [README -0.0466] | -0.0466 | -0.0486 | -0.0020 | yes | excl 0 / excl 0 | yes |
| O1. oracle: early-Field2 gap BEFORE colour-matching [README +0.0160] | +0.0160 | +0.0173 | +0.0013 | yes | excl 0 / excl 0 | yes |
| O2. oracle: gap AFTER mean+std match [README +0.0159] | +0.0159 | +0.0170 | +0.0011 | yes | excl 0 / excl 0 | yes |
| O3. oracle: gap AFTER mean-only match [README +0.0161] | +0.0161 | +0.0174 | +0.0013 | yes | excl 0 / excl 0 | yes |
| O4. oracle: share of the gap remaining after mean+std [README 0.99 [0.97,1.01]] | +0.99 | +0.98 | -0.01 | yes | excl 0 / excl 0 | yes |

Where the fresh value lies outside the original CI half-width (all differences <= 0.004; no sign changes except Transformer K=1 minus Step C, +0.0004
vs -0.0001, both CIs including 0): ConvLSTM K=1 minus Step C (+0.0020 vs +0.0011), K=4 seed 42 minus Step C (+0.0044 vs +0.0032), Transformer minus Step C
(+0.0056 vs +0.0046), seed-to-seed K=4 (-0.0016 vs -0.0008), the Transformer identical-input reference (+0.0033 vs +0.0052), early-Field2 Transformer minus
K=4 (+0.0108 / +0.0100 vs +0.0130 / +0.0126), other pairs Transformer minus K=1 (+0.0038 vs +0.0031), the swap copy-last-frame arm (-0.0272 vs -0.0234) and the
early-Field2 share of the Transformer's pooled history effect (36% vs 43%). None changes a conclusion.
