#!/usr/bin/env bash
# Follow-up to the sharpness ablation, addressing its stated limits (docs/04 "Scope of the negative result"). Run from the project directory on Swan:
#   scripts/sharp_followup.sh
# All runs: up to 120 epochs (patience 25) instead of 80/12, checkpoint chosen on VAL structure SSIM instead of combined loss. Names have a "fu_" prefix.
#   fu_base        no ingredient (Step C recipe) -- the control for the longer budget + new selection metric
#   fu_perc1       perceptual weight 1.0 (as before), longer + structure selection
#   fu_perc01      perceptual weight 0.1   (L1/SSIM terms dominate)
#   fu_perc003     perceptual weight 0.03
#   fu_anygap_keep any-gap with ALL standard pairs kept every epoch + 5,823 sampled any-gap pairs
set -euo pipefail
cd "${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
SEED="${SEED:-42}"
COMMON="--max-epochs 120 --patience 25 --select-metric structure"
run() {  # run <name> <extra args...>
  local name="$1"; shift
  sbatch --parsable --time=12:00:00 --job-name="$name" --export=ALL,NAME="$name",SEED="$SEED",EXTRA="$COMMON $*" scripts/sharp_job.slurm
}
run fu_base
run fu_perc1       --w-perc 1.0
run fu_perc01      --w-perc 0.1
run fu_perc003     --w-perc 0.03
run fu_anygap_keep --any-gap --keep-standard --eval-any-gap
