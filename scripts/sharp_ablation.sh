#!/usr/bin/env bash
# Submit the sharpness ablation (single-frame model, one ingredient at a time) as parallel GPU jobs from the ORIGINAL project directory on Swan.
#   scripts/sharp_ablation.sh                # seed 42, all variants
#   SEED=43 scripts/sharp_ablation.sh        # another seed (outputs get a _seed43 suffix)
# Variants: perc | residual | residual_perc | anygap | full (= residual + perceptual + any-gap).  Step C itself (no ingredient) is the existing run.
set -euo pipefail
cd "${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
SEED="${SEED:-42}"
run() {  # run <name> <extra args...>
  local name="$1"; shift
  sbatch --parsable --job-name="sh-$name" --export=ALL,NAME="$name",SEED="$SEED",EXTRA="$*" scripts/sharp_job.slurm
}
run perc          --w-perc 1.0
run residual      --residual
run residual_perc --residual --w-perc 1.0
run anygap        --any-gap --eval-any-gap
run full          --residual --w-perc 1.0 --any-gap --eval-any-gap
