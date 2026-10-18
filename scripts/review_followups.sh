#!/usr/bin/env bash
# Submit every follow-up experiment from the external-review pass. Run from the project directory on Swan after `git pull`:
#   scripts/review_followups.sh
# 1) diameter clean-input rerun  2) robustness with the latest photo protected  3) sharpness follow-ups (5 jobs, scripts/sharp_followup.sh)
set -euo pipefail
cd "${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
export PROJECT_DIR="$PWD"
sbatch --parsable scripts/diameter_clean_rerun.slurm
sbatch --parsable scripts/robustness_older_only.slurm
scripts/sharp_followup.sh
