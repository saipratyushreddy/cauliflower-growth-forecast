#!/bin/bash
#SBATCH --job-name=rp-stepA
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Reproducibility pass, STEP A (data verification). Submit from inside the FRESH clone:
#   sbatch --export=ALL,ORIG=$WORK/cauliflower-growth-forecast scripts/repro/step_a.slurm
# Reads the original data (ORIG) read-only; rebuilds the pair table and the image cache into a scratch dir and compares.
set -euo pipefail
: "${ORIG:?set ORIG to the original project directory}"
cd "$SLURM_SUBMIT_DIR"
source "$ORIG/.venv/bin/activate"
python -u src/repro_verify_data.py --project-dir "$ORIG" --code-dir "$SLURM_SUBMIT_DIR" \
    --scratch-dir "${SLURM_SUBMIT_DIR}_scratch_stepA" --report repro_step_a.json
