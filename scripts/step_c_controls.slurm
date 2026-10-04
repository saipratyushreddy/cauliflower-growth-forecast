#!/bin/bash
#SBATCH --job-name=img-ctrls
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

# CPU only, closed-form, no training. Needs data/images256.npy (made by the Step C job) and the Step C outputs.
python -u src/step_c_controls.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs --reuse-val-curve \
    --model-checkpoint checkpoints/img_single_frame_best.pt \
    --model-per-pair-csv outputs/img_single_frame_per_pair.csv \
    --copy-forward-per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv

# Full Step-C-style breakdown (pooled / per-field / with-without day 28 / mean+median) for each control,
# paired against copy-forward on the same test pairs.
for k in 1 2 3; do
  python -u src/summarize_baseline.py --per-pair-csv outputs/img_ctrl${k}_per_pair.csv \
      --image-pairs data/image_pairs.parquet --no-oracle --label "control ${k}" \
      --baseline-per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv \
      --out outputs/img_ctrl${k}_final_protocol.json
done
