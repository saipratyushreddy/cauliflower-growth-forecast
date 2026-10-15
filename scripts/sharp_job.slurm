#!/bin/bash
#SBATCH --job-name=img-sharp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# One run of the sharpness ablation (see scripts/sharp_ablation.sh): NAME, SEED, EXTRA (flags of src/train_img_sharp.py) come from --export.
set -euo pipefail
: "${NAME:?}" "${SEED:?}"
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
# smoke on real images first (never reads test), then the full run
python -u src/train_img_sharp.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs --checkpoint-dir checkpoints \
    --device cuda --name "$NAME" --seed "$SEED" --smoke ${EXTRA:-}
python -u src/train_img_sharp.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs --checkpoint-dir checkpoints \
    --device cuda --name "$NAME" --seed "$SEED" ${EXTRA:-}
