#!/bin/bash
#SBATCH --job-name=trait-forecast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Direction 2: multi-trait forecasting with prediction ranges; see src/trait_forecast.py and docs/06.
# Needs data/traits.parquet (built on the laptop by src/build_traits.py from the dataset's samples.json; copy it to Swan's data/).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
test -f data/traits.parquet || { echo "data/traits.parquet missing: copy it from the laptop"; exit 1; }
python -u src/trait_forecast.py --traits data/traits.parquet --image-pairs data/image_pairs.parquet --exclude-list outputs/tiny_images_md5.txt \
    --embeddings-dir data/embeddings --out-dir outputs --checkpoint-dir checkpoints --device cuda
