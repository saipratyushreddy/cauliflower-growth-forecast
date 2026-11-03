#!/bin/bash
#SBATCH --job-name=trait-lopo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Leave-one-plot-out evaluation of the trait forecasts; see src/trait_forecast_lopo.py and docs/06. Needs outputs/trait_forecast_grid.csv and _per_pair.csv from the plant-wise run.
# Needs data/traits.parquet (built on the laptop by src/build_traits.py from the dataset's samples.json; copy it to Swan's data/).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
test -f data/traits.parquet || { echo "data/traits.parquet missing: copy it from the laptop"; exit 1; }
python -u src/trait_forecast_lopo.py --traits data/traits.parquet --image-pairs data/image_pairs.parquet --exclude-list outputs/tiny_images_md5.txt \
    --embeddings-dir data/embeddings --grid outputs/trait_forecast_grid.csv --plantwise outputs/trait_forecast_per_pair.csv --out-dir outputs \
    --checkpoint-dir checkpoints --device cuda
