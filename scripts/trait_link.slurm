#!/bin/bash
#SBATCH --job-name=trait-link
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Direction 3: do generated next photos carry trait information? see src/trait_from_generated.py and docs/06.
# Needs data/traits.parquet (built on the laptop by src/build_traits.py from the dataset's samples.json; copy it to Swan's data/).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
test -f data/traits.parquet || { echo "data/traits.parquet missing: copy it from the laptop"; exit 1; }
python -u src/trait_from_generated.py --image-pairs data/image_pairs.parquet --traits data/traits.parquet --cache-dir data --out-dir outputs \
    --sf-ckpt checkpoints/img_single_frame_best.pt --cl-ckpt checkpoints/img_convlstm_k4_best.pt --tf-ckpt checkpoints/img_transformer_best.pt --device cuda
