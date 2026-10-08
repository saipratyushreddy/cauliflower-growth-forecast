#!/usr/bin/env bash
# Reproducibility pass, between Step A and Step B. Run INSIDE the fresh clone, AFTER src/repro_verify_data.py passed:
#   cd $WORK/cauliflower-repro && scripts/repro/setup_fresh.sh $WORK/cauliflower-growth-forecast
# - big read-only inputs (raw images, image cache) and the python environment are SYMLINKED (nothing is re-downloaded or rebuilt;
#   the venv is shared, so library versions are identical by construction and are recorded by Step A)
# - the small tables the jobs read are COPIED, never symlinked, so no job can rewrite the original through a link
# - the ORIGINAL results are frozen into reference/ (read-only) so they cannot change while the fresh run is in progress
set -euo pipefail
ORIG="$(readlink -f "${1:?usage: setup_fresh.sh <original project dir>}")"
FRESH="$(pwd -P)"
[ "$ORIG" != "$FRESH" ] || { echo "ERROR: run this inside the FRESH clone, not in the original directory"; exit 1; }
[ -d .git ] && [ -f src/repro_verify_data.py ] || { echo "ERROR: this does not look like a fresh clone of the repository"; exit 1; }
for f in data/images data/images256.npy data/images256_index.parquet data/image_pairs.parquet data/metadata.parquet data/pairs_split.parquet \
         data/norm_stats.json outputs/tiny_images_md5.txt .venv; do
  [ -e "$ORIG/$f" ] || { echo "ERROR: $ORIG/$f missing"; exit 1; }
done
mkdir -p data outputs checkpoints
ln -sfn "$ORIG/data/images" data/images
ln -sfn "$ORIG/data/images256.npy" data/images256.npy
ln -sfn "$ORIG/data/images256_index.parquet" data/images256_index.parquet
ln -sfn "$ORIG/.venv" .venv
for f in image_pairs.parquet metadata.parquet pairs_split.parquet norm_stats.json; do cp -p "$ORIG/data/$f" "data/$f"; done
cp -p "$ORIG/outputs/tiny_images_md5.txt" outputs/
if [ ! -d reference ]; then
  mkdir -p reference/logs
  cp -a "$ORIG/outputs" reference/outputs
  cp -p "$ORIG"/img-*.out reference/logs/ 2>/dev/null || true
  cp -p "$ORIG"/outputs/img-*.out reference/logs/ 2>/dev/null || true
  chmod -R a-w reference
fi
echo "Fresh directory ready: $FRESH"
echo "  code commit : $(git rev-parse HEAD)"
echo "  reference   : $(ls reference/outputs | wc -l) result files frozen (read-only) in reference/"
echo "  next        : scripts/repro/submit_all.sh"
