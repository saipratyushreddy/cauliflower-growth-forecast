#!/usr/bin/env bash
# Copy result files from Swan to this repo with ONE login (SSH connection sharing).
#
# Usage:
#   scripts/pull_from_swan.sh                      # default: the image-Transformer result files
#   scripts/pull_from_swan.sh outputs/foo.csv 'outputs/img_ctrl*_per_pair.csv' img-foo_123.out
#
# Paths are relative to the project directory on Swan and are copied to the same relative path locally
# (remote globs are fine). Override the login or directory with SWAN_HOST / SWAN_DIR, e.g.
#   SWAN_HOST=skasara2@swan.unl.edu scripts/pull_from_swan.sh
set -euo pipefail

HOST="${SWAN_HOST:-swan}"
REMOTE_DIR="${SWAN_DIR:-/lustre/work/cseguo/skasara2/cauliflower-growth-forecast}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$HOME/.ssh/sockets"
SOCK="$HOME/.ssh/sockets/swan-pull-%C"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SOCK -o ControlPersist=10m"

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(outputs/img_transformer_per_pair.csv outputs/img_transformer_attn.csv outputs/img_transformer_curves.png
         outputs/img_transformer_examples.png outputs/img_transformer_history.json outputs/img_transformer_run.json
         'img-transformer_*.out')
fi

# One authentication here (password / Duo prompt appears once); everything below reuses this connection.
ssh $SSH_OPTS "$HOST" true

for f in "${FILES[@]}"; do
  dest="$LOCAL_DIR/$(dirname "$f")"
  mkdir -p "$dest"
  rsync -av -e "ssh $SSH_OPTS" "$HOST:$REMOTE_DIR/$f" "$dest/"
done

ssh $SSH_OPTS -O exit "$HOST" 2>/dev/null || true   # close the shared connection
echo "Done."
