#!/usr/bin/env bash
# Reproducibility pass, STEP B: submit every training run, control and diagnostic of the image-to-image track as PARALLEL SLURM jobs.
# Run INSIDE the fresh clone after setup_fresh.sh. Dependencies exist only where a job needs another job's output:
#   baseline (CPU) --> Step C (GPU), controls (CPU; after Step C);  Transformer + Transformer K=1 --> oracle diagnostic, swapped-history test.
# Seeds are the original ones (default 42; ConvLSTM K=4 second run 43). Records job ids + submit times in repro_jobs.tsv.
set -euo pipefail
export PROJECT_DIR="$(pwd -P)"
[ -d reference ] && [ -L data/images ] && [ -f data/image_pairs.parquet ] || { echo "ERROR: run scripts/repro/setup_fresh.sh first"; exit 1; }
[ ! -e repro_jobs.tsv ] || { echo "ERROR: repro_jobs.tsv exists: jobs were already submitted from this directory"; exit 1; }
printf "name\tjobid\tsubmitted\n" > repro_jobs.tsv
J() {  # J <name> [sbatch options...] <script>
  local name="$1"; shift
  local id; id="$(sbatch --parsable --job-name="$name" "$@")"
  printf "%s\t%s\t%s\n" "$name" "$id" "$(date +%FT%T)" >> repro_jobs.tsv
  echo "$name -> $id" >&2
  echo "$id"
}
E="--export=ALL"
BASE=$(J rp-baseline "--export=ALL,SKIP_PAIR_BUILD=1" scripts/image_copy_forward_baseline.slurm)
STEPC=$(J rp-stepc "--dependency=afterok:$BASE" $E scripts/train_img_single_frame.slurm)
J rp-controls "--dependency=afterok:$BASE:$STEPC" $E scripts/step_c_controls.slurm > /dev/null
J rp-genericdate $E scripts/generic_date_control.slurm > /dev/null
J rp-cl-k1 "--export=ALL,K=1" scripts/train_img_convlstm.slurm > /dev/null
J rp-cl-k4 "--export=ALL,K=4" scripts/train_img_convlstm.slurm > /dev/null
J rp-cl-k4-s43 "--export=ALL,K=4,SEED=43" scripts/train_img_convlstm.slurm > /dev/null
TF=$(J rp-tf $E scripts/train_img_transformer.slurm)
TFK1=$(J rp-tfk1 "--export=ALL,MAXHIST=1" --time=03:00:00 scripts/train_img_transformer.slurm)
J rp-oracle "--dependency=afterok:$TF:$TFK1" $E scripts/oracle_colormatch_transformer.slurm > /dev/null
J rp-swap "--dependency=afterok:$TF:$TFK1" $E scripts/swapped_history_test.slurm > /dev/null
echo; column -t -s$'\t' repro_jobs.tsv
echo; echo "Monitor: squeue -u \$USER   |  when all have left the queue: python src/repro_times.py && python src/repro_compare.py ..."
