#!/bin/bash
set -e

CONFIG=$1
RUN_NAME=$2
OUTPUT_DIR=$3

if [ -z "$CONFIG" ] || [ -z "$RUN_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 CONFIG RUN_NAME OUTPUT_DIR" >&2
    exit 2
fi

echo "======================================"
echo "Starting GRPO training job"
echo "Config:      $CONFIG"
echo "Run name:    $RUN_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Node:        $(hostname)"
echo "======================================"

# ----------------------------
# Environment setup
# ----------------------------
source /gscratch/ubicomp/lbakar/miniconda3/etc/profile.d/conda.sh
source /gscratch/ubicomp/lbakar/information-extraction/setup_env.sh
conda activate genome

# ----------------------------
# Debug GPU state
# ----------------------------
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "GRPO dependency preflight:"
python -c \
    "from src.training.train_rl import validate_training_environment as check;"\
" check()"

echo "GPUs:"
nvidia-smi

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ps -ef | grep python

# ----------------------------
# Run GRPO training
# ----------------------------
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
echo "Launching $NPROC_PER_NODE distributed GRPO workers"

srun --nodes=1 --ntasks=1 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$NPROC_PER_NODE" \
    --module src.training.train_rl \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --resume_from last

echo "GRPO training finished successfully."
