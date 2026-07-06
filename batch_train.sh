#!/bin/bash
set -e

CONFIG=$1
RUN_NAME=$2
OUTPUT_DIR=$3

echo "======================================"
echo "Starting Information Extraction training job"
echo "Config:      $CONFIG"
echo "Run name:    $RUN_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Node:        $(hostname)"
echo "======================================"

# ----------------------------
# Environment setup
# ----------------------------
source /gscratch/ubicomp/vysri/conv-fill/conversational-filler/setup_env.sh
source /gscratch/ubicomp/vysri/miniconda/etc/profile.d/conda.sh
conda activate conv

# ----------------------------
# Debug GPU state
# ----------------------------
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "GPUs:"
nvidia-smi

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ps -ef | grep python

# ----------------------------
# Run training
# ----------------------------
srun python /gscratch/ubicomp/vysri/conv-fill/conversational-filler/src/training/finetune_convfill.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR"

echo "Training finished successfully."