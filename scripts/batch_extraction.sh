#!/bin/bash
set -euo pipefail

echo "======================================"
echo "Starting resumable NuExtract3 extraction worker"
echo "Run name:     $RUN_NAME"
echo "Output dir:   $OUTPUT_DIR"
echo "Dataset path: $DATASET_PATH"
echo "Processed data path: $PROCESSED_DATA_PATH"
echo "Worker:       $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "======================================"

# ----------------------------
# Environment setup
# ----------------------------
source /gscratch/ubicomp/lbakar/miniconda3/etc/profile.d/conda.sh
conda activate genome
source /gscratch/ubicomp/lbakar/information-extraction/setup_env.sh

# ----------------------------
# Debug GPU state
# ----------------------------
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "GPUs:"
nvidia-smi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ----------------------------
# Run resumable extraction
# ----------------------------
python3 src/worker/worker.py

echo "Extraction worker finished successfully."
