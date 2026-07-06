#!/bin/bash
set -euo pipefail

echo "======================================"
echo "Starting resumable NuExtract3 extraction worker"
echo "Run name:     ${RUN_NAME:-unset}"
echo "Output dir:   ${OUTPUT_DIR:-unset}"
echo "Dataset path: ${DATASET_PATH:-datasets/preprocessed_dataset_claude_5_tones.jsonl}"
echo "Output path:  ${OUTPUT_PATH:-out/nuextract3_outputs.jsonl}"
echo "Worker:       ${SLURM_ARRAY_TASK_ID:-local}"
echo "Node:         $(hostname)"
echo "======================================"

# ----------------------------
# Environment setup
# ----------------------------
source /gscratch/ubicomp/lbakar/miniconda/etc/profile.d/conda.sh
conda activate genome

# ----------------------------
# Debug GPU state
# ----------------------------
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "GPUs:"
nvidia-smi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_PROCID=${SLURM_PROCID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ----------------------------
# Run resumable extraction
# ----------------------------
bash scripts/run_resume_extraction.sh

echo "Extraction worker finished successfully."
