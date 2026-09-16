#!/bin/bash
set -e

CONFIG=$1

if [ "$#" -ne 1 ] || [ -z "$CONFIG" ]; then
    echo "Usage: $0 CONFIG" >&2
    exit 2
fi

echo "======================================"
echo "Starting ConvFill training job"
echo "Config:      $CONFIG"
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

echo "GPUs:"
nvidia-smi

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ps -ef | grep python

# ----------------------------
# Run training
# ----------------------------
srun python src/training/finetune_qwen.py --config "$CONFIG"

echo "Training finished successfully."
