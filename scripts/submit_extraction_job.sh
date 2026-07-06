#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
    echo "Usage: $0 RUN_NAME OUTPUT_DIR NUM_WORKERS [--reset-scratch]" >&2
    exit 1
fi

RUN_NAME=$1
OUTPUT_DIR=$2
NUM_WORKERS=$3
RESET_SCRATCH=false

for arg in "${@:4}"; do
    if [[ "$arg" == "--reset-scratch" ]]; then
        RESET_SCRATCH=true
    fi
done

if ! [[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_WORKERS must be a positive integer." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

MAX_ARRAY_INDEX=$((NUM_WORKERS - 1))

# Global entrypoint: reset all incomplete rows once before workers start
# On requeue, per-worker release_worker_claims() handles recovery per worker
if [[ "$RESET_SCRATCH" == "true" ]]; then
    echo "Resetting from scratch (full wipe)..."
    python3 src/worker/resume.py reset-scratch
else
    echo "Resetting incomplete rows (entrypoint)..."
    bash scripts/reset_entrypoint.sh
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="info_extraction_${RUN_NAME}"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lbakar@uw.edu,vysri@cs.washington.edu

#SBATCH --account=cse
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a40|rtx6k|l40s"
#SBATCH --time=24:00:00
#SBATCH --array=0-${MAX_ARRAY_INDEX}

#SBATCH --open-mode=append
#SBATCH --chdir=/mmfs1/gscratch/ubicomp/lbakar/information-extraction
#SBATCH --export=all,RUN_NAME=${RUN_NAME},OUTPUT_DIR=${OUTPUT_DIR}
#SBATCH --output=${OUTPUT_DIR}/out_%A_%a.log
#SBATCH --error=${OUTPUT_DIR}/err_%A_%a.log

./scripts/batch_extraction.sh

exit 0
EOT
