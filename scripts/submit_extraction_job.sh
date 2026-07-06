#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 RUN_NAME OUTPUT_DIR NUM_WORKERS" >&2
    exit 1
fi

RUN_NAME=$1
OUTPUT_DIR=$2
NUM_WORKERS=$3

if ! [[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_WORKERS must be a positive integer." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

MAX_ARRAY_INDEX=$((NUM_WORKERS - 1))

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="info_extraction_${RUN_NAME}"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lbakar@uw.edu

#SBATCH --account=cse
#SBATCH --partition=ckpt-all
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
