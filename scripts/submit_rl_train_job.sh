#!/bin/bash
set -e

EXP_CFG=$1
RUN_NAME=$2
OUTPUT_DIR=$3

if [ -z "$EXP_CFG" ] || [ -z "$RUN_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 CONFIG RUN_NAME OUTPUT_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="${RUN_NAME}"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lbakar@uw.edu

#SBATCH --account=cse
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --constraint="a40|rtx6k|l40s"
#SBATCH --time=24:00:00

#SBATCH --open-mode=append
#SBATCH --chdir=/mmfs1/gscratch/ubicomp/lbakar/information-extraction
#SBATCH --export=all
#SBATCH --output=${OUTPUT_DIR}/out.log
#SBATCH --error=${OUTPUT_DIR}/err.log

set -e

./scripts/start_rl_training.sh "${EXP_CFG}" "${RUN_NAME}" "${OUTPUT_DIR}"

exit 0
EOT
