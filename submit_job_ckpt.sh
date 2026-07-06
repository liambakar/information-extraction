#!/bin/bash

EXP_CFG=$1
RUN_NAME=$2
OUTPUT_DIR=$3

mkdir -p $OUTPUT_DIR

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="info_extraction_${RUN_NAME}_train"
#SBATCH --mail-type=FAIL
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

#SBATCH --open-mode=append
#SBATCH --chdir=/mmfs1/gscratch/ubicomp/lbakar/information-extraction
#SBATCH --export=all
#SBATCH --output=${OUTPUT_DIR}/out.log
#SBATCH --error=${OUTPUT_DIR}/err.log

# Run training
./scripts/batch_train.sh ${EXP_CFG} ${RUN_NAME} ${OUTPUT_DIR}

exit 0
EOT