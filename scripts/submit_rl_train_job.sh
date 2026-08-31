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

sbatch \
    --job-name="$RUN_NAME" \
    --mail-type=FAIL,END \
    --mail-user=lbakar@uw.edu \
    --account=cse \
    --partition=ckpt-all \
    --requeue \
    --nodes=1 \
    --ntasks-per-node=1 \
    --mem=128G \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    --constraint="a40|rtx6k|l40s" \
    --time=24:00:00 \
    --open-mode=append \
    --chdir=/mmfs1/gscratch/ubicomp/lbakar/information-extraction \
    --export=all \
    --output="$OUTPUT_DIR/out.log" \
    --error="$OUTPUT_DIR/err.log" \
    scripts/start_rl_training.sh "$EXP_CFG" "$RUN_NAME" "$OUTPUT_DIR"
