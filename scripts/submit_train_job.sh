#!/bin/bash
set -e

EXP_CFG=$1

if [ "$#" -ne 1 ] || [ -z "$EXP_CFG" ]; then
    echo "Usage: $0 CONFIG" >&2
    exit 2
fi

EXP_CFG=$(realpath "$EXP_CFG")
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"
RUN_NAME=$(python3 - "$EXP_CFG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as config_file:
    run_name = json.load(config_file)['wandb']['run_name']
if not isinstance(run_name, str) or not run_name.strip():
    raise SystemExit('wandb.run_name must be a nonempty name')
print(run_name)
PY
)
OUTPUT_DIR=$(python3 - "$EXP_CFG" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding='utf-8') as config_file:
    output_dir = json.load(config_file)['data']['output_directory']
if not isinstance(output_dir, str) or not output_dir.strip():
    raise SystemExit('data.output_directory must be a nonempty path')
print(Path(output_dir).resolve())
PY
)
mkdir -p "$OUTPUT_DIR"

sbatch \
    --job-name="$RUN_NAME" \
    --mail-type=FAIL,END \
    --mail-user=lbakar@uw.edu \
    --account=cse \
    --partition=ckpt-all \
    --nodes=1 \
    --ntasks-per-node=4 \
    --mem=128G \
    --gres=gpu:4 \
    --cpus-per-task=8 \
    --constraint="a40|rtx6k|l40s" \
    --time=24:00:00 \
    --open-mode=append \
    --chdir="$PROJECT_DIR" \
    --export=all \
    --output="$OUTPUT_DIR/out.log" \
    --error="$OUTPUT_DIR/err.log" \
    scripts/start_training.sh "$EXP_CFG"
