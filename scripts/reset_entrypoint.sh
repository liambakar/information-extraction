#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 DATASET_PATH" >&2
    exit 1
fi

DATASET_PATH=$1

if [[ -z "$DATASET_PATH" ]]; then
    echo "DATASET_PATH must be provided." >&2
    exit 1
fi

python3 src/worker/resume.py reset-all --dataset_path "$DATASET_PATH"
