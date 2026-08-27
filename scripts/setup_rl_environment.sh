#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source /gscratch/ubicomp/lbakar/miniconda3/etc/profile.d/conda.sh
conda activate genome

python -m pip install --upgrade -r "$PROJECT_ROOT/requirements-rl.txt"

cd "$PROJECT_ROOT"
python -c \
    "from src.training.train_rl import validate_training_environment as check;"\
" check()"

echo "GRPO environment is ready."
