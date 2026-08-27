#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source /gscratch/ubicomp/lbakar/miniconda3/etc/profile.d/conda.sh
if [ "${CONDA_DEFAULT_ENV:-}" != "genome" ]; then
    conda activate genome
fi

# Some Conda activation/deactivation hooks reference optional variables, so
# enable unset-variable checking only after the environment is ready.
set -u

python -m pip install --upgrade -r "$PROJECT_ROOT/requirements-rl.txt"

cd "$PROJECT_ROOT"
python -c \
    "from src.training.train_rl import validate_training_environment as check;"\
" check()"

echo "GRPO environment is ready."
