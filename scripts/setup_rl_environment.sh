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

check_torch_installation() {
    python -c \
        "import torch;"\
" from torch.distributed.tensor import ("\
"DTensor, Partial, Replicate, Shard, distribute_tensor);"\
" from torch.nn.attention.flex_attention import flex_attention;"\
" assert torch.__version__.startswith('2.11.0+cu128')"
}

if ! check_torch_installation >/dev/null 2>&1; then
    echo "Repairing the CUDA 12.8 PyTorch installation..."
    python -m pip install \
        --force-reinstall \
        --no-cache-dir \
        --no-deps \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

echo "Verifying PyTorch installation..."
check_torch_installation

python -m pip install --upgrade -r "$PROJECT_ROOT/requirements-rl.txt"

cd "$PROJECT_ROOT"
python -c \
    "from src.training.train_rl import validate_training_environment as check;"\
" check()"

echo "GRPO environment is ready."
