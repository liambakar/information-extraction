# Data / output paths
TRAINING_DATASET_PATH = ''
OUTPUT_DIRECTORY = ''
EXTRACTION_TEMPLATE_PATH = 'extraction_templates/template.json'
NUM_WORKERS = 8

# Model / tokenization / optimization
MODEL_NAME = 'Qwen/Qwen3-0.6B'
MAX_LENGTH = 2048
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 8
MAX_EPOCHS = 3

# Distributed / hardware
DEVICES = 'auto'
NUM_NODES = 1

# Weights & Biases logging
USE_WANDB = True
WANDB_PROJECT = 'information-extraction'
WANDB_RUN_NAME = None
WANDB_ENTITY = None
WANDB_LOG_MODEL = False

# Checkpoint / resume behavior
# Use "last" to resume from: OUTPUT_DIRECTORY/checkpoints/last.ckpt
# Use None to start training from scratch.
RESUME_FROM = None
