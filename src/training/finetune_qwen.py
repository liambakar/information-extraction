import argparse
from pathlib import Path

import lightning as L

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from configs import qwen_training_config
from src.training.data.dataset import InstructionDataset
from src.training.modules.callbacks import SaveHFModelCallback
from src.training.modules.qwen_lightning_module import QwenFullFinetuneModule


def build_logger(output_dir):
    if not qwen_training_config.USE_WANDB:
        print('W&B logging disabled.', flush=True)
        return None

    try:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=qwen_training_config.WANDB_PROJECT,
            name=qwen_training_config.WANDB_RUN_NAME,
            entity=qwen_training_config.WANDB_ENTITY,
            save_dir=str(output_dir),
            log_model=qwen_training_config.WANDB_LOG_MODEL,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            'W&B logging is enabled, but wandb is not installed or available. '
            'Install wandb in the training environment or set USE_WANDB = False '
            'in configs/qwen_training_config.py.'
        ) from exc

    print('W&B logging enabled.', flush=True)

    return logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_workers', type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    TRAIN_PATH = (
        qwen_training_config.TRAINING_DATASET_PATH
        if args.train_path is None
        else args.train_path
    )

    OUTPUT_DIR = (
        qwen_training_config.OUTPUT_DIRECTORY
        if args.output_dir is None
        else args.output_dir
    )
    NUM_WORKERS = (
        qwen_training_config.NUM_WORKERS
        if args.num_workers is None
        else args.num_workers
    )
    MODEL_NAME = qwen_training_config.MODEL_NAME

    L.seed_everything(42, workers=True)

    output_dir = Path(OUTPUT_DIR)
    ckpt_dir = output_dir / 'checkpoints'
    hf_dir = output_dir / 'hf'

    print(f'Output directory: {output_dir}', flush=True)
    print(f'Lightning checkpoint directory: {ckpt_dir}', flush=True)
    print(f'Hugging Face export directory: {hf_dir}', flush=True)

    print(f'Loading tokenizer: {MODEL_NAME}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(
            'Tokenizer pad token was missing; using eos token as pad token.', flush=True
        )

    print('Tokenizer loaded.', flush=True)

    print(f'Loading training dataset: {TRAIN_PATH}', flush=True)
    train_ds = InstructionDataset(
        path=TRAIN_PATH,
        tokenizer=tokenizer,
        template_path=qwen_training_config.EXTRACTION_TEMPLATE_PATH,
        max_length=qwen_training_config.MAX_LENGTH,
    )
    print(f'Training dataset loaded with {len(train_ds)} examples.', flush=True)

    persistent_workers = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=qwen_training_config.BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    print(
        'Training dataloader ready: '
        f'  batch_size={qwen_training_config.BATCH_SIZE}, '
        f'  num_workers={NUM_WORKERS}, '
        f'  persistent_workers={persistent_workers}',
        flush=True,
    )

    print(f'Loading model: {MODEL_NAME}', flush=True)
    model = QwenFullFinetuneModule(
        model_name=MODEL_NAME,
        lr=qwen_training_config.LR,
        weight_decay=qwen_training_config.WEIGHT_DECAY,
        warmup_steps=qwen_training_config.WARMUP_STEPS,
    )
    print('Model loaded.', flush=True)

    # Store tokenizer only for final HF export.
    model.tokenizer = tokenizer

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='step-{step:08d}-epoch-{epoch:02d}',
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=500,
    )

    hf_export_callback = SaveHFModelCallback(output_dir=hf_dir)

    logger = build_logger(output_dir)

    trainer_kwargs = {
        'accelerator': 'gpu',
        'devices': qwen_training_config.DEVICES,
        'num_nodes': qwen_training_config.NUM_NODES,
        'strategy': DDPStrategy(find_unused_parameters=False),
        'precision': 'bf16-mixed',
        'max_epochs': qwen_training_config.MAX_EPOCHS,
        'accumulate_grad_batches': qwen_training_config.ACCUMULATE_GRAD_BATCHES,
        'gradient_clip_val': 1.0,
        'log_every_n_steps': 10,
        'default_root_dir': output_dir,
        'callbacks': [
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step'),
            hf_export_callback,
        ],
    }

    if logger is not None:
        trainer_kwargs['logger'] = logger

    print('Configuring Lightning trainer.', flush=True)
    trainer = L.Trainer(**trainer_kwargs)
    print('Lightning trainer configured.', flush=True)

    ckpt_path = qwen_training_config.RESUME_FROM

    if ckpt_path == 'last':
        ckpt_path = str(ckpt_dir / 'last.ckpt')

    print(f'Resume checkpoint: {ckpt_path}', flush=True)
    print('Starting training.', flush=True)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=ckpt_path,
    )

    print('Training completed.', flush=True)
    print(f'Final Hugging Face model export path: {hf_dir / "final"}', flush=True)


if __name__ == '__main__':
    main()
