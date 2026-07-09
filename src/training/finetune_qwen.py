import argparse
from pathlib import Path

import lightning as L
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from src.training.data.dataset import InstructionDataset
from src.training.modules.callbacks import SaveHFModelCallback
from src.training.modules.qwen_lightning_module import InfoExtractionModule
from src.training.utils.config_parser import TrainConfig
from src.training.utils.utils import build_logger, print_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_workers', type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainConfig(args.config_file)

    TRAIN_PATH = (
        config.data.training_dataset_path
        if args.train_path is None
        else args.train_path
    )

    OUTPUT_DIR = (
        config.data.output_directory if args.output_dir is None else args.output_dir
    )
    NUM_WORKERS = (
        config.data.num_workers if args.num_workers is None else args.num_workers
    )
    MODEL_NAME = config.model.model_name


    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    hf_dir = output_dir / 'hf'

    logger = build_logger(output_dir, config.wandb)

    L.seed_everything(42, workers=True)

    print_config(config, logger)

    print(f'Output directory: {output_dir}', flush=True)
    print(f'Lightning checkpoint directory: {ckpt_dir}', flush=True)
    print(f'Hugging Face export directory: {hf_dir}', flush=True)
    print(f'\nLoading tokenizer: {MODEL_NAME}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f'{MODEL_NAME} tokenizer loaded.', flush=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(
            'Tokenizer pad token was missing; using eos token as pad token.', flush=True
        )

    print(f'\nLoading training dataset: {TRAIN_PATH}', flush=True)
    train_ds = InstructionDataset(
        path=TRAIN_PATH,
        tokenizer=tokenizer,
        template_path=config.data.extraction_template_path,
        max_length=config.model.max_length,
    )
    print(f'Training dataset loaded with {len(train_ds)} examples.', flush=True)

    persistent_workers = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    print(
        '\nTraining dataloader ready: '
        f'  batch_size={config.optimization.batch_size}, '
        f'  num_workers={NUM_WORKERS}, '
        f'  persistent_workers={persistent_workers}',
        flush=True,
    )

    print(f'\nLoading model: {MODEL_NAME}', flush=True)
    model = InfoExtractionModule(
        model_name=MODEL_NAME,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        warmup_steps=config.optimization.warmup_steps,
    )
    print(f'{MODEL_NAME} model loaded.', flush=True)

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

    trainer_kwargs = {
        'accelerator': 'gpu',
        'devices': config.hardware.devices,
        'num_nodes': config.hardware.num_nodes,
        'strategy': DDPStrategy(find_unused_parameters=False),
        'precision': 'bf16-mixed',
        'max_epochs': config.optimization.max_epochs,
        'accumulate_grad_batches': config.optimization.accumulate_grad_batches,
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

    print(f'CUDA available: {torch.cuda.is_available()}', flush=True)
    print(f'Visible CUDA GPUs: {torch.cuda.device_count()}', flush=True)

    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}', flush=True)

    if trainer.global_rank == 0:
        print(f'Lightning devices per node: {trainer.num_devices}', flush=True)
        print(f'Lightning num nodes: {trainer.num_nodes}', flush=True)
        print(
            f'Total Lightning processes / world size: {trainer.num_devices * trainer.num_nodes}',
            flush=True,
        )

    ckpt_path = config.checkpointing.resume_from

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
