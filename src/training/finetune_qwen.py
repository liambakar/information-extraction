import argparse
from pathlib import Path

import lightning as L
import torch

from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from src.training.data.dataset import InstructionDataset
from src.training.modules.callbacks import SaveHFModelCallback
from src.training.modules.info_extraction_lightning_module import InfoExtractionModule
from src.training.utils.checkpointing import resolve_checkpointing
from src.training.utils.config_parser import TrainConfig
from src.training.utils.utils import build_logger, print_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', '--config', type=str, required=True)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--output_dir', type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainConfig(args.config_file)

    TRAIN_PATH = (
        args.train_path if args.train_path else config.data.training_dataset_path
    )

    OUTPUT_DIR = args.output_dir if args.output_dir else config.data.output_directory
    NUM_WORKERS = config.data.num_workers
    MODEL_NAME = config.model.model_name

    if args.run_name:
        config.wandb.run_name = args.run_name

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpointing = resolve_checkpointing(
        output_dir=output_dir,
        run_name=config.wandb.run_name,
        resume_from=config.checkpointing.resume_from,
    )
    ckpt_dir = checkpointing.checkpoint_dir
    hf_dir = output_dir / 'hf'

    logger = build_logger(output_dir, config.wandb)

    L.seed_everything(42, workers=True)

    print_config(config, logger)

    print(f'[LOG] Output directory: {output_dir}', flush=True)
    print(f'[LOG] Lightning checkpoint directory: {ckpt_dir}', flush=True)
    print(f'[LOG] Hugging Face export directory: {hf_dir}', flush=True)
    print(f'\n[LOG] Loading tokenizer: {MODEL_NAME}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f'[LOG] {MODEL_NAME} tokenizer loaded.', flush=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(
            '[LOG] Tokenizer pad token was missing; using eos token as pad token.',
            flush=True,
        )

    print(f'\n[LOG] Loading training dataset: {TRAIN_PATH}', flush=True)
    train_ds = InstructionDataset(
        path=TRAIN_PATH,
        tokenizer=tokenizer,
        template_path=config.data.extraction_template_path,
        max_length=config.model.max_length,
    )
    max_rows = config.data.max_rows
    if max_rows > -1:
        shuffled_indices = torch.randperm(len(train_ds)).tolist()
        shuffled_indices = shuffled_indices[:max_rows]
        train_ds = Subset(train_ds, shuffled_indices)

    print(f'[LOG] Training dataset loaded with {len(train_ds)} examples.', flush=True)

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
        '\n[LOG] Training dataloader ready: '
        f'  batch_size={config.optimization.batch_size}, '
        f'  num_workers={NUM_WORKERS}, '
        f'  persistent_workers={persistent_workers}',
        flush=True,
    )

    print(f'\n[LOG] Loading model: {MODEL_NAME}', flush=True)
    model = InfoExtractionModule(
        model_name=MODEL_NAME,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        warmup_steps=config.optimization.warmup_steps,
    )
    print(f'[LOG] {MODEL_NAME} model loaded.', flush=True)

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

    print('[LOG] Configuring Lightning trainer.', flush=True)
    trainer = L.Trainer(**trainer_kwargs)
    print('[LOG] Lightning trainer configured.', flush=True)

    print(f'[LOG] CUDA available: {torch.cuda.is_available()}', flush=True)
    print(f'[LOG] Visible CUDA GPUs: {torch.cuda.device_count()}', flush=True)

    for i in range(torch.cuda.device_count()):
        print(f'       GPU {i}: {torch.cuda.get_device_name(i)}', flush=True)

    if trainer.global_rank == 0:
        print(f'[LOG] Lightning devices per node: {trainer.num_devices}', flush=True)
        print(f'[LOG] Lightning num nodes: {trainer.num_nodes}', flush=True)
        print(
            f'[LOG] Total Lightning processes / world size: {trainer.num_devices * trainer.num_nodes}',
            flush=True,
        )

    ckpt_path = checkpointing.resume_path

    print('[LOG] Setting model to train mode.', flush=True)
    model.train()
    model.model.train()

    print(f'[LOG] Resume decision: {checkpointing.message}', flush=True)
    print(f'[LOG] Resume checkpoint: {ckpt_path}', flush=True)
    print('[LOG] Starting training...', flush=True)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=ckpt_path,
    )

    print('[LOG] Training completed! 🥳', flush=True)
    print(f'[LOG] Final Hugging Face model export path: {hf_dir / "final"}', flush=True)


if __name__ == '__main__':
    main()
