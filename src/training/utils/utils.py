import json
import torch
import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import random_split


from src.training.utils.config_parser import TrainConfig, WandbConfig


def build_logger(output_dir, wandb_config: WandbConfig) -> WandbLogger | None:
    if not wandb_config.use_wandb:
        print('W&B logging disabled.', flush=True)
        return None

    try:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=wandb_config.project,
            name=wandb_config.run_name,
            entity=wandb_config.entity,
            save_dir=str(output_dir),
            log_model=wandb_config.log_model,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            'W&B logging is enabled, but wandb is not installed or available. '
            'Install wandb in the training environment or set USE_WANDB = False '
            'in configs/qwen_training_config.py.'
        ) from exc

    print('W&B logging enabled.', flush=True)

    return logger


@rank_zero_only
def print_config(config: TrainConfig, logger: WandbLogger | None) -> None:
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
    config_string = json.dumps(config_dict, indent=4)

    border = '=' * 50
    header = f'{"=" * 11}     Loaded config file     {"=" * 11}'
    full_output = f'\n\n{border}\n{header}\n{border}\n{config_string}\n{border}\n\n'

    print(full_output, flush=True)

    if logger is not None:
        logger.experiment.config.update(config_dict, allow_val_change=True)

        html_content = f"""
        <h3>Configuration File Run-Time</h3>
        <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: monospace;">
{config_string}
        </pre>
        """

        logger.experiment.log({'runtime_config': wandb.Html(html_content)})


def split_train_validation(dataset, validation_split: float, seed: int):
    if validation_split <= 0.0:
        return dataset, None

    n = len(dataset)

    val_n = int(n * validation_split)
    val_n = max(1, val_n)
    train_n = n - val_n

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_n, val_n], generator=generator)
