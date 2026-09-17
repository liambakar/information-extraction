import argparse
import importlib.metadata
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from src.training.modules.rewards import (
    RewardConfig,
    build_reward_functions,
    build_reward_weights,
)
from src.training.utils.config_parser import TrainConfig


START_SEQ = '<|im_start|>'
END_SEQ = '<|im_end|>\n'

CLASSIFICATION_TO_SECTION = {
    'Activities': 'activity',
    'Food': 'food',
    'Mood': 'mood',
    'Symptoms': 'symptom',
    'Treatment': 'treatment',
}

RL_DEPENDENCY_REQUIREMENTS = {
    'torch': '==2.11.0',
    'transformers': '==5.13.1',
    'trl': '==1.10.0',
    'datasets': '==4.7.0',
    'accelerate': '>=1.4.0,<2',
}
RL_SETUP_SCRIPT = 'scripts/setup_rl_environment.sh'


@dataclass(frozen=True)
class RLTrainingSettings:
    """GRPO-specific settings loaded from the config's ``rl`` section."""

    num_generations: int
    max_completion_length: int
    temperature: float
    beta: float
    logging_steps: int
    eval_steps: int
    num_completions_to_print: int
    seed: int
    use_vllm: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fine-tune an extraction model with TRL GRPO.',
    )
    parser.add_argument('--config_file', '--config', required=True)
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def format_message(role: str, content: str) -> str:
    """Format a prompt message using the project's Qwen chat convention."""
    return f'{START_SEQ}{role}\n{content}\n{END_SEQ}'


def build_prompt(template: dict[str, Any], utterance: str) -> str:
    """Build the generation prompt expected by the supervised checkpoint."""
    template_text = json.dumps(template, indent=4, ensure_ascii=False)
    return (
        format_message('template', template_text)
        + format_message('user', utterance)
        + START_SEQ
        + 'assistant\n'
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows and report malformed line locations."""
    rows = []
    with Path(path).open('r', encoding='utf-8') as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid JSON in {path} at line {line_number}.'
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected an object in {path} at line {line_number}.')
            rows.append(row)
    return rows


def build_rl_records(
    rows: Iterable[dict[str, Any]],
    template: dict[str, Any],
    dataset_type: str,
) -> list[dict[str, str]]:
    """Convert extraction rows into prompt and reward-metadata records."""
    records = []

    for row_number, row in enumerate(rows, start=1):
        try:
            utterance = row['utterance']
            extraction = row['extraction']
        except KeyError as exc:
            raise ValueError(
                f'Dataset row {row_number} is missing {exc.args[0]!r}.'
            ) from exc

        if dataset_type == 'full':
            records.append(
                {
                    'prompt': build_prompt(template, utterance),
                    'ground_truth': json.dumps(extraction, ensure_ascii=False),
                    'schema_section': '',
                    'utterance': utterance,
                }
            )
            continue

        if dataset_type != 'partial':
            raise ValueError(f'Unsupported dataset type: {dataset_type!r}.')

        classifications = row.get('type')
        if not isinstance(classifications, list) or not classifications:
            raise ValueError(
                f'Dataset row {row_number} needs a non-empty type list.'
            )

        for classification in classifications:
            try:
                section = CLASSIFICATION_TO_SECTION[classification]
                section_template = template[section]
                section_truth = extraction[section]
            except KeyError as exc:
                raise ValueError(
                    f'Dataset row {row_number} cannot build partial section '
                    f'{classification!r}; missing {exc.args[0]!r}.'
                ) from exc

            records.append(
                {
                    'prompt': build_prompt(section_template, utterance),
                    'ground_truth': json.dumps(section_truth, ensure_ascii=False),
                    'schema_section': section,
                    'utterance': utterance,
                }
            )

    return records


def select_rows(
    rows: list[dict[str, Any]],
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a reproducible random subset without mutating the input rows."""
    if max_rows < 0 or max_rows >= len(rows):
        return list(rows)

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:max_rows]]


def split_records(
    records: list[dict[str, str]],
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]] | None]:
    """Create a deterministic train/validation split."""
    if validation_split <= 0.0:
        return records, None
    if len(records) < 2:
        raise ValueError('At least two records are required for validation splitting.')

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(len(records) * validation_split))
    validation_size = min(validation_size, len(records) - 1)
    validation_indices = set(indices[:validation_size])

    training = [
        record
        for index, record in enumerate(records)
        if index not in validation_indices
    ]
    validation = [
        record for index, record in enumerate(records) if index in validation_indices
    ]
    return training, validation


def load_reward_config(path: str | Path) -> RewardConfig:
    """Load explicitly configured reward values from a JSON object."""
    with Path(path).open('r', encoding='utf-8') as config_file:
        values = json.load(config_file)
    if not isinstance(values, dict):
        raise ValueError('Reward config must contain a JSON object.')

    if 'rewards' in values:
        values = values['rewards']
    elif 'data' in values:
        raise ValueError('Training config must contain a rewards section.')
    if not isinstance(values, dict):
        raise ValueError('The rewards config entry must be a JSON object.')

    try:
        return RewardConfig(**values)
    except TypeError as exc:
        raise ValueError(f'Invalid reward config: {exc}') from exc


def load_rl_settings(path: str | Path) -> RLTrainingSettings:
    """Load GRPO-specific settings from a training config."""
    with Path(path).open('r', encoding='utf-8') as config_file:
        config_data = json.load(config_file)
    if not isinstance(config_data, dict):
        raise ValueError('Training config must contain a JSON object.')

    if 'rl' not in config_data:
        raise ValueError('Training config must contain an rl section.')
    values = config_data['rl']
    if not isinstance(values, dict):
        raise ValueError('The rl config entry must be a JSON object.')

    try:
        settings = RLTrainingSettings(**values)
    except TypeError as exc:
        raise ValueError(f'Invalid rl config: {exc}') from exc

    validate_rl_settings(settings)
    return settings


def validate_rl_settings(settings: RLTrainingSettings) -> None:
    """Validate GRPO settings loaded from config."""

    if settings.num_generations <= 1:
        raise ValueError('rl.num_generations must be greater than one.')
    if settings.max_completion_length <= 0:
        raise ValueError('rl.max_completion_length must be positive.')
    if settings.temperature <= 0:
        raise ValueError('rl.temperature must be positive.')
    if settings.beta < 0:
        raise ValueError('rl.beta cannot be negative.')
    for name in ('logging_steps', 'eval_steps'):
        if getattr(settings, name) <= 0:
            raise ValueError(f'rl.{name} must be positive.')
    if settings.num_completions_to_print < 0:
        raise ValueError('rl.num_completions_to_print cannot be negative.')


def find_last_checkpoint(output_dir: str | Path) -> str | None:
    """Return the highest-numbered Hugging Face trainer checkpoint."""
    checkpoints = []
    for path in Path(output_dir).glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-(\d+)', path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def resolve_resume_checkpoint(
    output_dir: str | Path,
    resume_from: str | None,
) -> str | None:
    """Resolve ``last`` or preserve an explicit trainer checkpoint path."""
    if not resume_from:
        return None
    if resume_from == 'last':
        return find_last_checkpoint(output_dir)
    return resume_from


def validate_generation_batch(
    config: TrainConfig,
    num_generations: int,
) -> None:
    """Validate the GRPO group size when the configured world size is known."""
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than one for GRPO.')
    if not isinstance(config.hardware.devices, int):
        return

    effective_batch_size = (
        config.hardware.devices
        * config.hardware.num_nodes
        * config.optimization.batch_size
        * config.optimization.accumulate_grad_batches
    )
    if effective_batch_size % num_generations:
        raise ValueError(
            'The effective batch size must be divisible by num_generations; '
            f'got {effective_batch_size} and {num_generations}.'
        )


def validate_completion_logging_support(
    grpo_config_class: type,
    grpo_trainer_class: type,
) -> None:
    """Fail before training when TRL cannot produce the requested W&B table."""
    config_fields = getattr(grpo_config_class, '__dataclass_fields__', {})
    required_fields = {
        'log_completions',
        'num_completions_to_print',
        'log_unique_prompts',
    }
    missing_fields = required_fields - set(config_fields)
    supports_extra_columns = hasattr(
        grpo_trainer_class,
        '_log_completion_extra',
    )
    if missing_fields or not supports_extra_columns:
        missing = sorted(missing_fields)
        raise RuntimeError(
            'The installed TRL version cannot log the required completion '
            'table with ground truths. Upgrade TRL before training. '
            f'Missing config fields: {missing}; '
            f'log_extra support: {supports_extra_columns}.'
        )


def training_dependency_versions() -> dict[str, str]:
    """Return installed training-library versions without importing them."""
    versions = {}
    for package in RL_DEPENDENCY_REQUIREMENTS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return versions


def validate_training_dependency_versions(
    versions: dict[str, str] | None = None,
) -> None:
    """Reject untested RL dependency combinations before importing them."""
    installed_versions = (
        versions if versions is not None else training_dependency_versions()
    )
    problems = []

    for package, requirement in RL_DEPENDENCY_REQUIREMENTS.items():
        installed = installed_versions.get(package, 'not installed')
        try:
            is_supported = Version(installed) in SpecifierSet(requirement)
        except InvalidVersion:
            is_supported = False
        if not is_supported:
            problems.append(
                f'{package} {installed!r} does not satisfy {requirement}'
            )

    if problems:
        details = '\n'.join(f'  - {problem}' for problem in problems)
        raise RuntimeError(
            'Incompatible GRPO dependency versions:\n'
            f'{details}\n'
            'Install or repair the tested CUDA/GRPO stack with:\n'
            f'  bash {RL_SETUP_SCRIPT}'
        )


def load_training_dependencies():
    """Import the heavy training stack after checking the tested versions."""
    validate_training_dependency_versions()

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except (ImportError, RuntimeError) as exc:
        versions = training_dependency_versions()
        raise RuntimeError(
            'Could not import the GRPO training dependencies. '
            f'Installed versions: {versions}. The installed PyTorch files may '
            f'be inconsistent; repair the environment with: '
            f'bash {RL_SETUP_SCRIPT}'
        ) from exc

    return Dataset, AutoTokenizer, GRPOConfig, GRPOTrainer


def validate_training_environment() -> None:
    """Run the launcher preflight once before starting distributed workers."""
    versions = training_dependency_versions()
    print(f'[LOG] Checking GRPO dependencies: {versions}', flush=True)
    _, _, grpo_config_class, grpo_trainer_class = load_training_dependencies()
    validate_completion_logging_support(grpo_config_class, grpo_trainer_class)
    print('[LOG] GRPO dependency preflight passed.', flush=True)


def main() -> None:
    args = parse_args()
    config = TrainConfig(args.config_file)
    rl_settings = load_rl_settings(args.config_file)
    reward_config = load_reward_config(args.config_file)

    train_path = config.data.training_dataset_path
    output_dir = Path(config.data.output_directory)
    if os.environ.get('RANK', '0') == '0':
        print(f'[LOG] GRPO output directory: {output_dir.resolve()}', flush=True)
    run_name = config.wandb.run_name
    if os.environ.get('RANK', '0') == '0':
        print(f'[LOG] GRPO run name: {run_name}', flush=True)
    max_rows = config.data.max_rows

    validate_generation_batch(config, rl_settings.num_generations)

    with Path(config.data.extraction_template_path).open(
        'r', encoding='utf-8'
    ) as template_file:
        template = json.load(template_file)

    rows = load_jsonl(train_path)
    records = build_rl_records(rows, template, config.data.dataset_type)
    records = select_rows(records, max_rows, rl_settings.seed)
    training_records, validation_records = split_records(
        records,
        config.data.validation_split,
        rl_settings.seed,
    )
    if not training_records:
        raise ValueError('The RL training dataset is empty.')

    print(
        '[LOG] Prepared GRPO records: '
        f'train={len(training_records)}, '
        f'validation={len(validation_records) if validation_records else 0}, '
        f'dataset_type={config.data.dataset_type}.',
        flush=True,
    )
    if args.dry_run:
        print(
            '[LOG] Dry run complete. First prompt:\n'
            f'{training_records[0]["prompt"]}',
            flush=True,
        )
        return

    # Heavy training dependencies stay local so dataset helpers remain testable.
    Dataset, AutoTokenizer, GRPOConfig, GRPOTrainer = (
        load_training_dependencies()
    )

    validate_completion_logging_support(GRPOConfig, GRPOTrainer)

    train_dataset = Dataset.from_list(training_records)
    eval_dataset = (
        Dataset.from_list(validation_records) if validation_records else None
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir.mkdir(parents=True, exist_ok=True)
    report_to = ['wandb'] if config.wandb.use_wandb else ['none']
    if config.wandb.use_wandb:
        os.environ.setdefault('WANDB_PROJECT', config.wandb.project)
        if config.wandb.entity:
            os.environ.setdefault('WANDB_ENTITY', config.wandb.entity)
    reward_funcs = build_reward_functions(reward_config)
    reward_weights = build_reward_weights(reward_config)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        warmup_steps=config.optimization.warmup_steps,
        per_device_train_batch_size=config.optimization.batch_size,
        per_device_eval_batch_size=config.optimization.batch_size,
        gradient_accumulation_steps=config.optimization.accumulate_grad_batches,
        num_train_epochs=config.optimization.max_epochs,
        max_completion_length=rl_settings.max_completion_length,
        num_generations=rl_settings.num_generations,
        temperature=rl_settings.temperature,
        beta=rl_settings.beta,
        reward_weights=reward_weights,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        logging_steps=rl_settings.logging_steps,
        log_completions=True,
        num_completions_to_print=rl_settings.num_completions_to_print,
        log_unique_prompts=True,
        save_strategy='steps',
        save_steps=config.checkpointing.save_steps,
        save_total_limit=2,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=rl_settings.eval_steps if eval_dataset is not None else None,
        report_to=report_to,
        seed=rl_settings.seed,
        use_vllm=rl_settings.use_vllm,
    )

    trainer = GRPOTrainer(
        model=config.model.model_name,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    requested_resume = config.checkpointing.resume_from
    resume_checkpoint = resolve_resume_checkpoint(output_dir, requested_resume)
    if requested_resume == 'last' and resume_checkpoint is None:
        print('[LOG] No GRPO checkpoint found; starting fresh.', flush=True)
    else:
        print(f'[LOG] GRPO resume checkpoint: {resume_checkpoint}', flush=True)

    print(
        '[LOG] Starting GRPO training: '
        f'train={len(train_dataset)}, '
        f'validation={len(eval_dataset) if eval_dataset is not None else 0}, '
        f'reward_weights={reward_weights}',
        flush=True,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_dir = output_dir / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    print(f'[LOG] Final GRPO model saved to {final_dir}.', flush=True)


if __name__ == '__main__':
    main()
