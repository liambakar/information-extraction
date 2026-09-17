from collections.abc import Sequence
from functools import wraps
from typing import Any

from src.training.modules.rewards.evaluator import evaluate_reward, parse_json_object
from src.training.modules.rewards.extraction import score_extraction
from src.training.modules.rewards.models import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    RewardResult,
)
from src.training.modules.rewards.structure import score_schema


def _completion_text(completion: Any) -> Any:
    """Extract text from plain or conversational TRL completions."""
    if isinstance(completion, dict):
        return completion.get('content', completion)

    if (
        isinstance(completion, list)
        and completion
        and isinstance(completion[-1], dict)
    ):
        return completion[-1].get('content', completion[-1])

    return completion


def _completion_batch(completions: Any) -> list[Any]:
    if isinstance(completions, str):
        return [completions]

    if isinstance(completions, list):
        if not completions or not isinstance(completions[0], dict):
            return completions
        return [completions]

    return [completions]


def _ground_truth_batch(ground_truth: Any, size: int) -> list[Any]:
    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, str):
        values = list(ground_truth)
        if len(values) != size:
            raise ValueError(
                'ground_truth must contain one value for each completion; '
                f'got {len(values)} values for {size} completions.'
            )
        return values

    return [ground_truth] * size


def _optional_batch(value: Any, size: int) -> list[Any]:
    if value is None:
        return [None] * size
    return _ground_truth_batch(value, size)


def json_validity_reward(
    completions: Any,
    *,
    config: RewardConfig,
    **_kwargs: Any,
) -> list[float]:
    """Return the configured JSON-validity reward for each completion."""
    rewards = []
    for completion in _completion_batch(completions):
        try:
            parse_json_object(_completion_text(completion), name='completion')
        except ValueError:
            rewards.append(config.invalid_json)
        else:
            rewards.append(config.valid_json)
    return rewards


def schema_reward(
    completions: Any,
    *,
    ground_truth: Any = None,
    schema_section: Any = None,
    log_extra: Any = None,
    config: RewardConfig,
    **_kwargs: Any,
) -> list[float]:
    """Return unweighted schema scores for a batch of completions."""
    batch = _completion_batch(completions)
    sections = _optional_batch(schema_section, len(batch))
    if log_extra is not None and ground_truth is not None:
        log_extra(
            'ground_truth',
            _ground_truth_batch(ground_truth, len(batch)),
        )

    rewards = []
    for completion, section in zip(batch, sections):
        try:
            prediction = parse_json_object(
                _completion_text(completion),
                name='completion',
            )
        except ValueError:
            rewards.append(0.0)
            continue
        rewards.append(score_schema(prediction, section=section or None))
    return rewards


def extraction_reward(
    completions: Any,
    ground_truth: Any,
    *,
    config: RewardConfig,
    **_kwargs: Any,
) -> list[float]:
    """Return unweighted extraction scores for a batch of completions."""
    batch = _completion_batch(completions)
    truths = _ground_truth_batch(ground_truth, len(batch))
    return [
        evaluate_reward(_completion_text(completion), truth, config).extraction
        for completion, truth in zip(batch, truths)
    ]


def hallucination_reward(
    completions: Any,
    ground_truth: Any,
    *,
    utterance: Any = None,
    config: RewardConfig,
    **_kwargs: Any,
) -> list[float]:
    """Return unweighted hallucination penalties for a batch of completions."""
    batch = _completion_batch(completions)
    truths = _ground_truth_batch(ground_truth, len(batch))
    utterances = _optional_batch(utterance, len(batch))
    return [
        evaluate_reward(
            _completion_text(completion), truth, config, utterance=text,
        ).hallucination
        for completion, truth, text in zip(batch, truths, utterances)
    ]


def total_reward(
    completions: Any,
    ground_truth: Any,
    *,
    schema_section: Any = None,
    utterance: Any = None,
    config: RewardConfig,
    **_kwargs: Any,
) -> list[float]:
    """Return aggregate rewards for a batch of completions."""
    batch = _completion_batch(completions)
    truths = _ground_truth_batch(ground_truth, len(batch))
    sections = _optional_batch(schema_section, len(batch))
    utterances = _optional_batch(utterance, len(batch))
    return [
        evaluate_reward(
            _completion_text(completion),
            truth,
            config,
            schema_section=section or None,
            utterance=text,
        ).total
        for completion, truth, section, text in zip(
            batch, truths, sections, utterances
        )
    ]


def reward_function(
    model_prediction: Any,
    ground_truth: Any,
    config: RewardConfig,
    schema_section: str | None = None,
    utterance: str | None = None,
) -> float:
    """Compatibility wrapper returning the total reward for one prediction."""
    return evaluate_reward(
        model_prediction,
        ground_truth,
        config,
        schema_section=schema_section,
        utterance=utterance,
    ).total


def build_reward_functions(config: RewardConfig) -> list[Any]:
    """Bind a config to separately logged TRL reward functions."""

    def bind(reward_func):
        @wraps(reward_func)
        def configured_reward(*args, **kwargs):
            return reward_func(*args, config=config, **kwargs)

        return configured_reward

    return [
        bind(json_validity_reward),
        bind(schema_reward),
        bind(extraction_reward),
        bind(hallucination_reward),
    ]


def build_reward_weights(config: RewardConfig) -> list[float]:
    """Return TRL weights in the same order as ``build_reward_functions``."""
    return [
        config.json_validity_weight,
        config.schema_weight,
        config.extraction_weight,
        config.hallucination_weight,
    ]


__all__ = [
    'DEFAULT_REWARD_CONFIG',
    'RewardConfig',
    'RewardResult',
    'build_reward_functions',
    'build_reward_weights',
    'evaluate_reward',
    'extraction_reward',
    'hallucination_reward',
    'json_validity_reward',
    'reward_function',
    'schema_reward',
    'score_extraction',
    'score_schema',
    'total_reward',
]
