import json
from typing import Any

from src.training.modules.rewards.extraction import score_extraction
from src.training.modules.rewards.models import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    RewardResult,
)
from src.training.modules.rewards.structure import score_schema


def parse_json_object(value: Any, *, name: str) -> dict[str, Any]:
    """Parse a JSON object while accepting already-decoded dictionaries."""
    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        raise ValueError(f'{name} must be a JSON object or a JSON string.')

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{name} is not valid JSON.') from exc

    if not isinstance(parsed, dict):
        raise ValueError(f'{name} must decode to a JSON object.')

    return parsed


def evaluate_reward(
    prediction: Any,
    ground_truth: Any,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
    schema_section: str | None = None,
) -> RewardResult:
    """Evaluate one prediction and return its reward with diagnostics."""
    truth = parse_json_object(ground_truth, name='ground_truth')

    try:
        parsed_prediction = parse_json_object(prediction, name='prediction')
    except ValueError:
        return RewardResult(
            total=config.json_validity_weight * config.invalid_json,
            json_validity=config.invalid_json,
            schema=0.0,
            extraction=0.0,
            hallucination=0.0,
        )

    schema_score = score_schema(parsed_prediction, section=schema_section)
    extraction = score_extraction(parsed_prediction, truth, config)

    total = (
        config.json_validity_weight * config.valid_json
        + config.schema_weight * schema_score
        + config.extraction_weight * extraction.extraction
        + config.hallucination_weight * extraction.hallucination
    )

    return RewardResult(
        total=total,
        json_validity=config.valid_json,
        schema=schema_score,
        extraction=extraction.extraction,
        hallucination=extraction.hallucination,
        correct_values=extraction.correct_values,
        correct_nulls=extraction.correct_nulls,
        omissions=extraction.omissions,
        incorrect_values=extraction.incorrect_values,
        hallucinated_values=extraction.hallucinated_values,
        ground_truth_fields=extraction.ground_truth_fields,
        predicted_fields=extraction.predicted_fields,
    )
