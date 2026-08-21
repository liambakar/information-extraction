from dataclasses import dataclass
from typing import Any

from src.training.modules.rewards.models import RewardConfig
from src.training.utils.normalization import (
    flatten_leaves,
    is_empty,
    values_match,
)


@dataclass(frozen=True)
class ExtractionScore:
    """Field matching scores and counts before component weighting."""

    extraction: float
    hallucination: float
    correct_values: int
    correct_nulls: int
    omissions: int
    incorrect_values: int
    hallucinated_values: int
    ground_truth_fields: int
    predicted_fields: int


def score_extraction(
    model_prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    config: RewardConfig,
) -> ExtractionScore:
    """Score matching, omitted, incorrect, and hallucinated leaf values."""
    prediction = flatten_leaves(model_prediction)
    truth = flatten_leaves(ground_truth)

    correct_values = 0
    correct_nulls = 0
    omissions = 0
    incorrect_values = 0
    extraction_points = 0.0

    for path, truth_value in truth.items():
        prediction_is_empty = path not in prediction or is_empty(prediction[path])

        if is_empty(truth_value):
            if prediction_is_empty:
                correct_nulls += 1
                extraction_points += config.correct_null
            continue

        if prediction_is_empty:
            omissions += 1
            extraction_points += config.omission
        elif values_match(
            prediction[path],
            truth_value,
            case_sensitive=config.case_sensitive,
            list_order_sensitive=config.list_order_sensitive,
        ):
            correct_values += 1
            extraction_points += config.correct_value
        else:
            incorrect_values += 1
            extraction_points += config.incorrect_value

    predicted_fields = sum(not is_empty(value) for value in prediction.values())
    hallucinated_values = sum(
        not is_empty(predicted_value)
        and (path not in truth or is_empty(truth[path]))
        for path, predicted_value in prediction.items()
    )

    ground_truth_fields = len(truth)
    extraction_score = (
        extraction_points / ground_truth_fields if ground_truth_fields else 0.0
    )
    hallucination_score = (
        config.hallucination * hallucinated_values / predicted_fields
        if predicted_fields
        else 0.0
    )

    return ExtractionScore(
        extraction=extraction_score,
        hallucination=hallucination_score,
        correct_values=correct_values,
        correct_nulls=correct_nulls,
        omissions=omissions,
        incorrect_values=incorrect_values,
        hallucinated_values=hallucinated_values,
        ground_truth_fields=ground_truth_fields,
        predicted_fields=predicted_fields,
    )
