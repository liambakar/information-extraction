"""Check whether extracted claims follow from the user's utterance."""

from functools import lru_cache
from typing import Any

from src.training.utils.normalization import flatten_leaves, is_empty


NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


def _claim(path: tuple[str, ...], value: Any) -> str | None:
    """Turn one extraction value into an NLI hypothesis."""
    if not isinstance(value, str):
        value = str(value)

    field = '.'.join(path)
    templates = {
        'activity.type': 'The speaker did {} activity.',
        'activity.keywords': 'The speaker did {}.',
        'activity.duration': 'The activity lasted {}.',
        'activity.location': 'The activity happened at {}.',
        'activity.date': 'The activity happened on {}.',
        'food.consumed_items': 'The speaker ate or drank {}.',
        'food.missing_items': 'The speaker did not have {}.',
        'food.amount': 'The speaker consumed {}.',
        'mood.description': 'The speaker felt {}.',
        'mood.classification': "The speaker's mood was {}.",
        'symptom.keywords': 'The speaker feels {}.',
        'symptom.description': 'The speaker experienced {}.',
        'treatment.name': 'The speaker took or was advised to take {}.',
        'treatment.dosage': 'The treatment dosage was {}.',
        'treatment.status': 'The treatment was {}.',
    }
    template = templates.get(field)
    return template.format(value) if template else None


def prediction_claims(prediction: dict[str, Any]) -> list[str]:
    """Generate one claim per populated scalar or list item in known fields."""
    claims = []
    for path, value in flatten_leaves(prediction).items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not is_empty(item):
                claim = _claim(path, item)
                if claim is not None:
                    claims.append(claim)
    return claims


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    return tokenizer, model, torch


def classify_claims(utterance: str, claims: list[str]) -> list[tuple[str, float]]:
    """Return the most likely NLI label and its confidence for each claim."""
    if not claims:
        return []

    tokenizer, model, torch = _model()
    labels = []
    for start in range(0, len(claims), 16):
        batch = claims[start : start + 16]
        inputs = tokenizer(
            [utterance] * len(batch), batch, padding=True, truncation=True,
            return_tensors='pt',
        )
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1)
        for scores in probabilities:
            index = int(scores.argmax())
            labels.append((model.config.id2label[index].lower(), float(scores[index])))
    return labels


def score_semantic_hallucinations(
    prediction: dict[str, Any], utterance: str, penalty: float,
    confidence_threshold: float,
) -> tuple[float, int]:
    """Penalize confident contradiction or lack of support, not paraphrases."""
    claims = prediction_claims(prediction)
    if not claims:
        return 0.0, 0

    results = classify_claims(utterance, claims)
    confident = [
        label if confidence >= confidence_threshold else 'uncertain'
        for label, confidence in results
    ]
    contradictions = confident.count('contradiction')
    unsupported = confident.count('neutral')
    score = penalty * (contradictions + 0.5 * unsupported) / len(claims)
    return score, contradictions + unsupported
