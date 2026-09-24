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
    device = torch.device(
        'cuda', torch.cuda.current_device()
    ) if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def classify_claim_pairs(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, float]]:
    """Classify premise/hypothesis pairs in shared, deduplicated batches."""
    if not pairs:
        return []

    tokenizer, model, torch, device = _model()
    unique_pairs = list(dict.fromkeys(pairs))
    predictions = {}

    for start in range(0, len(unique_pairs), 32):
        batch = unique_pairs[start : start + 32]
        premises, hypotheses = zip(*batch)
        inputs = tokenizer(
            list(premises), list(hypotheses), padding=True, truncation=True,
            return_tensors='pt',
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1)
        confidences, indices = probabilities.max(dim=-1)
        for pair, index, confidence in zip(
            batch,
            indices.cpu().tolist(),
            confidences.float().cpu().tolist(),
        ):
            predictions[pair] = (
                model.config.id2label[index].lower(),
                confidence,
            )

    return [predictions[pair] for pair in pairs]


def classify_claims(utterance: str, claims: list[str]) -> list[tuple[str, float]]:
    """Return the most likely NLI label and its confidence for each claim."""
    return classify_claim_pairs([(utterance, claim) for claim in claims])


def score_semantic_hallucination_batch(
    predictions: list[dict[str, Any]],
    utterances: list[str | None],
    penalty: float,
    confidence_threshold: float,
) -> list[tuple[float, int]]:
    """Score a completion batch with one set of NLI inference calls."""
    if len(predictions) != len(utterances):
        raise ValueError('predictions and utterances must have the same length.')

    claims_by_prediction = [
        prediction_claims(prediction) if utterance is not None else []
        for prediction, utterance in zip(predictions, utterances)
    ]
    pairs = [
        (utterance, claim)
        for utterance, claims in zip(utterances, claims_by_prediction)
        if utterance is not None
        for claim in claims
    ]
    classifications = iter(classify_claim_pairs(pairs))
    scores = []

    for claims in claims_by_prediction:
        results = [next(classifications) for _ in claims]
        if not results:
            scores.append((0.0, 0))
            continue

        confident = [
            label if confidence >= confidence_threshold else 'uncertain'
            for label, confidence in results
        ]
        contradictions = confident.count('contradiction')
        unsupported = confident.count('neutral')
        score = penalty * (contradictions + 0.5 * unsupported) / len(claims)
        scores.append((score, contradictions + unsupported))

    return scores


def score_semantic_hallucinations(
    prediction: dict[str, Any], utterance: str, penalty: float,
    confidence_threshold: float,
) -> tuple[float, int]:
    """Penalize confident contradiction or lack of support, not paraphrases."""
    return score_semantic_hallucination_batch(
        [prediction], [utterance], penalty, confidence_threshold,
    )[0]
