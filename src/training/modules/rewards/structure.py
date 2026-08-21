import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


def is_duration(value: str) -> bool:
    """Validate ISO, natural-language, and clock-style durations."""
    if not value:
        return True

    # ISO 8601: PT30M, PT1H30M, P2D
    iso_pattern = (
        r'P(?=\d|T\d)'
        r'(?:\d+D)?'
        r'(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?'
    )
    if re.fullmatch(iso_pattern, value.upper()):
        return True

    # Natural language: 30 minutes, 1.5 hours, 1 hour 20 minutes
    unit = (
        r'\d+(?:\.\d+)?\s*'
        r'(?:seconds?|secs?|s|minutes?|mins?|m|'
        r'hours?|hrs?|h|days?|d)'
    )
    if re.fullmatch(rf'{unit}(?:\s+{unit})*', value.lower()):
        return True

    # Clock style: 30:00, 01:30:00
    return bool(re.fullmatch(r'(?:\d{1,3}:)?[0-5]?\d:[0-5]\d', value))


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Activity(SchemaModel):
    type: (
        Literal[
            '',
            'physical',
            'social',
            'intellectual',
            'productive',
        ]
        | None
    )

    keywords: list[str]
    duration: str | None
    location: str | None
    date: datetime | str | None

    @field_validator('duration')
    @classmethod
    def validate_duration(cls, value):
        if value is not None and not is_duration(value):
            raise ValueError('Invalid duration.')
        return value

    @field_validator('date')
    @classmethod
    def validate_date(cls, value):
        if value in (None, ''):
            return value

        if isinstance(value, datetime):
            return value

        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('Invalid date-time.') from exc


class Food(SchemaModel):
    consumed_items: list[str]
    missing_items: list[str]
    amount: list[str]


class Mood(SchemaModel):
    description: str | None
    classification: (
        Literal[
            '',
            'positive',
            'neutral',
            'negative',
        ]
        | None
    )


class Symptom(SchemaModel):
    keywords: list[str]
    description: str | None


class Treatment(SchemaModel):
    name: str | None
    dosage: str | None
    status: (
        Literal[
            '',
            'taken',
            'recommended',
            'postponed',
            'missed',
        ]
        | None
    )


class HealthLog(SchemaModel):
    activity: Activity
    food: Food
    mood: Mood
    symptom: Symptom
    treatment: Treatment


SECTION_MODELS = {
    'activity': Activity,
    'food': Food,
    'mood': Mood,
    'symptom': Symptom,
    'treatment': Treatment,
}


def field_is_valid(
    model: type[BaseModel],
    field: str,
    value: Any,
) -> bool:
    """Validate one leaf field independently."""
    try:
        model.model_validate(
            {
                name: value if name == field else empty_value(info.annotation)
                for name, info in model.model_fields.items()
            }
        )
        return True
    except ValidationError:
        return False


def empty_value(annotation: Any) -> Any:
    """Return an empty value appropriate for a leaf type."""
    if getattr(annotation, '__origin__', None) is list:
        return []
    return None


def score_schema(model_output: Any, section: str | None = None) -> float:
    """Score structural schema correctness.

    Returns:
        0.0-1.0 based on the percentage of correct fields"""
    if not isinstance(model_output, dict):
        return 0.0

    if section:
        model = SECTION_MODELS.get(section)
        if model is None:
            raise ValueError(f'Unknown schema section: {section!r}.')

        expected_fields = set(model.model_fields)
        earned = int(set(model_output) == expected_fields)
        earned += sum(
            field in model_output
            and field_is_valid(model, field, model_output[field])
            for field in model.model_fields
        )
        return earned / (1 + len(model.model_fields))

    earned = 0
    possible = 1 + sum(
        2 + len(model.model_fields) for model in SECTION_MODELS.values()
    )

    def score(condition: bool) -> None:
        nonlocal earned
        earned += int(condition)

    # Require exactly the expected top-level structure.
    score(set(model_output) == set(SECTION_MODELS))

    for section, model in SECTION_MODELS.items():
        section_data = model_output.get(section)

        # Every section must exist and be an object.
        is_section = isinstance(section_data, dict)
        score(is_section)

        expected_fields = set(model.model_fields)

        # Every leaf field must exist, with no extra fields.
        score(is_section and set(section_data) == expected_fields)

        # Validate each leaf value.
        for field in model.model_fields:
            score(
                is_section
                and field in section_data
                and field_is_valid(
                    model,
                    field,
                    section_data[field],
                )
            )

    return earned / possible
