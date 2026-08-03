import json
import re
from datetime import datetime
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, RootModel, create_model


TEMPLATE_TYPES = {
    'string': str,
    'date-time': datetime,
    # NuExtract uses loose ISO-8601-like duration strings (including values
    # such as PT1D), so timedelta validation would reject existing outputs.
    'duration': str,
    'integer': int,
    'number': float,
    'boolean': bool,
}


def load_template_model(
    template_path: str | Path,
    dataset_type: Literal['full', 'partial'],
) -> type[BaseModel]:
    with Path(template_path).open(encoding='utf-8') as template_file:
        template = json.load(template_file)

    if not isinstance(template, dict) or not template:
        raise ValueError('The extraction template must be a non-empty JSON object.')

    if dataset_type == 'full':
        return template_to_pydantic_model(template, 'Extraction')
    if dataset_type != 'partial':
        raise ValueError(f'Unsupported dataset type: {dataset_type!r}')

    category_models = [
        template_to_pydantic_model(category_template, category.title())
        for category, category_template in template.items()
    ]
    category_union = reduce(or_, category_models)
    partial_model = RootModel[category_union]
    partial_model.__name__ = 'PartialExtraction'
    partial_model.__qualname__ = 'PartialExtraction'
    return partial_model


def template_to_pydantic_model(
    template: dict[str, Any],
    model_name: str = 'Extraction',
) -> type[BaseModel]:
    """Convert a NuExtract JSON template into a strict Pydantic model.

    A one-item array is a list marker (for example, ``["string"]``), while an
    array with multiple scalar values is an enum. Scalar and enum fields remain
    required but accept null, matching the extraction data's empty-value format.
    """
    if not isinstance(template, dict) or not template:
        raise ValueError(f'{model_name} must be a non-empty JSON object.')

    field_definitions: dict[str, Any] = {
        field_name: (_template_value_to_type(value, model_name, field_name), ...)
        for field_name, value in template.items()
    }
    return create_model(
        _safe_model_name(model_name),
        __config__=ConfigDict(extra='forbid'),
        **field_definitions,
    )


def _template_value_to_type(value: Any, parent_name: str, field_name: str):
    child_name = f'{parent_name}_{field_name}'

    if isinstance(value, dict):
        return template_to_pydantic_model(value, child_name)

    if isinstance(value, list):
        if not value:
            raise ValueError(
                f'{parent_name}.{field_name} cannot use an empty template array.'
            )
        if len(value) == 1:
            return list[_list_item_type(value[0], child_name)]
        if not all(isinstance(item, (str, int, float, bool)) for item in value):
            raise ValueError(
                f'{parent_name}.{field_name} enum values must be JSON scalars.'
            )
        return Literal[tuple(value)] or None

    if isinstance(value, str) and value in TEMPLATE_TYPES:
        return TEMPLATE_TYPES[value] | None

    raise ValueError(
        f'Unsupported template value for {parent_name}.{field_name}: {value!r}'
    )


def _list_item_type(value: Any, model_name: str):
    if isinstance(value, str) and value in TEMPLATE_TYPES:
        return TEMPLATE_TYPES[value]
    if isinstance(value, dict):
        return template_to_pydantic_model(value, model_name)

    raise ValueError(f'Unsupported list item marker for {model_name}: {value!r}')


def _safe_model_name(name: str) -> str:
    safe_name = re.sub(r'\W|^(?=\d)', '_', name)
    return safe_name or 'Extraction'
