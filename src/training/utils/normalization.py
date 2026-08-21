import json
import re
import unicodedata
from typing import Any


FieldPath = tuple[str, ...]


def flatten_leaves(data: dict[str, Any]) -> dict[FieldPath, Any]:
    """Flatten an arbitrarily nested dictionary into leaf-value paths."""
    leaves: dict[FieldPath, Any] = {}

    def visit(value: Any, path: FieldPath) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)))
            return

        leaves[path] = value

    visit(data, ())
    return leaves


def is_empty(value: Any) -> bool:
    """Return whether a value contains no extracted information."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return not value or all(is_empty(item) for item in value)
    if isinstance(value, dict):
        return not value or all(is_empty(item) for item in value.values())
    return False


def normalize_value(
    value: Any,
    *,
    case_sensitive: bool = False,
    list_order_sensitive: bool = False,
) -> Any:
    """Canonicalize extracted values before exact comparison."""
    if isinstance(value, str):
        normalized = unicodedata.normalize('NFKC', value)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized if case_sensitive else normalized.casefold()

    if isinstance(value, dict):
        return {
            key: normalize_value(
                child,
                case_sensitive=case_sensitive,
                list_order_sensitive=list_order_sensitive,
            )
            for key, child in sorted(value.items())
        }

    if isinstance(value, list):
        normalized = [
            normalize_value(
                item,
                case_sensitive=case_sensitive,
                list_order_sensitive=list_order_sensitive,
            )
            for item in value
        ]
        if not list_order_sensitive:
            normalized.sort(
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
            )
        return normalized

    return value


def values_match(
    prediction: Any,
    ground_truth: Any,
    *,
    case_sensitive: bool = False,
    list_order_sensitive: bool = False,
) -> bool:
    """Compare two extraction values after canonicalization."""
    return normalize_value(
        prediction,
        case_sensitive=case_sensitive,
        list_order_sensitive=list_order_sensitive,
    ) == normalize_value(
        ground_truth,
        case_sensitive=case_sensitive,
        list_order_sensitive=list_order_sensitive,
    )
