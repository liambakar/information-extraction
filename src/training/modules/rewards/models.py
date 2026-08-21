from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and field-level outcomes used by the reward evaluator."""

    valid_json: float = 0.1
    invalid_json: float = -1.0

    json_validity_weight: float = 1.0
    schema_weight: float = 0.1
    extraction_weight: float = 0.7
    hallucination_weight: float = 0.2

    correct_value: float = 1.0
    correct_null: float = 0.25
    omission: float = -0.5
    hallucination: float = -1.0
    incorrect_value: float = -0.75

    case_sensitive: bool = False
    list_order_sensitive: bool = False


@dataclass(frozen=True)
class RewardResult:
    """Reward components and diagnostics for one prediction."""

    total: float
    json_validity: float
    schema: float
    extraction: float
    hallucination: float

    correct_values: int = 0
    correct_nulls: int = 0
    omissions: int = 0
    incorrect_values: int = 0
    hallucinated_values: int = 0
    ground_truth_fields: int = 0
    predicted_fields: int = 0


DEFAULT_REWARD_CONFIG = RewardConfig()
