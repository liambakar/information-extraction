from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and field-level outcomes used by the reward evaluator."""

    valid_json: float
    invalid_json: float

    json_validity_weight: float
    schema_weight: float
    extraction_weight: float
    hallucination_weight: float

    correct_value: float
    correct_null: float
    omission: float
    hallucination: float
    incorrect_value: float

    case_sensitive: bool
    list_order_sensitive: bool


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


DEFAULT_REWARD_CONFIG = RewardConfig(
    valid_json=0.1,
    invalid_json=-1.0,
    json_validity_weight=1.0,
    schema_weight=0.1,
    extraction_weight=0.7,
    hallucination_weight=0.2,
    correct_value=1.0,
    correct_null=0.25,
    omission=-0.5,
    hallucination=-1.0,
    incorrect_value=-0.75,
    case_sensitive=False,
    list_order_sensitive=False,
)
