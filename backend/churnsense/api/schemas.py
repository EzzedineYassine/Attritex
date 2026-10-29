from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from churnsense.data.schema import ID_COLUMN, MODEL_FEATURES, FeatureSpec

RiskTier = Literal["low", "medium", "high"]


def _field_definition(spec: FeatureSpec) -> tuple[Any, Any]:
    if spec.name == "SeniorCitizen":
        annotation: Any = Literal["No", "Yes", "0", "1"] | int
        return annotation, Field(..., title=spec.label, description=spec.description)
    if spec.kind == "categorical":
        annotation: Any = Literal[spec.allowed_values]
        return annotation, Field(..., title=spec.label, description=spec.description)
    numeric_type: Any = int if spec.integer else float
    annotation = numeric_type | None if spec.nullable else numeric_type
    default = None if spec.nullable else ...
    return annotation, Field(default, title=spec.label, description=spec.description,
                             ge=spec.minimum, le=spec.maximum)


class _CustomerBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


CustomerRecord = create_model(
    "CustomerRecord",
    __base__=_CustomerBase,
    **{
        ID_COLUMN: (str | None, Field(None, max_length=64, title="Customer ID")),
        **{spec.name: _field_definition(spec) for spec in MODEL_FEATURES},
    },
)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    reason: str | None = None
    detail: str | None = None
    warnings: list[str] = []


class TierCutoffsModel(BaseModel):
    medium: float
    high: float
    medium_recall_target: float


class TierContext(BaseModel):
    tier: RiskTier
    customers: int
    share_of_customers: float
    observed_churn_rate: float
    share_of_churners: float


class RiskFactor(BaseModel):
    feature: str
    label: str
    customer_value: str
    reference_value: str
    probability_if_reference: float
    effect: float


class PredictionResponse(BaseModel):
    customer_id: str | None
    churn_probability: float
    risk_tier: RiskTier
    risk_tier_label: str
    flagged: bool
    decision_threshold: float
    tier_cutoffs: TierCutoffsModel
    tier_context: TierContext | None
    factors: list[RiskFactor]
    imputed_fields: list[str]
    warnings: list[str]
    limitations: list[str]
    model_version: str


class BatchRow(BaseModel):
    row: int
    customer_id: str | None
    status: Literal["scored", "invalid"]
    churn_probability: float | None
    risk_tier: RiskTier | None
    flagged: bool
    issues: str
    imputed_fields: list[str]


class HistogramBin(BaseModel):
    lower: float
    upper: float
    count: int


class BatchSummary(BaseModel):
    input_rows: int
    scored_rows: int
    invalid_rows: int
    flagged_rows: int
    mean_probability: float | None
    expected_churners: float
    tier_counts: dict[RiskTier, int]
    histogram: list[HistogramBin]


class BatchPredictionResponse(BaseModel):
    model_version: str
    decision_threshold: float
    tier_cutoffs: TierCutoffsModel
    summary: BatchSummary
    rows: list[BatchRow]


class FeatureSpecModel(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    label: str
    group: str
    description: str
    allowed_values: list[str]
    minimum: float | None
    maximum: float | None
    unit: str | None
    nullable: bool
    integer: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    created_at: str
    model: dict[str, Any]
    selection: dict[str, Any]
    features: list[FeatureSpecModel]
    excluded_features: list[dict[str, str]]
    training_config: dict[str, Any]
    dataset: dict[str, Any]
    risk_tiers: dict[str, Any]
    headline_metrics: dict[str, Any]
    reference_profile: dict[str, Any]
    environment: dict[str, Any]
    warnings: list[str]
    limitations: list[str]
