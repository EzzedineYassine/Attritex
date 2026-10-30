from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RiskTier = Literal["low", "medium", "high"]
YesNo = Literal["No", "Yes"]
SeniorCitizenValue = Literal["No", "Yes", "0", "1", 0, 1]
MultipleLinesValue = Literal["No", "Yes", "No phone service"]
InternetServiceValue = Literal["DSL", "Fiber optic", "No"]
InternetAddonValue = Literal["No", "Yes", "No internet service"]
ContractValue = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodValue = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    customerID: str | None = Field(default=None, max_length=64, title="Customer ID")
    SeniorCitizen: SeniorCitizenValue = Field(..., title="Senior citizen")
    Partner: YesNo = Field(..., title="Has partner")
    Dependents: YesNo = Field(..., title="Has dependents")
    tenure: int = Field(..., ge=0, le=120, strict=False, title="Tenure")
    Contract: ContractValue = Field(..., title="Contract")
    PaperlessBilling: YesNo = Field(..., title="Paperless billing")
    PaymentMethod: PaymentMethodValue = Field(..., title="Payment method")
    PhoneService: YesNo = Field(..., title="Phone service")
    MultipleLines: MultipleLinesValue = Field(..., title="Multiple lines")
    InternetService: InternetServiceValue = Field(..., title="Internet service")
    OnlineSecurity: InternetAddonValue = Field(..., title="Online security")
    OnlineBackup: InternetAddonValue = Field(..., title="Online backup")
    DeviceProtection: InternetAddonValue = Field(..., title="Device protection")
    TechSupport: InternetAddonValue = Field(..., title="Tech support")
    StreamingTV: InternetAddonValue = Field(..., title="Streaming TV")
    StreamingMovies: InternetAddonValue = Field(..., title="Streaming movies")
    MonthlyCharges: float = Field(..., ge=0.0, le=1000.0, title="Monthly charges")
    TotalCharges: float | None = Field(default=None, ge=0.0, le=100000.0, title="Total charges")


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
    attribution_score: float | None = None
    direction: Literal["risk_increasing", "risk_reducing"] | None = None


class LocalAttribution(BaseModel):
    top_risk_increasing: list[RiskFactor]
    top_risk_reducing: list[RiskFactor]


class PredictionResponse(BaseModel):
    customer_id: str | None
    churn_probability: float
    expected_revenue_at_risk: float
    risk_tier: RiskTier
    risk_tier_label: str
    flagged: bool
    decision_threshold: float
    tier_cutoffs: TierCutoffsModel
    tier_context: TierContext | None
    factors: list[RiskFactor]
    local_attribution: LocalAttribution
    imputed_fields: list[str]
    warnings: list[str]
    limitations: list[str]
    model_version: str


class BatchRow(BaseModel):
    row: int
    customer_id: str | None
    status: Literal["scored", "invalid"]
    churn_probability: float | None
    expected_revenue_at_risk: float | None = None
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
    predicted_churn_rate: float
    baseline_churn_rate: float
    expected_churners: float
    total_arr_at_risk: float
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
