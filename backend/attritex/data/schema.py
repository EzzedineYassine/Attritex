from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FeatureKind = Literal["numeric", "categorical"]

YES_NO = ("No", "Yes")
INTERNET_ADDON = ("No", "Yes", "No internet service")
NO_INTERNET = "No internet service"
NO_PHONE = "No phone service"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
TARGET_VALUES = ("No", "Yes")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    label: str
    group: str
    description: str
    allowed_values: tuple[str, ...] = field(default_factory=tuple)
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    nullable: bool = False
    integer: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "label": self.label,
            "group": self.group,
            "description": self.description,
            "allowed_values": list(self.allowed_values),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "nullable": self.nullable,
            "integer": self.integer,
        }


INTERNET_ADDON_FEATURES = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

ALL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("gender", "categorical", "Gender", "Profile",
                "Customer gender as recorded in the source data.", ("Female", "Male")),
    FeatureSpec("SeniorCitizen", "categorical", "Senior citizen", "Profile",
                "Whether the customer is 65 or older.", YES_NO),
    FeatureSpec("Partner", "categorical", "Has partner", "Profile",
                "Whether the customer has a partner.", YES_NO),
    FeatureSpec("Dependents", "categorical", "Has dependents", "Profile",
                "Whether the customer has dependents.", YES_NO),
    FeatureSpec("tenure", "numeric", "Tenure", "Account",
                "Number of months the customer has been with the company.",
                minimum=0, maximum=120, unit="months", integer=True),
    FeatureSpec("Contract", "categorical", "Contract", "Account",
                "Current contract term.", ("Month-to-month", "One year", "Two year")),
    FeatureSpec("PaperlessBilling", "categorical", "Paperless billing", "Account",
                "Whether the customer receives paperless bills.", YES_NO),
    FeatureSpec("PaymentMethod", "categorical", "Payment method", "Account",
                "How the customer pays.",
                ("Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)")),
    FeatureSpec("PhoneService", "categorical", "Phone service", "Services",
                "Whether the customer has phone service.", YES_NO),
    FeatureSpec("MultipleLines", "categorical", "Multiple lines", "Services",
                "Whether the customer has multiple phone lines.", ("No", "Yes", NO_PHONE)),
    FeatureSpec("InternetService", "categorical", "Internet service", "Services",
                "Internet service type.", ("DSL", "Fiber optic", "No")),
    FeatureSpec("OnlineSecurity", "categorical", "Online security", "Services",
                "Online security add-on.", INTERNET_ADDON),
    FeatureSpec("OnlineBackup", "categorical", "Online backup", "Services",
                "Online backup add-on.", INTERNET_ADDON),
    FeatureSpec("DeviceProtection", "categorical", "Device protection", "Services",
                "Device protection add-on.", INTERNET_ADDON),
    FeatureSpec("TechSupport", "categorical", "Tech support", "Services",
                "Premium technical support add-on.", INTERNET_ADDON),
    FeatureSpec("StreamingTV", "categorical", "Streaming TV", "Services",
                "Streaming TV add-on.", INTERNET_ADDON),
    FeatureSpec("StreamingMovies", "categorical", "Streaming movies", "Services",
                "Streaming movies add-on.", INTERNET_ADDON),
    FeatureSpec("MonthlyCharges", "numeric", "Monthly charges", "Billing",
                "Amount currently charged per month.", minimum=0, maximum=1000, unit="USD"),
    FeatureSpec("TotalCharges", "numeric", "Total charges", "Billing",
                "Total amount charged to date. Blank for customers in their first billing cycle.",
                minimum=0, maximum=100_000, unit="USD", nullable=True),
)

EXCLUDED_FEATURES: dict[str, str] = {
    "gender": "Protected attribute excluded from modelling by policy.",
}

MODEL_FEATURES: tuple[FeatureSpec, ...] = tuple(
    spec for spec in ALL_FEATURES if spec.name not in EXCLUDED_FEATURES
)

FEATURES_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in ALL_FEATURES}


def model_feature_names() -> list[str]:
    return [spec.name for spec in MODEL_FEATURES]


def numeric_feature_names() -> list[str]:
    return [spec.name for spec in MODEL_FEATURES if spec.kind == "numeric"]


def categorical_feature_names() -> list[str]:
    return [spec.name for spec in MODEL_FEATURES if spec.kind == "categorical"]
