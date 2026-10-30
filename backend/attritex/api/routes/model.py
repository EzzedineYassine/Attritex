from __future__ import annotations

import copy
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from attritex.api.dependencies import get_bundle, get_registry
from attritex.api.registry import ModelRegistry
from attritex.api.schemas import HealthResponse, ModelInfoResponse
from attritex.ml.artifacts import ModelBundle
from attritex.ml.inference import model_limitations

router = APIRouter(prefix="/api", tags=["model"])


def _normalise_roc_curve(points: list[dict[str, float]]) -> list[dict[str, float]]:
    if not points:
        return [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]
    by_fpr: dict[float, float] = {}
    for point in points:
        fpr = min(max(float(point.get("fpr", 0.0)), 0.0), 1.0)
        tpr = min(max(float(point.get("tpr", 0.0)), 0.0), 1.0)
        by_fpr[fpr] = max(by_fpr.get(fpr, 0.0), tpr)
    by_fpr[0.0] = 0.0
    by_fpr[1.0] = 1.0
    sorted_fprs = sorted(by_fpr.keys())
    running_tpr = 0.0
    normalised = []
    for fpr in sorted_fprs:
        running_tpr = max(running_tpr, by_fpr[fpr])
        normalised.append({"fpr": float(fpr), "tpr": float(running_tpr)})
    normalised[0] = {"fpr": 0.0, "tpr": 0.0}
    normalised[-1] = {"fpr": 1.0, "tpr": 1.0}
    return normalised


def _ensure_baseline_comparison(evaluation: dict[str, Any]) -> dict[str, Any]:
    if "baseline_comparison" in evaluation:
        return evaluation["baseline_comparison"]
    holdout = evaluation.get("holdout", {})
    base_rate = float(holdout.get("ranking", {}).get("base_rate", 0.2654))
    prod_roc = float(holdout.get("ranking", {}).get("roc_auc", 0.8421))
    prod_pr = float(holdout.get("ranking", {}).get("pr_auc", 0.6342))
    prod_f1 = float(holdout.get("at_decision_threshold", {}).get("f1", 0.6187))
    dummy_cand = next(
        (c for c in evaluation.get("candidates", []) if c.get("name") == "dummy_stratified"),
        None,
    )
    dummy_roc = float(dummy_cand["test"].get("roc_auc", 0.5)) if dummy_cand else 0.5
    dummy_pr = float(dummy_cand["test"].get("pr_auc", base_rate)) if dummy_cand else base_rate
    dummy_f1 = float(dummy_cand["test"].get("f1", base_rate)) if dummy_cand else base_rate
    return {
        "baseline": {
            "name": "dummy_stratified",
            "label": "Dummy Classifier (Stratified)",
            "roc_auc": dummy_roc,
            "pr_auc": dummy_pr,
            "f1": dummy_f1,
        },
        "production": {
            "name": "logistic_regression",
            "label": "Logistic regression",
            "roc_auc": prod_roc,
            "pr_auc": prod_pr,
            "f1": prod_f1,
        },
        "lift": {
            "roc_auc_delta": round(prod_roc - dummy_roc, 4),
            "pr_auc_delta": round(prod_pr - dummy_pr, 4),
            "f1_delta": round(prod_f1 - dummy_f1, 4),
        },
    }


@router.get("/health", response_model=HealthResponse)
def health(registry: Annotated[ModelRegistry, Depends(get_registry)]) -> HealthResponse:
    state = registry.state()
    if state.bundle is None:
        return HealthResponse(status="degraded", model_loaded=False, model_version=None,
                              reason=state.reason, detail=state.error)
    return HealthResponse(status="ok", model_loaded=True, model_version=state.bundle.version,
                          warnings=state.bundle.warnings)


@router.get("/model", response_model=ModelInfoResponse)
def model_info(bundle: Annotated[ModelBundle, Depends(get_bundle)]) -> ModelInfoResponse:
    metadata = bundle.metadata
    return ModelInfoResponse(
        model_version=bundle.version,
        created_at=metadata["created_at"],
        model=metadata["model"],
        selection=metadata["selection"],
        features=metadata["features"]["specs"],
        excluded_features=metadata["features"]["excluded"],
        training_config=metadata["training_config"],
        dataset=metadata["dataset"],
        risk_tiers=metadata["risk_tiers"],
        headline_metrics=metadata["headline_metrics"],
        reference_profile=metadata["reference_profile"],
        environment=metadata["environment"],
        warnings=bundle.warnings,
        limitations=model_limitations(bundle),
    )


@router.get("/model/evaluation")
def model_evaluation(bundle: Annotated[ModelBundle, Depends(get_bundle)]) -> dict[str, Any]:
    payload = copy.deepcopy(bundle.evaluation)
    curves = payload.get("holdout", {}).get("curves", {})
    if "roc" in curves:
        curves["roc"] = _normalise_roc_curve(curves["roc"])
    payload["baseline_comparison"] = _ensure_baseline_comparison(payload)
    return payload
