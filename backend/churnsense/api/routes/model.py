from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from churnsense.api.dependencies import get_bundle, get_registry
from churnsense.api.registry import ModelRegistry
from churnsense.api.schemas import HealthResponse, ModelInfoResponse
from churnsense.ml.artifacts import ModelBundle
from churnsense.ml.inference import model_limitations

router = APIRouter(prefix="/api", tags=["model"])


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
    return bundle.evaluation
