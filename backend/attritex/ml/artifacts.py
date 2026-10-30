from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from attritex.data.schema import model_feature_names
from attritex.ml.thresholds import TierCutoffs

ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
EVALUATION_FILENAME = "evaluation.json"


class ArtifactNotFoundError(FileNotFoundError):
    pass


class ArtifactLoadError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    evaluation: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def cutoffs(self) -> TierCutoffs:
        return TierCutoffs(**self.metadata["risk_tiers"]["cutoffs"])

    @property
    def reference_profile(self) -> dict[str, Any]:
        return self.metadata["reference_profile"]


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_json(target: Path, payload: dict) -> None:
    _atomic_write_bytes(target, json.dumps(payload, indent=2, allow_nan=False).encode("utf-8"))


def save_bundle(artifact_dir: Path, pipeline: Pipeline, metadata: dict, evaluation: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=artifact_dir, prefix=f".{MODEL_FILENAME}.")
    os.close(descriptor)
    try:
        joblib.dump(pipeline, temporary_name, compress=3)
        os.replace(temporary_name, artifact_dir / MODEL_FILENAME)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    _write_json(artifact_dir / EVALUATION_FILENAME, evaluation)
    _write_json(artifact_dir / METADATA_FILENAME, metadata)


def artifact_signature(artifact_dir: Path) -> tuple[float, ...] | None:
    paths = [artifact_dir / name for name in (MODEL_FILENAME, METADATA_FILENAME, EVALUATION_FILENAME)]
    if not all(path.exists() for path in paths):
        return None
    return tuple(path.stat().st_mtime_ns for path in paths)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactLoadError(f"Could not read {path.name}: {error}") from error


def load_bundle(artifact_dir: Path) -> ModelBundle:
    model_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    evaluation_path = artifact_dir / EVALUATION_FILENAME
    missing = [path.name for path in (model_path, metadata_path, evaluation_path) if not path.exists()]
    if missing:
        raise ArtifactNotFoundError(
            f"Model artifacts missing in {artifact_dir}: {', '.join(missing)}. "
            "Train a model with `attritex train`."
        )

    metadata = _read_json(metadata_path)
    evaluation = _read_json(evaluation_path)
    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactLoadError(
            f"Artifact schema version {metadata.get('artifact_schema_version')} is not supported; "
            f"expected {ARTIFACT_SCHEMA_VERSION}. Retrain the model."
        )
    expected_features = model_feature_names()
    if metadata.get("features", {}).get("model_inputs") != expected_features:
        raise ArtifactLoadError("Artifact feature list does not match the current schema. Retrain.")

    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        raise ArtifactLoadError(f"Could not deserialise {MODEL_FILENAME}: {error}") from error
    if not hasattr(pipeline, "predict_proba"):
        raise ArtifactLoadError("Loaded artifact is not a probabilistic classifier pipeline.")

    warnings = []
    trained_with = metadata.get("environment", {}).get("scikit_learn")
    if trained_with and trained_with != sklearn.__version__:
        warnings.append(
            f"Model trained with scikit-learn {trained_with}; running {sklearn.__version__}. "
            "Retrain to guarantee identical behaviour."
        )
    return ModelBundle(pipeline=pipeline, metadata=metadata, evaluation=evaluation, warnings=warnings)
