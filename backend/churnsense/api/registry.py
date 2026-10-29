from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from churnsense.ml.artifacts import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    ModelBundle,
    artifact_signature,
    load_bundle,
)

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RegistryState:
    bundle: ModelBundle | None
    error: str | None
    reason: str | None


class ModelRegistry:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self._lock = threading.Lock()
        self._signature: tuple[float, ...] | None = None
        self._state = RegistryState(bundle=None, error=None, reason=None)
        self._loaded_once = False

    def _load(self, signature: tuple[float, ...] | None) -> RegistryState:
        try:
            bundle = load_bundle(self.artifact_dir)
        except ArtifactNotFoundError as error:
            return RegistryState(bundle=None, error=str(error), reason="artifacts_missing")
        except ArtifactLoadError as error:
            logger.error("Model artifact rejected: %s", error)
            return RegistryState(bundle=None, error=str(error), reason="artifacts_invalid")
        logger.info("Loaded model %s from %s", bundle.version, self.artifact_dir)
        for warning in bundle.warnings:
            logger.warning(warning)
        return RegistryState(bundle=bundle, error=None, reason=None)

    def state(self) -> RegistryState:
        signature = artifact_signature(self.artifact_dir)
        if self._loaded_once and signature == self._signature:
            return self._state
        with self._lock:
            if not self._loaded_once or signature != self._signature:
                self._state = self._load(signature)
                self._signature = signature
                self._loaded_once = True
            return self._state

    def require(self) -> ModelBundle:
        state = self.state()
        if state.bundle is None:
            raise ModelUnavailableError(state.error or "Model is not loaded.",
                                        state.reason or "artifacts_missing")
        return state.bundle
