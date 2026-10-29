from __future__ import annotations

from fastapi import Request

from churnsense.api.registry import ModelRegistry
from churnsense.config import Settings
from churnsense.ml.artifacts import ModelBundle


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_bundle(request: Request) -> ModelBundle:
    return get_registry(request).require()
