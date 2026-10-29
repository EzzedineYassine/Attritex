from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from churnsense import __version__
from churnsense.api.errors import register_error_handlers
from churnsense.api.registry import ModelRegistry
from churnsense.api.routes import model, predictions
from churnsense.config import Settings, load_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.registry.state()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = FastAPI(
        title="ChurnSense API",
        version=__version__,
        description="Customer churn probability, risk tiers, and model evaluation.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = ModelRegistry(settings.artifact_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Content-Disposition"],
    )
    register_error_handlers(app)
    app.include_router(model.router)
    app.include_router(predictions.router)

    return app
