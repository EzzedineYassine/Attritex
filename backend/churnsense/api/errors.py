from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from churnsense.api.registry import ModelUnavailableError
from churnsense.data.validation import SchemaError
from churnsense.ml.inference import InputValidationError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str,
                 details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


def error_response(status_code: int, code: str, message: str,
                   details: list[dict[str, Any]] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or []}},
    )


def _field_from_location(location: tuple) -> str:
    parts = [str(part) for part in location if part not in ("body", "query", "path")]
    return ".".join(parts) or "request"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        return error_response(error.status_code, error.code, error.message, error.details)

    @app.exception_handler(ModelUnavailableError)
    async def handle_model_unavailable(_: Request, error: ModelUnavailableError) -> JSONResponse:
        return error_response(503, "model_unavailable", str(error), [{"reason": error.reason}])

    @app.exception_handler(InputValidationError)
    async def handle_input_validation(_: Request, error: InputValidationError) -> JSONResponse:
        details = [{"field": issue.column, "message": issue.message} for issue in error.issues]
        return error_response(422, "invalid_customer", "The customer record is inconsistent.", details)

    @app.exception_handler(SchemaError)
    async def handle_schema_error(_: Request, error: SchemaError) -> JSONResponse:
        details = [{"field": column, "message": "column is missing"} for column in error.missing_columns]
        return error_response(422, "invalid_schema", str(error), details)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(_: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {"field": _field_from_location(tuple(item.get("loc", ()))), "message": item.get("msg", "")}
            for item in error.errors()
        ]
        return error_response(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        return error_response(error.status_code, "http_error", str(error.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "An unexpected error occurred.")
