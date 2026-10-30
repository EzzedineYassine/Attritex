from __future__ import annotations

import io
import re
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from attritex.api.dependencies import get_bundle, get_settings
from attritex.api.errors import ApiError
from attritex.api.schemas import BatchPredictionResponse, CustomerRecord, PredictionResponse
from attritex.config import Settings
from attritex.data.fixture import generate_synthetic_customers
from attritex.data.schema import ID_COLUMN, TARGET_COLUMN, model_feature_names
from attritex.data.validation import require_columns
from attritex.ml.artifacts import ModelBundle
from attritex.ml.inference import score_batch, score_customer

router = APIRouter(prefix="/api/predict", tags=["predictions"])

READ_CHUNK_BYTES = 1 << 16
FORMULA_PREFIX = re.compile(r"^[=+@\t\r]|^-(?!\d)")
ALLOWED_CONTENT_TYPES = {
    "text/csv", "application/csv", "application/vnd.ms-excel", "text/plain",
    "application/octet-stream", "",
}


def _neutralise_formula(value: object) -> object:
    if isinstance(value, str) and FORMULA_PREFIX.match(value):
        return "'" + value
    return value


def _safe_csv(frame: pd.DataFrame) -> str:
    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    safe = frame.copy()
    for column in text_columns:
        safe[column] = safe[column].map(_neutralise_formula)
    return safe.to_csv(index=False)


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    buffer = bytearray()
    while chunk := await upload.read(READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise ApiError(413, "file_too_large",
                           f"File exceeds the {limit // (1024 * 1024)} MB upload limit.")
    return bytes(buffer)


def _parse_csv(payload: bytes, max_rows: int) -> pd.DataFrame:
    if not payload.strip():
        raise ApiError(422, "empty_file", "The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False,
                            skipinitialspace=True, encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ApiError(422, "invalid_encoding", "The file must be UTF-8 encoded CSV.") from error
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ApiError(422, "invalid_csv", f"The file could not be parsed as CSV: {error}") from error
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ApiError(422, "empty_file", "The CSV has a header but no customer rows.")
    if len(frame) > max_rows:
        raise ApiError(413, "too_many_rows",
                       f"The CSV has {len(frame):,} rows; the limit is {max_rows:,}.")
    require_columns(frame, require_target=False)
    return frame


@router.post("", response_model=PredictionResponse)
def predict_customer(
    customer: CustomerRecord,
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
) -> PredictionResponse:
    return PredictionResponse(**score_customer(bundle, customer.model_dump()))


@router.post("/batch", response_model=BatchPredictionResponse,
             responses={200: {"content": {"text/csv": {}}}})
async def predict_batch(
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="CSV with one customer per row")],
    output_format: Annotated[Literal["json", "csv"], Query(alias="format")] = "json",
):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise ApiError(415, "unsupported_file", "Upload a file with a .csv extension.")
    if (file.content_type or "") not in ALLOWED_CONTENT_TYPES:
        raise ApiError(415, "unsupported_file", f"Unsupported content type {file.content_type}.")

    payload = await _read_limited(file, settings.max_upload_bytes)
    frame = _parse_csv(payload, settings.max_batch_rows)
    result = await run_in_threadpool(score_batch, bundle, frame)

    if output_format == "csv":
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", filename.rsplit(".", 1)[0])[:60] or "customers"
        return Response(
            content=_safe_csv(result.output_frame),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}_scored.csv"'},
        )
    return BatchPredictionResponse(
        model_version=bundle.version,
        decision_threshold=bundle.cutoffs.high,
        tier_cutoffs=bundle.cutoffs.to_dict(),
        summary=result.summary,
        rows=result.rows,
    )


@router.get("/batch/template", response_class=Response,
            responses={200: {"content": {"text/csv": {}}}})
def batch_template() -> Response:
    sample = generate_synthetic_customers(rows=5, seed=2024)
    columns = [ID_COLUMN, *model_feature_names()]
    sample = sample.drop(columns=[TARGET_COLUMN])[columns]
    return Response(
        content=sample.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="attritex_batch_template.csv"'},
    )
