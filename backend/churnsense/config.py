from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _path_from_env(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name, "").strip()
    return Path(raw_value).expanduser().resolve() if raw_value else default


def _origins_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    return tuple(origin.strip() for origin in raw_value.split(",") if origin.strip())


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifact_dir: Path
    cors_origins: tuple[str, ...]
    max_upload_bytes: int
    max_batch_rows: int

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "raw" / "telco_customer_churn.csv"

    @property
    def fixture_path(self) -> Path:
        return self.data_dir / "fixtures" / "synthetic_customers.csv"


def load_settings() -> Settings:
    return Settings(
        data_dir=_path_from_env("CHURNSENSE_DATA_DIR", PROJECT_ROOT / "data"),
        artifact_dir=_path_from_env("CHURNSENSE_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"),
        cors_origins=_origins_from_env("CHURNSENSE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS),
        max_upload_bytes=_positive_int_from_env("CHURNSENSE_MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
        max_batch_rows=_positive_int_from_env("CHURNSENSE_MAX_BATCH_ROWS", 50_000),
    )
