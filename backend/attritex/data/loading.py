from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TELCO_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
TELCO_SOURCE_REPOSITORY = "https://github.com/IBM/telco-customer-churn-on-icp4d"
TELCO_LICENSE = "Apache-2.0"
TELCO_DATASET_NAME = "IBM Telco Customer Churn"
TELCO_EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


class DataLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetProvenance:
    name: str
    source_url: str | None
    repository: str | None
    license: str | None
    sha256: str
    rows: int
    retrieved_at: str
    synthetic: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def provenance_path_for(data_path: Path) -> Path:
    return data_path.with_suffix(data_path.suffix + ".provenance.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_customer_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"Dataset not found at {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise DataLoadError(f"Could not parse CSV at {path}: {error}") from error
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def write_provenance(data_path: Path, provenance: DatasetProvenance) -> Path:
    target = provenance_path_for(data_path)
    target.write_text(json.dumps(provenance.to_dict(), indent=2), encoding="utf-8")
    return target


def read_provenance(data_path: Path) -> DatasetProvenance:
    source = provenance_path_for(data_path)
    sha256 = file_sha256(data_path)
    if source.exists():
        stored = json.loads(source.read_text(encoding="utf-8"))
        if stored.get("sha256") == sha256:
            return DatasetProvenance(**stored)
        logger.warning("Provenance checksum mismatch for %s; treating file as user-provided", data_path)
    rows = len(read_customer_csv(data_path))
    return DatasetProvenance(
        name=data_path.name,
        source_url=None,
        repository=None,
        license=None,
        sha256=sha256,
        rows=rows,
        retrieved_at=datetime.fromtimestamp(data_path.stat().st_mtime, UTC).isoformat(),
        synthetic=False,
        notes="User-provided file without a matching provenance record.",
    )


def download_telco_dataset(destination: Path, timeout_seconds: int = 60) -> DatasetProvenance:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(TELCO_SOURCE_URL, timeout=timeout_seconds) as response:
            temporary.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise DataLoadError(
            f"Download failed from {TELCO_SOURCE_URL}: {error}. "
            "Place the CSV manually at the destination or use the synthetic fixture."
        ) from error

    sha256 = file_sha256(temporary)
    notes = ""
    if sha256 != TELCO_EXPECTED_SHA256:
        notes = f"Checksum differs from the pinned value {TELCO_EXPECTED_SHA256}."
        logger.warning(notes)
    temporary.replace(destination)

    provenance = DatasetProvenance(
        name=TELCO_DATASET_NAME,
        source_url=TELCO_SOURCE_URL,
        repository=TELCO_SOURCE_REPOSITORY,
        license=TELCO_LICENSE,
        sha256=sha256,
        rows=len(read_customer_csv(destination)),
        retrieved_at=datetime.now(UTC).isoformat(),
        synthetic=False,
        notes=notes,
    )
    write_provenance(destination, provenance)
    return provenance
