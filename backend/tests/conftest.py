from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from attritex.api.app import create_app
from attritex.config import Settings
from attritex.data.fixture import generate_synthetic_customers
from attritex.ml.artifacts import load_bundle
from attritex.ml.training import TrainingConfig, train_from_frame


@pytest.fixture(scope="session")
def synthetic_frame():
    return generate_synthetic_customers(rows=600, seed=123)


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, synthetic_frame):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    config = TrainingConfig(
        seed=10,
        cv_folds=3,
        bootstrap_samples=50,
        permutation_repeats=2,
    )
    provenance = {
        "name": "Synthetic test fixture",
        "source_url": None,
        "repository": None,
        "license": "Test",
        "sha256": "test-sha",
        "rows": len(synthetic_frame),
        "retrieved_at": "2026-01-01T00:00:00Z",
        "synthetic": True,
        "notes": "Testing",
    }
    train_from_frame(synthetic_frame, provenance, artifact_dir, config)
    return artifact_dir


@pytest.fixture
def bundle(trained_artifacts):
    return load_bundle(trained_artifacts)


@pytest.fixture
def test_settings(trained_artifacts, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return Settings(
        data_dir=data_dir,
        artifact_dir=trained_artifacts,
        cors_origins=("http://localhost:5173",),
        max_upload_bytes=1024 * 1024,
        max_batch_rows=1000,
    )


@pytest.fixture
def client(test_settings):
    app = create_app(test_settings)
    return TestClient(app)
