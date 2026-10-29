from __future__ import annotations

import json

import pytest

from churnsense.ml.artifacts import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    load_bundle,
)


def test_load_bundle_success(trained_artifacts):
    bundle = load_bundle(trained_artifacts)
    assert bundle.version != ""
    assert hasattr(bundle.pipeline, "predict_proba")
    assert bundle.cutoffs.medium <= bundle.cutoffs.high


def test_load_bundle_missing_file_raises(tmp_path):
    with pytest.raises(ArtifactNotFoundError):
        load_bundle(tmp_path)


def test_load_bundle_incompatible_schema_raises(trained_artifacts, tmp_path):
    meta_path = trained_artifacts / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["artifact_schema_version"] = 999
    bad_dir = tmp_path / "bad_artifacts"
    bad_dir.mkdir()
    (bad_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    eval_text = (trained_artifacts / "evaluation.json").read_text(encoding="utf-8")
    (bad_dir / "evaluation.json").write_text(eval_text, encoding="utf-8")
    (bad_dir / "model.joblib").write_bytes((trained_artifacts / "model.joblib").read_bytes())

    with pytest.raises(ArtifactLoadError) as exc_info:
        load_bundle(bad_dir)
    assert "schema version" in str(exc_info.value).lower()
