from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from churnsense.config import load_settings
from churnsense.data.fixture import write_synthetic_fixture
from churnsense.data.loading import DataLoadError, download_telco_dataset
from churnsense.ml.artifacts import ArtifactLoadError, ArtifactNotFoundError, load_bundle
from churnsense.ml.training import TrainingConfig, TrainingDataError, train_from_file

logger = logging.getLogger("churnsense")


def _download(args: argparse.Namespace) -> int:
    destination = Path(args.dest) if args.dest else load_settings().raw_data_path
    provenance = download_telco_dataset(destination)
    print(f"Downloaded {provenance.rows:,} rows to {destination}")
    print(f"sha256 {provenance.sha256}")
    if provenance.notes:
        print(f"Note: {provenance.notes}")
    return 0


def _fixture(args: argparse.Namespace) -> int:
    destination = Path(args.dest) if args.dest else load_settings().fixture_path
    provenance = write_synthetic_fixture(destination, rows=args.rows, seed=args.seed)
    print(f"Wrote {provenance.rows:,} SYNTHETIC rows to {destination}")
    return 0


def _resolve_training_data(args: argparse.Namespace) -> Path:
    settings = load_settings()
    if args.data:
        return Path(args.data)
    if args.fixture:
        if not settings.fixture_path.exists():
            write_synthetic_fixture(settings.fixture_path)
        return settings.fixture_path
    return settings.raw_data_path


def _train(args: argparse.Namespace) -> int:
    settings = load_settings()
    data_path = _resolve_training_data(args)
    artifact_dir = Path(args.artifacts) if args.artifacts else settings.artifact_dir
    if args.if_missing and (artifact_dir / "metadata.json").exists():
        print(f"Artifacts already present in {artifact_dir}; skipping training.")
        return 0
    config = TrainingConfig(
        seed=args.seed,
        cv_folds=args.folds,
        bootstrap_samples=200 if args.quick else 1000,
        permutation_repeats=3 if args.quick else 10,
    )
    result = train_from_file(data_path, artifact_dir, config)
    metrics = result.metadata["headline_metrics"]
    print(f"Model version   {result.metadata['model_version']}")
    print(f"Selected        {result.metadata['model']['label']}")
    print(f"Holdout ROC-AUC {metrics['roc_auc']:.4f}")
    print(f"Holdout PR-AUC  {metrics['pr_auc']:.4f}")
    print(f"Threshold       {metrics['decision_threshold']:.4f}  precision {metrics['precision']:.4f}"
          f"  recall {metrics['recall']:.4f}  F1 {metrics['f1']:.4f}")
    print(f"Artifacts       {artifact_dir}")
    if result.metadata["dataset"]["provenance"]["synthetic"]:
        print("WARNING: trained on SYNTHETIC fixture data. Metrics are not meaningful.")
    return 0


def _info(args: argparse.Namespace) -> int:
    settings = load_settings()
    artifact_dir = Path(args.artifacts) if args.artifacts else settings.artifact_dir
    bundle = load_bundle(artifact_dir)
    summary = {
        "model_version": bundle.version,
        "model": bundle.metadata["model"]["label"],
        "dataset": bundle.metadata["dataset"]["provenance"]["name"],
        "headline_metrics": bundle.metadata["headline_metrics"],
        "warnings": bundle.warnings,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("churnsense.api.app:create_app", factory=True, host=args.host, port=args.port,
                reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churnsense", description="ChurnSense command line")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Obtain training data").add_subparsers(
        dest="data_command", required=True)
    download = data.add_parser("download", help="Download the IBM Telco Customer Churn dataset")
    download.add_argument("--dest", help="Destination CSV path")
    download.set_defaults(handler=_download)
    fixture = data.add_parser("fixture", help="Write the synthetic development fixture")
    fixture.add_argument("--dest", help="Destination CSV path")
    fixture.add_argument("--rows", type=int, default=1500)
    fixture.add_argument("--seed", type=int, default=7)
    fixture.set_defaults(handler=_fixture)

    train = commands.add_parser("train", help="Train, evaluate, and save the model")
    source = train.add_mutually_exclusive_group()
    source.add_argument("--data", help="CSV with the Telco schema and a Churn column")
    source.add_argument("--fixture", action="store_true", help="Train on the synthetic fixture")
    train.add_argument("--artifacts", help="Output directory for model artifacts")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--quick", action="store_true", help="Fewer bootstrap and permutation repeats")
    train.add_argument("--if-missing", action="store_true", help="Skip when artifacts exist")
    train.set_defaults(handler=_train)

    info = commands.add_parser("info", help="Show the current model summary")
    info.add_argument("--artifacts", help="Artifact directory")
    info.set_defaults(handler=_info)

    serve = commands.add_parser("serve", help="Run the prediction API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (DataLoadError, TrainingDataError, ArtifactNotFoundError, ArtifactLoadError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
