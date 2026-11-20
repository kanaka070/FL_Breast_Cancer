"""Create a curated, copy-only project package under submission/project."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submission"
PACKAGE = SUBMISSION / "project"

SOURCE_FILES = [
    "academic_experiments.py",
    "centralized.py",
    "client.py",
    "dashboard.py",
    "dashboard_charts.py",
    "data.py",
    "dp_accountant.py",
    "dp_federated.py",
    "dp_training.py",
    "evaluation.py",
    "experiments.py",
    "feature_schema.py",
    "inference.py",
    "model.py",
    "retraining.py",
    "server.py",
    "workflow_utils.py",
    "utils.py",
]
METRIC_FILES = [
    "final_three_model_comparison.csv",
    "four_model_comparison.csv",
    "fedavg_training_history.csv",
    "dp_federated_history.csv",
    "privacy_repeated_summary.csv",
    "hospital_reliability.csv",
    "hospital_distribution.csv",
    "iid_non_iid_comparison.csv",
]
MODEL_FILES = [
    "centralized_logistic_regression.pkl",
    "fedavg_model.npz",
    "fedavg_scaler.pkl",
    "dp_federated_model.npz",
    "dp_federated_scaler.pkl",
]
DATA_FILES = ["wdbc.data", "wdbc.names"]
RESEARCH_FILES = [
    "cross_validation_folds.csv",
    "cross_validation_summary.csv",
    "federated_cross_validation_folds.csv",
    "federated_cross_validation_summary.csv",
    "trust_weights.csv",
    "trust_aggregation_test_metrics.csv",
    "trust_aggregation_method.json",
    "iid_noniid_results.csv",
    "iid_noniid_training_history.csv",
    "execution_metadata.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_files(source_dir: Path, names: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Required submission input is missing: {source}")
        shutil.copy2(source, destination / name)


def main() -> None:
    if not SUBMISSION.is_dir():
        raise FileNotFoundError(f"Create the report/checklist first: {SUBMISSION}")
    if PACKAGE.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing submission package: {PACKAGE}"
        )

    PACKAGE.mkdir()
    try:
        copy_files(ROOT, SOURCE_FILES, PACKAGE)
        copy_files(ROOT, ["README.md", "requirements.txt"], PACKAGE)
        copy_files(ROOT / "tests", ["test_workflow.py"], PACKAGE / "tests")
        copy_files(ROOT / "data", DATA_FILES, PACKAGE / "data")
        copy_files(ROOT / "saved_models", MODEL_FILES, PACKAGE / "saved_models")
        copy_files(ROOT / "metrics", METRIC_FILES, PACKAGE / "metrics")
        copy_files(
            ROOT / "research_outputs" / "verified",
            RESEARCH_FILES,
            PACKAGE / "research_outputs" / "verified",
        )

        protected_paths = [
            *(ROOT / "data" / name for name in DATA_FILES),
            *(ROOT / "saved_models" / name for name in MODEL_FILES),
            *(ROOT / "metrics" / name for name in METRIC_FILES),
        ]
        manifest = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in protected_paths
        }
        manifest_path = PACKAGE / "PROTECTED_ARTIFACT_SHA256.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        readme_path = PACKAGE / "README.md"
        packaged_readme = readme_path.read_text(encoding="utf-8")
        if "## Final academic materials" in packaged_readme:
            before_materials, materials_and_rest = packaged_readme.split(
                "## Final academic materials", 1
            )
            _, after_materials = materials_and_rest.split(
                "## Research limitations and interpretation", 1
            )
            packaged_readme = (
                before_materials
                + "## Final academic materials\n\n"
                + "- Research report: `../RESEARCH_REPORT.md`\n"
                + "- Presentation: `../PRESENTATION.html`\n"
                + "- Requirement/evidence checklist: `../FINAL_REQUIREMENTS_CHECKLIST.md`\n"
                + "- Verification record: `../VERIFICATION_RECORD.md`\n\n"
                + "This is the curated project copy. The package-preparation script "
                + "is kept in the original project root.\n\n"
                + "## Research limitations and interpretation"
                + after_materials
            )
        readme_path.write_text(packaged_readme, encoding="utf-8")
        (PACKAGE / "PACKAGE_CONTENTS.md").write_text(
            "# Curated project package\n\n"
            "Contains the application/research Python source, focused tests, "
            "pinned dependencies, public WDBC source data, required baseline "
            "models and dashboard metrics, and newly verified experiment outputs.\n\n"
            "The package intentionally excludes virtual environments, caches, "
            "mutable `active_model.json` approval state, confirmed-label ledgers, "
            "candidate models, unselected historical outputs, and notebook "
            "execution state. `PROTECTED_ARTIFACT_SHA256.json` records the "
            "source hashes of the copied data, baseline model, and metric files. "
            "The report, presentation, and checklist are in the parent "
            "`submission/` folder.\n\n"
            "From this directory, create a Python 3.13 environment, install "
            "`requirements.txt`, run `python -m unittest discover -s tests -v`, "
            "then launch with `streamlit run dashboard.py`. Additional isolated "
            "experiments are run with `python academic_experiments.py "
            "--output-dir research_outputs\\\\verified`.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(PACKAGE)
        raise

    print(f"Created curated package: {PACKAGE}")


if __name__ == "__main__":
    main()
