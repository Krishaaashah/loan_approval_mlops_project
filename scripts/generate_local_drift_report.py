"""Generate an Evidently data-drift report without requiring AWS."""

import argparse
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "artifacts" / "train.csv"
DEFAULT_CURRENT = ROOT / "artifacts" / "test.csv"
DEFAULT_OUTPUT = ROOT / "poster_screenshots" / "evidently_drift_report.html"


def generate_report(reference_path: Path, current_path: Path, output_path: Path) -> Path:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Loan_Status is the training label, not an input feature for drift checks.
    reference = reference.drop(columns=["Loan_Status"], errors="ignore")
    current = current.drop(columns=["Loan_Status"], errors="ignore")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate_report(args.reference, args.current, args.output))


if __name__ == "__main__":
    main()
