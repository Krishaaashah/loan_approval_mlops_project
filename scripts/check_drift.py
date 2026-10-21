"""Calculate a lightweight PSI drift report and return 2 when retraining is needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.nanquantile(reference.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        reference_counts, _ = np.histogram(reference, bins=edges)
        current_counts, _ = np.histogram(current, bins=edges)
    else:
        categories = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
        reference_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0).values
        current_counts = current.astype(str).value_counts().reindex(categories, fill_value=0).values

    reference_share = np.clip(reference_counts / max(reference_counts.sum(), 1), 1e-6, None)
    current_share = np.clip(current_counts / max(current_counts.sum(), 1), 1e-6, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def build_report(reference_path: Path, current_path: Path, output_path: Path, threshold: float) -> dict:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    columns = [column for column in reference.columns if column in current.columns and column != "Loan_Status"]
    features = {column: psi(reference[column], current[column]) for column in columns}
    drifted = [column for column, value in features.items() if value >= threshold]
    report = {
        "reference": str(reference_path),
        "current": str(current_path),
        "threshold": threshold,
        "feature_psi": features,
        "drifted_features": drifted,
        "retraining_required": bool(drifted),
        "recommendation": "Run the gated retraining workflow." if drifted else "Continue monitoring.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift_report.json"))
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    report = build_report(args.reference, args.current, args.output, args.threshold)
    print(json.dumps(report, indent=2))
    raise SystemExit(2 if report["retraining_required"] else 0)


if __name__ == "__main__":
    main()
