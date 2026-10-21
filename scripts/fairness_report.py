"""Produce approval-rate comparisons for responsible-AI evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_report(dataset: Path) -> dict:
    frame = pd.read_csv(dataset)
    frame["approved"] = frame["Loan_Status"].eq("Y").astype(int)
    groups = {}
    for feature in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        grouped = frame.groupby(feature, dropna=False)["approved"].agg(["mean", "count"]).reset_index()
        groups[feature] = [
            {"group": str(row[feature]), "approval_rate": round(float(row["mean"]), 4), "count": int(row["count"])}
            for _, row in grouped.iterrows()
        ]
    return {"dataset": str(dataset), "groups": groups, "note": "Descriptive audit only; not a legal fairness certification."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fairness_report.json"))
    args = parser.parse_args()
    report = build_report(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
