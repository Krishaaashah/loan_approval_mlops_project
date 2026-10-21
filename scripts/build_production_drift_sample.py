"""Build a feature CSV from downloaded production prediction event JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prediction_model.config import config


def build_sample(source: Path, output: Path, limit: int) -> int:
    events = []
    for path in source.rglob("*.json"):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
            features = event.get("features", {})
            if all(feature in features for feature in config.FEATURES):
                events.append((event.get("timestamp", ""), features))
        except (OSError, json.JSONDecodeError, TypeError):
            continue

    events.sort(key=lambda item: item[0])
    rows = [features for _, features in events[-limit:]]
    if not rows:
        raise SystemExit("No valid production prediction events were found.")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=config.FEATURES).to_csv(output, index=False)
    print(f"Wrote {len(rows)} production records to {output}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/production_sample.csv"))
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    build_sample(args.source, args.output, args.limit)


if __name__ == "__main__":
    main()
