"""Create deterministic train and test artifacts for the DVC pipeline."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from prediction_model.config import config


def prepare_data():
    source_path = os.path.join(config.DATAPATH, config.TRAIN_FILE)
    data = pd.read_csv(source_path)

    required_columns = set(config.FEATURES + [config.TARGET])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")

    data = data[config.FEATURES + [config.TARGET]].dropna(
        subset=[config.TARGET]
    )
    stratify = data[config.TARGET] if data[config.TARGET].nunique() > 1 else None
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    train_data.to_csv(config.PREPARED_TRAIN_PATH, index=False)
    test_data.to_csv(config.PREPARED_TEST_PATH, index=False)


if __name__ == "__main__":
    prepare_data()
