"""Compare an auxiliary XGBoost regressor score with the deployed classifier.

The regressor is an academic comparison only. The binary classifier remains the
production decision model because it is trained for the actual approval target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prediction_model.config import config
from prediction_model.pipeline import preprocessing_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/regressor_metrics.json"))
    args = parser.parse_args()

    train = pd.read_csv(config.PREPARED_TRAIN_PATH)
    test = pd.read_csv(config.PREPARED_TEST_PATH)
    X_train = preprocessing_pipeline.fit_transform(train[config.FEATURES], train[config.TARGET])
    X_test = preprocessing_pipeline.transform(test[config.FEATURES])
    y_train = train[config.TARGET].map({"N": 0, "Y": 1}).astype(float)
    y_test = test[config.TARGET].map({"N": 0, "Y": 1}).astype(float)

    regressor = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:logistic",
        eval_metric="rmse",
        random_state=42,
    )
    regressor.fit(X_train, y_train)
    score = np.clip(regressor.predict(X_test), 0, 1)
    metrics = {
        "model": "XGBRegressor auxiliary score",
        "mae": float(mean_absolute_error(y_test, score)),
        "rmse": float(mean_squared_error(y_test, score) ** 0.5),
        "r2": float(r2_score(y_test, score)),
        "note": "Not deployed as the binary decision model; compare with classifier probability metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
