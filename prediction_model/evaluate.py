"""Evaluate the DVC-produced model and write tracked metrics."""

import json
import os

import joblib
import pandas as pd

from prediction_model.config import config
from prediction_model.training_pipeline import calculate_metrics


def evaluate_model():
    test_data = pd.read_csv(config.PREPARED_TEST_PATH)
    model = joblib.load(config.MODEL_PATH)
    y_true = test_data[config.TARGET].map({"N": 0, "Y": 1})
    y_pred = model.predict(test_data[config.FEATURES])
    y_probability = model.predict_proba(test_data[config.FEATURES])[:, 1]
    metrics = calculate_metrics(y_true, y_pred, y_probability)

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    if metrics["f1_score"] < config.MIN_F1_SCORE:
        raise ValueError(
            f"F1 score {metrics['f1_score']:.4f} is below the required "
            f"threshold {config.MIN_F1_SCORE:.4f}."
        )
    return metrics


if __name__ == "__main__":
    evaluate_model()
