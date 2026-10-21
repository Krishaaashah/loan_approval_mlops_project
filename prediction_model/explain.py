"""Confidence and local feature explanations for the deployed model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prediction_model.config import config


MODEL_FEATURES = [feature for feature in config.FEATURES if feature not in config.DROP_FEATURES]


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _confidence(probability: float) -> str:
    distance = abs(probability - 0.5) * 2
    if distance >= 0.6:
        return "High"
    if distance >= 0.25:
        return "Medium"
    return "Low"


def _transformed_values(model, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())[:-1]
        transformed = frame
        for step in steps:
            transformed = step.transform(transformed)
        return np.asarray(transformed, dtype=float)
    numeric = frame.select_dtypes(include=["number"])
    return np.asarray(numeric, dtype=float)


def _contributions(model, frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    transformed = _transformed_values(model, frame)
    estimator = getattr(model, "named_steps", {}).get("XGBoostClassifier", model)

    try:
        import shap

        values = shap.TreeExplainer(estimator)(transformed).values
        if values.ndim == 3:
            values = values[:, :, 1]
        return np.asarray(values[0], dtype=float), "SHAP TreeExplainer"
    except Exception:
        # Keep the API explainable even when the optional SHAP wheel is absent.
        importance = np.asarray(getattr(estimator, "feature_importances_", []), dtype=float)
        if importance.size != transformed.shape[1]:
            importance = np.ones(transformed.shape[1], dtype=float)
        baseline = transformed
        try:
            training = pd.read_csv(config.PREPARED_TRAIN_PATH)
            baseline = _transformed_values(model, training[config.FEATURES])
        except (FileNotFoundError, KeyError, ValueError):
            pass
        centered = transformed[0] - np.nanmedian(baseline, axis=0)
        return importance * centered, "feature-importance fallback"


def explain_prediction(model, data_input: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(data_input)
    prediction = int(np.asarray(model.predict(frame))[0])
    if hasattr(model, "predict_proba"):
        probability = float(np.asarray(model.predict_proba(frame))[0, 1])
    else:
        probability = float(prediction)

    contributions, method = _contributions(model, frame)
    values = frame.iloc[0]
    ranked = sorted(
        zip(MODEL_FEATURES, contributions), key=lambda item: abs(float(item[1])), reverse=True
    )

    positive = []
    negative = []
    neutral = []
    for feature, contribution in ranked:
        item = {
            "feature": feature,
            "value": _json_value(values.get(feature)),
            "contribution": round(float(contribution), 6),
        }
        if contribution > 0.01:
            positive.append(item)
        elif contribution < -0.01:
            negative.append(item)
        else:
            neutral.append(item)

    return {
        "decision": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": round(probability, 6),
        "rejection_probability": round(1 - probability, 6),
        "confidence": _confidence(probability),
        "explanation_method": method,
        "positive_factors": positive[:3],
        "negative_factors": negative[:3],
        "neutral_factors": neutral[:3],
    }
