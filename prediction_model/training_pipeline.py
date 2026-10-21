"""Train, tune, track, and persist the loan prediction model."""

import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from prediction_model.config import config
from prediction_model.processing import preprocessing as pp


SEARCH_SPACE = {
    "max_depth": hp.choice("max_depth", np.arange(3, 10, dtype=int)),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "n_estimators": hp.choice("n_estimators", np.arange(50, 300, 50, dtype=int)),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "gamma": hp.uniform("gamma", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
}


def load_training_data():
    train_data = pd.read_csv(config.PREPARED_TRAIN_PATH)
    test_data = pd.read_csv(config.PREPARED_TEST_PATH)
    return (
        train_data[config.FEATURES],
        train_data[config.TARGET].map({"N": 0, "Y": 1}),
        test_data[config.FEATURES],
        test_data[config.TARGET].map({"N": 0, "Y": 1}),
    )


def build_pipeline(params):
    classifier = xgb.XGBClassifier(
        **params,
        random_state=42,
        eval_metric="logloss",
    )
    return Pipeline(
        [
            (
                "DomainProcessing",
                pp.DomainProcessing(
                    variable_to_modify=config.FEATURE_TO_MODIFY,
                    variable_to_add=config.FEATURE_TO_ADD,
                ),
            ),
            ("MeanImputation", pp.MeanImputer(variables=config.NUM_FEATURES)),
            ("ModeImputation", pp.ModeImputer(variables=config.CAT_FEATURES)),
            ("DropFeatures", pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
            ("LabelEncoder", pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
            ("LogTransform", pp.LogTransforms(variables=config.LOG_FEATURES)),
            ("MinMaxScale", MinMaxScaler()),
            ("XGBoostClassifier", classifier),
        ]
    )


def calculate_metrics(y_true, y_pred, y_probability=None):
    metrics = {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }
    if y_probability is not None:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_probability))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
    return metrics


def train_model(max_evals=5):
    X_train, y_train, X_test, y_test = load_training_data()
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    mlflow.xgboost.autolog(log_models=False)

    with mlflow.start_run(run_name="hyperparameter-search"):

        def objective(params):
            model = build_pipeline(params)
            with mlflow.start_run(nested=True, run_name="candidate"):
                mlflow.log_params(params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]
                metrics = calculate_metrics(y_test, predictions, probabilities)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, config.MODEL_NAME)
            return {"loss": 1 - metrics["f1_score"], "status": STATUS_OK}

        trials = Trials()
        fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42),
        )
        best_assignment = {
            key: value[0] if isinstance(value, list) else value
            for key, value in trials.best_trial["misc"]["vals"].items()
        }
        best_params = space_eval(SEARCH_SPACE, best_assignment)
        best_params = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in best_params.items()
        }

        best_model = build_pipeline(best_params)
        best_model.fit(X_train, y_train)
        best_metrics = calculate_metrics(
            y_test,
            best_model.predict(X_test),
            best_model.predict_proba(X_test)[:, 1],
        )
        mlflow.log_params(best_params)
        mlflow.log_metrics({f"best_{key}": value for key, value in best_metrics.items()})
        model_info = mlflow.sklearn.log_model(best_model, config.MODEL_NAME)
        if config.REGISTERED_MODEL_NAME:
            mlflow.register_model(
                model_info.model_uri,
                config.REGISTERED_MODEL_NAME,
            )

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH)
    with open(config.BEST_PARAMS_PATH, "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)

    return best_metrics


if __name__ == "__main__":
    print(train_model())
