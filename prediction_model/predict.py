import pandas as pd
import numpy as np
import joblib
from functools import lru_cache
from pathlib import Path
from prediction_model.config import config


def _mlflow():
    import mlflow

    return mlflow


def _load_best_mlflow_model():
    mlflow = _mlflow()
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{config.EXPERIMENT_NAME}' does not exist."
        )

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )
    if runs_df.empty:
        raise RuntimeError("No trained model runs are available in MLflow.")

    best_run_id = runs_df.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/{config.MODEL_NAME.lstrip('/')}"
    return mlflow.sklearn.load_model(model_uri)


@lru_cache(maxsize=1)
def get_model():
    """Load one configured model and reuse it for subsequent requests."""
    if config.MODEL_URI:
        return _mlflow().sklearn.load_model(config.MODEL_URI)

    local_model = Path(config.MODEL_PATH)
    if local_model.exists():
        return joblib.load(local_model)

    # Compatibility fallback for an existing MLflow deployment.
    return _load_best_mlflow_model()


def clear_model_cache():
    get_model.cache_clear()


def _predict(model, data_input):
    data = pd.DataFrame(data_input)
    prediction = model.predict(data)
    output = np.where(prediction == 1, "Y", "N")
    return {"prediction": output.tolist()}



def generate_predictions(data_input):
    return _predict(get_model(), data_input)


def generate_predictions_batch(data_input):
    return _predict(get_model(), data_input)





if __name__=='__main__':
    raise SystemExit("Use the FastAPI service or import generate_predictions().")
