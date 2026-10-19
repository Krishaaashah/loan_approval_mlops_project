# Loan Prediction MLOps Project

This repository contains a reproducible machine-learning workflow for loan-status
prediction. The local workflow can be run without AWS. Cloud deployment is kept
configuration-driven so it can be enabled later when the required AWS resources are
available.

## Workflow

```text
DVC-managed dataset
        |
        v
prepare -> train/tune with Hyperopt + MLflow -> evaluate metrics
                                              |
                                              v
                                  versioned model artifact
                                              |
                                              v
                                      FastAPI + Docker
```

The DVC stages are defined in `dvc.yaml`:

- `prepare` validates the source data and creates deterministic train/test artifacts.
- `train` runs Hyperopt, logs candidate runs to MLflow, and saves `artifacts/model.joblib`.
- `evaluate` writes `artifacts/metrics.json` for reproducible comparison.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The dataset must be available at `prediction_model/datasets/train.csv`. If it is
managed by DVC, run `dvc pull` after configuring credentials for the DVC remote.

Run the pipeline after the data is available:

```powershell
dvc repro
python -m pytest -q
```

Start the API locally:

```powershell
uvicorn main:app --reload --port 8005
```

Useful endpoints are `/health`, `/ready`, `/docs`, `/metrics`, and
`/prediction_api`.

Install `requirements-monitoring.txt` separately when running the Streamlit drift
dashboard.

Generate a local Evidently report without AWS:

```powershell
$env:DO_NOT_TRACK = "1"
python scripts/generate_local_drift_report.py
```

The report compares `artifacts/train.csv` with `artifacts/test.csv` and writes
`poster_screenshots/evidently_drift_report.html`.

## Configuration

Copy `.env.example` to `.env` for local defaults. Important values are read from
environment variables:

- `MLFLOW_TRACKING_URI` selects the local file store or a later remote MLflow server.
- `MODEL_PATH` selects the persisted local model artifact.
- `MODEL_URI` can point directly to a pinned MLflow model version or alias.
- `MLFLOW_REGISTERED_MODEL_NAME` optionally registers the selected model in MLflow.
- `MLOPS_S3_BUCKET` and `MLOPS_DRIFT_PREFIX` configure later S3-based monitoring.
- `MLOPS_ALLOWED_ORIGINS` controls permitted browser origins.
- `MLOPS_UPLOAD_PREDICTIONS` stays `false` locally and enables S3 uploads only in the cloud deployment.

Do not put AWS access keys in the Dockerfile, workflow, or source code. The later
cloud deployment should use an EC2 instance role and GitHub OIDC or another short-
lived credential mechanism.

## Docker

The image only packages and serves the API. Training and tests run before image
creation, so image builds do not require cloud credentials or a live MLflow server.

```powershell
docker build -t loan-prediction-api .
docker run --rm -p 8005:8005 --env-file .env loan-prediction-api
```

## Monitoring

FastAPI metrics are exposed through `/metrics`. The drift-monitoring application is
in `drift_monitoring/app_v1.py` and uses Evidently for data-drift and data-quality
reports. Prometheus/Grafana deployment and AWS CloudWatch integration are intentionally
left for the cloud phase.

## Planned Cloud Phase

The intended deployment uses S3 for the DVC remote, ECR for the container image, and
one free-tier-sized EC2 instance for the running API. The manual release workflow is in
`.github/workflows/deploy.yml` and expects the `AWS_ROLE_TO_ASSUME` GitHub secret for
OIDC-based publishing. Deployment details are in `DEPLOYMENT_RUNBOOK.md`.

The release and rollback procedure is documented in
`DEPLOYMENT_RUNBOOK.md`. The Kubernetes manifest uses a placeholder immutable image
tag and readiness/liveness probes; replace the placeholder only during a real cloud
deployment.
