# Loan Prediction MLOps Project

This repository contains a reproducible machine-learning workflow for loan-status
prediction. The local workflow can be run without AWS, and the current cloud path
deploys the same tested model through GitHub Actions, ECR, SSM, and one free-tier-sized
EC2 instance in `ap-south-1`.

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

`/prediction_api` returns the decision, approval/rejection probabilities, confidence,
and local feature contributions. SHAP is used when available; a deterministic
feature-importance explanation is used as a runtime fallback. The root page renders
these same details in the browser form.

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
- `MLOPS_S3_BUCKET` and `MLOPS_DRIFT_PREFIX` configure optional S3-based monitoring.
- `MLOPS_ALLOWED_ORIGINS` controls permitted browser origins.
- `MLOPS_UPLOAD_PREDICTIONS` stays `false` locally and enables S3 uploads only in the cloud deployment.
- `MLOPS_AUTH_USERNAME`, `MLOPS_AUTH_PASSWORD`, and `MLOPS_AUTH_SECRET` enable bearer-token authentication when all three are set.

Do not put AWS access keys in the Dockerfile, workflow, or source code. The cloud
deployment uses an EC2 instance role and GitHub OIDC with short-lived credentials.

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
reports. The live API exposes Prometheus-compatible metrics, and container logs are
sent to CloudWatch at `/mlops/loan-approval` with seven-day retention. The local
Evidently report command remains available when a full HTML drift artifact is needed.

## Cloud Deployment

The deployment uses S3 for the DVC remote, ECR for the container image, and one
free-tier-sized EC2 instance for the running API. The manual release workflow in
`.github/workflows/deploy.yml` rebuilds the deterministic data and model, publishes an
immutable `release-<commit-sha>` image, deploys it through AWS Systems Manager, and
runs `/health` and `/ready` smoke tests inside the instance. It expects the
`AWS_ROLE_TO_ASSUME` GitHub secret for OIDC-based publishing. Deployment details and
the current resource identifiers are in `DEPLOYMENT_RUNBOOK.md`.

The release and rollback procedure is documented in `DEPLOYMENT_RUNBOOK.md`. The
Kubernetes manifest remains a portable alternative with an immutable image-tag
placeholder and readiness/liveness probes.

## Monitoring and Responsible AI

Start the local Prometheus/Grafana stack with:

```powershell
docker compose -f monitoring/docker-compose.yml up -d
```

The provisioned dashboard covers request rate, p95 latency, HTTP errors, total
requests, and process uptime. `scripts/check_drift.py` writes a PSI report and exits
with code 2 when the configured threshold is exceeded. `scripts/fairness_report.py`
creates approval-rate comparisons for the audit attributes in the synthetic data.

The scheduled GitHub workflow `.github/workflows/drift-monitor.yml` can trigger the
gated retraining workflow. Retraining must pass the configured F1 threshold before the
immutable deployment workflow is started; drift alone never deploys a model.
