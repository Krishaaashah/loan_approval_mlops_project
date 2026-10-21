# Project Context

## Problem

This project demonstrates an MLOps workflow for a loan-status prediction service.
Given an applicant's demographic, employment, income, credit-history, and property
information, the model predicts whether a loan should be approved (`Y`) or rejected
(`N`). The prediction is a technical demonstration and must not be used as a real
credit-decision system without appropriate legal, fairness, privacy, and validation
controls.

## Why MLOps Is Needed

A model is not enough by itself. The project must also make data changes traceable,
training repeatable, experiments comparable, deployment reproducible, and production
behavior observable. This repository demonstrates those concerns locally and on a
small AWS deployment.

## Current Local Flow

1. The dataset is versioned with DVC metadata.
2. `prepare_data.py` validates the schema and creates a deterministic train/test split.
3. `training_pipeline.py` applies preprocessing, searches XGBoost hyperparameters with Hyperopt, and logs candidate runs to MLflow.
4. The best candidate is selected by F1 score and saved as `artifacts/model.joblib`.
5. `evaluate.py` writes accuracy, precision, recall, and F1 to `artifacts/metrics.json`.
6. FastAPI loads one configured model and serves single and batch predictions.
7. Docker packages the API with a health check; it does not train during image creation.
8. Prometheus-compatible API metrics, Grafana dashboards, drift reports, and fairness
   reports provide monitoring and governance evidence.

## Synthetic Dataset

`scripts/generate_synthetic_data.py` creates a deterministic local dataset with the
same schema expected by the model. It includes controlled missing values so the
imputation steps are exercised and a small amount of label noise so the task is not
perfectly separable.

Generate it with:

```powershell
python scripts/generate_synthetic_data.py
```

This data is suitable for testing the pipeline, API, Docker image, and documentation.
It is not a substitute for a real, representative, consented, and legally approved
credit dataset.

## Target Cloud Flow

When cloud work begins, the local artifacts map to the target architecture as follows:

```text
DVC/S3 dataset -> DVC pipeline -> MLflow experiment and selected model
                                      |
                                      v
                         Docker image -> ECR -> EC2 API service
                                      |
                                      v
                             client prediction request
```

The implemented cloud flow adds IAM roles, ECR publishing, SSM-based EC2 deployment,
CloudWatch logging, secrets isolation through GitHub OIDC, immutable releases,
rollback guidance, and submission evidence. Prometheus-compatible metrics are scraped
by a resource-limited cloud Grafana/Prometheus stack, while each production prediction
is stored as a feature event in the existing private S3 bucket for scheduled drift
checks. Authentication remains opt-in, while confidence scores, SHAP explanations,
PSI drift gates, and fairness reports are implemented in the application code.
