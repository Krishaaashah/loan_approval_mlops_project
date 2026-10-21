# Rubric Completion Checklist

## CO1: Cloud Services and MLOps Fundamentals

Completed:

- DVC remote is configured in `.dvc/config`, and the S3 remote is versioned and private.
- EC2 instance `i-02fd037d73f8a7b68` uses an instance role and restricted security group.
- ECR contains immutable release tag `release-3f320f4974cb91422e30447b83beb20e90e23522`.
- CloudWatch log group `/mlops/loan-approval` receives container logs with seven-day retention.
- Live health, readiness, prediction, `/docs`, and `/metrics` requests were verified.

## CO2: ML Pipeline and CI/CD

Completed:

- `dvc.yaml` defines prepare, train, and evaluate stages.
- Hyperopt searches multiple XGBoost candidates and MLflow records the runs.
- GitHub Actions tests the repository, builds the image, pushes to ECR, and deploys through SSM.
- Releases use immutable commit-based tags, never `latest`.
- The SSM deployment runs `/health` and `/ready` smoke tests inside EC2.
- Rollback guidance and one successful reference deployment are recorded in `DEPLOYMENT_RUNBOOK.md`.

## CO3: Tracking, Monitoring, and Optimization

Completed:

- MLflow tracks candidate runs and F1-based model selection.
- The selected model is persisted as `artifacts/model.joblib`.
- `artifacts/metrics.json` records accuracy, precision, recall, and F1.
- Probability credibility is measured with Brier score and ROC-AUC; an auxiliary
  XGBoost regressor comparison is available at `scripts/compare_xgboost_regressor.py`.
- The API returns confidence and local feature explanations, using SHAP when installed.
- The API exposes Prometheus-compatible `/metrics` output.
- CloudWatch captures the running container logs.
- `scripts/generate_local_drift_report.py` provides the Evidently drift-report path.
- `scripts/check_drift.py` calculates PSI drift and the scheduled workflow gates retraining.
- `scripts/fairness_report.py` produces group approval-rate audits.
- Optional bearer authentication is implemented through environment variables.

The repository includes a complete Prometheus/Grafana dashboard configuration under
`monitoring/`. The live EC2 deployment still uses the lightweight API/CloudWatch path
unless the monitoring compose stack is deliberately started on a suitably sized host.

## CO4: Documentation, Deployment Strategy, and Presentation

Completed:

- Local setup, configuration, architecture, and cloud deployment are documented.
- Tests cover preprocessing and API behavior; the Docker image has a health check and
  runs as a non-root user.
- Evidence is mapped in `SUBMISSION_EVIDENCE.md` and `poster_screenshots/`.
- AWS services, IAM boundaries, region, instance type, cost controls, rollback,
  secrets handling, model selection, and monitoring limitations are recorded.

## Viva Topics

Be ready to explain DVC versus Git, MLflow tracking versus model registry, Hyperopt,
the DVC stage graph, Docker image versus model artifact, ECR, EC2, IAM roles, health
checks, Prometheus metrics, drift detection, and the difference between a candidate
model and the deployed champion model.
