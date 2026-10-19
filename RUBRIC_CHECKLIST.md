# Rubric Completion Checklist

## CO1: Cloud Services and MLOps Fundamentals

Local work completed:

- DVC remote configuration exists in `.dvc/config`.
- The application reads S3 and MLflow settings from environment variables.
- Docker no longer receives AWS credentials during image creation.
- The API can run locally without S3 uploads.
- Kubernetes deployment manifest uses an immutable image-tag placeholder and health probes.
- Rollback and secrets-handling guidance is documented in `DEPLOYMENT_RUNBOOK.md`.

Cloud evidence still required:

- S3 DVC remote successfully configured and demonstrated with `dvc pull`/`dvc push`.
- EC2 instance with an instance role and restricted security group.
- ECR repository containing an immutable image tag.
- CloudWatch logs and a successful public or private client request.

## CO2: ML Pipeline and CI/CD

Local work completed:

- `dvc.yaml` defines prepare, train, and evaluate stages.
- Hyperopt searches multiple candidates.
- Evaluation writes `artifacts/metrics.json`.
- GitHub Actions runs tests and builds the container.

Cloud-phase work:

- Add the ECR push and EC2 deployment job after AWS is configured.
- Deploy using an immutable image tag, not `latest`.
- Add a post-deployment `/health` or `/ready` smoke test.
- Add rollback instructions and demonstrate one successful deployment.

## CO3: Tracking, Monitoring, and Optimization

Local work completed:

- MLflow tracks candidate runs and metrics.
- Hyperopt selects the best candidate by F1 score.
- The selected model is persisted as `artifacts/model.joblib`.
- Optional MLflow model registration is supported.
- The API caches one configured model instead of searching MLflow per request.
- `/metrics` and Evidently drift-monitoring code are available.
- A local Evidently report command is available at `scripts/generate_local_drift_report.py`.

Remaining evidence:

- Run the pipeline on the real DVC dataset and record the metrics.
- Show the MLflow experiment and selected model version.
- Connect Prometheus/Grafana or CloudWatch and capture response-rate/latency/error graphs.
- Demonstrate a drift report and the action taken when a threshold is exceeded.

## CO4: Documentation, Deployment Strategy, and Presentation

Completed:

- Local setup and configuration are documented in `README.md`.
- The local architecture and DVC workflow are documented.
- Six automated tests cover preprocessing and API behavior.
- The Docker image has a health check and runs as a non-root user.

Before submission:

- Add screenshots of the DVC graph, MLflow runs, CI result, deployed API, and monitoring dashboard.
- Record the exact AWS services, IAM permissions, instance type, region, and estimated cost.
- Explain rollback, secrets handling, model version selection, and monitoring limitations.

## Viva Topics

Be ready to explain DVC versus Git, MLflow tracking versus model registry, Hyperopt,
the DVC stage graph, Docker image versus model artifact, ECR, EC2, IAM roles, health
checks, Prometheus metrics, drift detection, and the difference between a candidate
model and the deployed champion model.
