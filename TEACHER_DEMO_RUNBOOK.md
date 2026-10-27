# Teacher Demonstration Runbook

## Current Status

The cloud service is running in `ap-south-1`.

- API: `http://43.205.216.66/`
- FastAPI docs: `http://43.205.216.66/docs`
- Health: `http://43.205.216.66/health`
- Readiness: `http://43.205.216.66/ready`
- Metrics: `http://43.205.216.66/metrics`
- Grafana: `http://43.205.216.66/grafana/`
- Grafana dashboard: `Loan Approval API Monitoring`

The timeout was caused by the EC2 security group allowing the old public IP `219.91.213.185`, while the current public IP was `123.201.90.57`. The current IP has been added for ports 80, 3000, and 22. If the internet connection changes, the IP rule may need updating again.

## Demonstration Order

### 1. Explain the problem

Say that loan approval is not only a classification problem. Data, preprocessing, experiments, deployment, prediction explanations, monitoring, and retraining must also be repeatable.

Why show it: this establishes why MLOps is needed instead of presenting only a notebook model.

### 2. Show the repository

Show these folders and files:

- `prediction_model/`: preprocessing, training, evaluation, and explanations.
- `artifacts/`: model and metric outputs.
- `scripts/`: synthetic data, fairness, drift, and production-event utilities.
- `monitoring/`: Prometheus and Grafana configuration.
- `.github/workflows/`: CI/CD and drift monitoring.
- `Dockerfile`: reproducible runtime packaging.
- `dvc.yaml` and `dvc.lock`: reproducible pipeline definition.
- `RUBRIC_CHECKLIST.md`: CO1-CO4 evidence mapping.

Why show it: it proves that the work is an organized engineering project.

### 3. Explain the dataset

Show the synthetic dataset or generator. Explain that it contains income, loan amount, loan term, credit history, education, employment, marital status, dependents, and property area. The target is `Loan_Status`, approved or rejected.

Why show it: synthetic data protects privacy while allowing the complete pipeline and monitoring workflow to be demonstrated. State that it is not a real lending dataset.

### 4. Demonstrate DVC

Run:

```powershell
dvc dag
dvc repro
```

Show the DVC graph, model artifact, and `artifacts/metrics.json`.

Why show it: DVC proves that data preparation, training, and evaluation are repeatable from declared stages.

### 5. Demonstrate MLflow

Show the MLflow run comparison page or screenshot. Point out candidate runs, hyperparameters, metrics, and the selected best run.

Why show it: MLflow proves that experiments are recorded and comparable instead of selecting a model without evidence.

### 6. Show evaluation results

Show `artifacts/metrics.json`:

```text
Accuracy: 0.9500
Precision: 0.9444
Recall: 0.9714
F1-score: 0.9577
ROC-AUC: 0.9566
Brier score: 0.0532
```

Why show it: these are held-out test-set measures of classification quality and probability calibration.

### 7. Show the XGBoost regressor comparison

Show the auxiliary values:

```text
MAE: 0.1316
RMSE: 0.2442
R2: 0.7547
```

Explain that the classifier remains the decision model. The regressor is an auxiliary numerical credibility check.

### 8. Show CI/CD

Open the successful GitHub Actions run. Point out testing, Docker image build, ECR push, and EC2 deployment.

Why show it: it proves that a tested commit can become a repeatable deployment.

### 9. Explain the cloud architecture

Show this flow:

```text
GitHub Actions -> Amazon ECR -> EC2 Docker container -> FastAPI -> Client
                                                   |
                                                   +-> Prometheus -> Grafana
                                                   +-> S3 production events
```

Why show it: this directly demonstrates cloud services and deployment for CO1.

### 10. Show the prediction interface

Open `http://43.205.216.66/` and submit an applicant record. Then open `http://43.205.216.66/docs`.

Why show it: the browser page proves user interaction, and `/docs` proves that the API contract is documented.

### 11. Show a prediction response

The verified sample response contains:

```text
Decision: Approved
Approval probability: 0.740694
Rejection probability: 0.259306
Confidence: Medium
Explanation method: SHAP TreeExplainer
```

Positive factors include `Credit_History`, `LoanAmount`, and `Education`. Negative factors include `ApplicantIncome`, `Property_Area`, and `Loan_Amount_Term`.

Why show it: it proves that the API returns both a decision and an explanation of the decision.

### 12. Show health and readiness

Open:

```text
http://43.205.216.66/health
http://43.205.216.66/ready
```

Expected output:

```json
{"status":"ok"}
{"status":"ready"}
```

Why show it: `/health` proves that the process responds; `/ready` proves that the model is loaded and ready.

### 13. Show Prometheus metrics

Open `http://43.205.216.66/metrics`. Point out request count, latency, errors, and process metrics.

Why show it: this is the machine-readable monitoring source consumed by Grafana.

### 14. Show Grafana

Open `http://43.205.216.66/grafana/` and open `Loan Approval API Monitoring`. Show request rate, P95 latency, HTTP errors, total requests, and process uptime.

Why show it: Grafana turns raw metrics into an operational dashboard.

### 15. Explain production drift and retraining

Show this flow:

```text
Prediction request -> private S3 event -> scheduled drift job -> PSI check
                                                  -> gated retraining
```

Explain that retraining occurs only when the production sample is large enough and drift crosses the configured threshold. Otherwise the workflow exits safely.

Why show it: it proves that the project continues after deployment and addresses CO3 monitoring and optimization.

### 16. Explain fairness and governance

Show the fairness report and explain that approval-rate comparisons are generated for gender, marital status, education, property area, and self-employment.

State that these are audit signals, not proof of social neutrality. Real lending would require representative data, legal review, privacy controls, and human oversight.

## Rubric Mapping

| Area | Show | Purpose |
|---|---|---|
| CO1 | EC2, ECR, S3, API, Grafana | Proves cloud services are integrated. |
| CO2 | DVC, GitHub Actions, Docker | Proves repeatable pipeline and CI/CD. |
| CO3 | MLflow, metrics, Grafana, drift job | Proves tracking, monitoring, and optimization. |
| CO4 | README, runbook, `/docs`, screenshots | Proves documentation and presentation quality. |

## Troubleshooting

### Grafana timeout

The likely cause is a changed public IP. Check it:

```powershell
(Invoke-RestMethod https://checkip.amazonaws.com).Trim()
```

The dashboard now uses the same port-80 entry point as the API, so no separate port
3000 rule is required. If the API itself is reachable but Grafana is not, verify port
80 is allowed for the current public IP and open `/grafana/` rather than the old
direct port.

```powershell
Invoke-WebRequest http://43.205.216.66/grafana/api/health
```

### API does not open

Test `http://43.205.216.66/health`. If it fails, check the EC2 instance state, port 80 rule, and API container health.

### `/ready` fails

The API process is alive but the model cannot be loaded. Check the deployed image tag, model artifact, and container logs.

### Grafana panels are empty

Check that Prometheus is running, Prometheus can scrape `/metrics`, and Grafana uses the Prometheus data source.

## Limitation to State Honestly

Authentication is currently disabled for this academic demonstration. The API has opt-in authentication code, but the deployed demo is protected primarily by the EC2 security-group restriction. Production hardening would add authentication, HTTPS, managed secrets, stronger network boundaries, and legal/fairness validation.
