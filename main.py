import base64
import hashlib
import hmac
import json
import time
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import generate_predictions_batch, generate_prediction_details, get_model
from prediction_model.config import config
import io
from datetime import datetime

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None


def upload_to_s3(file_content, filename):
    import boto3

    s3 = boto3.client('s3')

    current_date = datetime.now().strftime("%Y-%m-%d")
    if filename.endswith('.csv'):
        filename = filename[:-4]

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder_path = f"{config.FOLDER}/{current_date}"

    filename_with_datetime = f"{filename}_{current_datetime}.csv"

    s3_key = f"{folder_path}/{filename_with_datetime}"

    s3.put_object(Bucket=config.S3_BUCKET, Key=s3_key, Body=file_content)

    return s3_key


def record_production_event(data: dict, details: dict) -> None:
    """Store a prediction input for drift monitoring without blocking inference."""
    if not config.PRODUCTION_LOG_ENABLED:
        return

    event = {
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "request_id": uuid.uuid4().hex,
        "features": {feature: data.get(feature) for feature in config.FEATURES},
        "prediction": details.get("decision"),
        "approval_probability": details.get("approval_probability"),
    }
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    current_hour = datetime.utcnow().strftime("%H")
    key = f"{config.PRODUCTION_PREFIX}/{current_date}/{current_hour}/{event['request_id']}.json"

    try:
        import boto3

        boto3.client("s3").put_object(
            Bucket=config.PRODUCTION_BUCKET,
            Key=key,
            Body=json.dumps(event).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        # Monitoring must never make a valid prediction fail.
        print(f"production drift event upload failed: {exc}")

if mlflow is not None:
    mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(
    title="Loan Prediction App using FastAPI - MLOps",
    description = "MLOps Demo",
    version='1.0'
)

origins = config.ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=bool(origins),
    allow_methods=["*"],
    allow_headers=["*"]
)

if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app)

bearer = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


def _encode_token(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    signature = hmac.new(config.AUTH_SECRET.encode(), body, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return f"{body.decode()}.{encoded_signature.decode()}"


def _valid_token(token: str) -> bool:
    try:
        encoded_body, encoded_signature = token.split(".", 1)
        expected = hmac.new(config.AUTH_SECRET.encode(), encoded_body.encode(), hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
        payload = json.loads(base64.urlsafe_b64decode(encoded_body + "=" * (-len(encoded_body) % 4)))
        return hmac.compare_digest(expected, supplied) and payload["exp"] > int(time.time())
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if not config.AUTH_ENABLED:
        return None
    if credentials is None or credentials.scheme.lower() != "bearer" or not _valid_token(credentials.credentials):
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

class LoanPrediction(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str



@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="theme-color" content="#0b1220">
        <title>Loan Approval MLOps | Decision Console</title>
        <style>
          :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; --ink: #172033; --muted: #6d7890; --line: #e5eaf2; --navy: #101a31; --blue: #4263eb; --cyan: #1bb7c9; --green: #16805c; --red: #bd3f50; --surface: #ffffff; }
          * { box-sizing: border-box; }
          body { margin: 0; color: var(--ink); background: radial-gradient(circle at 8% 0%, #e7f3ff 0, transparent 34%), radial-gradient(circle at 100% 20%, #e8fbf5 0, transparent 30%), #f7f9fc; }
          a { color: inherit; }
          .shell { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 26px 0 42px; }
          .topbar { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 34px; }
          .brand, .top-actions, .brand-copy { display: flex; align-items: center; }
          .brand { gap: 12px; text-decoration: none; }
          .brand-mark { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 13px; color: white; font-size: 15px; font-weight: 850; letter-spacing: -.04em; background: linear-gradient(145deg, #536df4, #1aaec4); box-shadow: 0 10px 22px rgba(66, 99, 235, .25); }
          .brand-copy { align-items: flex-start; flex-direction: column; gap: 2px; }
          .brand-copy strong { font-size: 15px; letter-spacing: -.02em; }
          .kicker, .eyebrow { color: var(--blue); font-size: 10px; font-weight: 850; letter-spacing: .14em; text-transform: uppercase; }
          .top-actions { gap: 18px; font-size: 13px; font-weight: 700; }
          .top-actions a { color: #4e5e78; text-decoration: none; }
          .top-actions a:hover { color: var(--blue); }
          .live-pill, .tag { display: inline-flex; align-items: center; gap: 7px; border: 1px solid #ccefe2; border-radius: 999px; padding: 7px 11px; color: var(--green); background: #f0fbf6; font-size: 11px; font-weight: 800; }
          .live-dot { width: 7px; height: 7px; border-radius: 50%; background: #1ec98b; box-shadow: 0 0 0 4px #d6f6e8; }
          .hero { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 30px; align-items: end; padding: 30px 34px; border-radius: 25px; color: white; background: linear-gradient(120deg, #0e1830 0%, #172b56 62%, #185e78 100%); box-shadow: 0 22px 50px rgba(24, 43, 86, .18); overflow: hidden; position: relative; }
          .hero::after { content: ""; position: absolute; width: 310px; height: 310px; right: -80px; top: -150px; border: 1px solid rgba(255,255,255,.18); border-radius: 50%; box-shadow: 0 0 0 30px rgba(255,255,255,.03), 0 0 0 60px rgba(255,255,255,.025); }
          .hero .eyebrow { color: #7ee7e5; }
          h1 { margin: 8px 0 10px; max-width: 650px; font-size: clamp(29px, 4vw, 48px); line-height: 1.05; letter-spacing: -.055em; }
          .hero p { max-width: 610px; margin: 0; color: #b9c7df; font-size: 15px; line-height: 1.65; }
          .hero-meta { z-index: 1; display: grid; grid-template-columns: repeat(2, 132px); gap: 10px; }
          .hero-stat { padding: 14px; border: 1px solid rgba(255,255,255,.14); border-radius: 15px; background: rgba(255,255,255,.08); }
          .hero-stat span { display: block; color: #aab9d3; font-size: 10px; text-transform: uppercase; letter-spacing: .1em; }
          .hero-stat strong { display: block; margin-top: 7px; font-size: 14px; }
          .layout { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(290px, .65fr); gap: 22px; align-items: start; margin-top: 22px; }
          .card { border: 1px solid rgba(218, 225, 236, .9); border-radius: 21px; background: rgba(255,255,255,.94); box-shadow: 0 15px 38px rgba(34, 49, 82, .07); }
          .form-card { padding: 28px; }
          .card-head { display: flex; justify-content: space-between; align-items: start; gap: 16px; margin-bottom: 26px; }
          h2 { margin: 5px 0 0; font-size: 22px; letter-spacing: -.035em; }
          .required { color: var(--muted); font-size: 11px; font-weight: 700; }
          .group-title { display: flex; align-items: center; gap: 10px; margin: 22px 0 13px; color: #34415b; font-size: 12px; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }
          .group-title::after { content: ""; height: 1px; flex: 1; background: var(--line); }
          .fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
          label { display: grid; gap: 7px; color: #4c5870; font-size: 12px; font-weight: 800; }
          input, select { width: 100%; min-height: 44px; border: 1px solid #d9e0eb; border-radius: 11px; padding: 10px 12px; color: var(--ink); background: #fbfcfe; font: inherit; transition: border .2s, box-shadow .2s, background .2s; }
          input:hover, select:hover { background: #fff; border-color: #b8c5dc; }
          input:focus, select:focus { outline: 0; border-color: var(--blue); background: #fff; box-shadow: 0 0 0 4px rgba(66, 99, 235, .12); }
          .actions { display: grid; grid-template-columns: 1fr 110px; gap: 10px; margin-top: 25px; }
          button { min-height: 47px; border: 0; border-radius: 12px; font: inherit; font-weight: 850; cursor: pointer; transition: transform .15s, box-shadow .15s, background .15s; }
          .primary { display: flex; align-items: center; justify-content: space-between; padding: 0 17px; color: #fff; background: linear-gradient(110deg, #4263eb, #279bb6); box-shadow: 0 10px 20px rgba(66, 99, 235, .2); }
          .primary:hover { transform: translateY(-1px); box-shadow: 0 13px 24px rgba(66, 99, 235, .28); }
          .primary:disabled { opacity: .65; cursor: wait; transform: none; }
          .arrow { font-size: 20px; line-height: 1; }
          .secondary { color: #536079; border: 1px solid var(--line); background: #f8fafc; }
          .secondary:hover { background: #eef2f8; }
          #result { margin-top: 20px; }
          .result-empty, .result-error { padding: 18px; border: 1px dashed #d5ddea; border-radius: 15px; color: var(--muted); background: #fbfcfe; font-size: 13px; line-height: 1.5; }
          .result-error { border-style: solid; border-color: #f0c9cd; color: var(--red); background: #fff5f6; }
          .result-card { padding: 20px; border-radius: 16px; border: 1px solid #cfe9dd; background: linear-gradient(145deg, #f6fffb, #f0f8ff); }
          .result-card.rejected { border-color: #f1cfd4; background: linear-gradient(145deg, #fff8f8, #fff8f0); }
          .result-head, .probability-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
          .decision { margin-top: 5px; color: var(--green); font-size: 25px; font-weight: 900; letter-spacing: -.04em; }
          .rejected .decision { color: var(--red); }
          .confidence { padding: 7px 10px; border-radius: 999px; color: #78621b; background: #fff5cd; font-size: 11px; font-weight: 850; }
          .probability-row { margin-top: 22px; color: #536079; font-size: 12px; font-weight: 750; }
          .probability-row strong { color: var(--ink); font-size: 19px; }
          .bar { height: 9px; margin: 8px 0 20px; overflow: hidden; border-radius: 99px; background: #dfe8e7; }
          .bar span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #21b789, #5bd7b1); }
          .rejected .bar span { background: linear-gradient(90deg, #e06a74, #f4ad78); }
          .factor-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
          .factor-box { min-height: 78px; padding: 12px; border-radius: 12px; background: rgba(255,255,255,.75); }
          .factor-box.support { border: 1px solid #ccebdc; }
          .factor-box.risk { border: 1px solid #efd2d5; }
          .factor-label { display: block; margin-bottom: 8px; color: var(--muted); font-size: 10px; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }
          .chip { display: inline-block; margin: 0 4px 4px 0; padding: 5px 7px; border-radius: 7px; color: #34415b; background: #eef2f7; font-size: 11px; font-weight: 750; }
          .explanation-note { margin: 14px 0 0; color: #6a758a; font-size: 11px; }
          .side-stack { display: grid; gap: 16px; }
          .side-card { padding: 22px; }
          .side-card h3 { margin: 5px 0 12px; font-size: 17px; letter-spacing: -.025em; }
          .side-card p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.65; }
          .links { display: grid; gap: 8px; margin-top: 16px; }
          .service-link { display: flex; align-items: center; justify-content: space-between; padding: 11px 12px; border: 1px solid var(--line); border-radius: 11px; color: #43516a; text-decoration: none; font-size: 12px; font-weight: 800; background: #fbfcfe; }
          .service-link:hover { border-color: #b9c6df; color: var(--blue); background: #fff; }
          .service-link span:last-child { color: #9ca8ba; font-size: 16px; }
          .mini-metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 9px; margin-top: 16px; }
          .mini-metric { padding: 12px; border-radius: 12px; background: #f5f7fb; }
          .mini-metric span { display: block; color: var(--muted); font-size: 10px; font-weight: 750; }
          .mini-metric strong { display: block; margin-top: 5px; color: #2d3b59; font-size: 15px; }
          .footer { margin-top: 22px; color: #8a95a7; font-size: 11px; text-align: center; }
          code { padding: 3px 6px; border-radius: 6px; color: #44516a; background: #eef2f7; }
          @media (max-width: 820px) { .hero { grid-template-columns: 1fr; } .hero-meta { grid-template-columns: repeat(2, minmax(120px, 1fr)); max-width: 300px; } .layout { grid-template-columns: 1fr; } }
          @media (max-width: 560px) { .shell { width: min(100% - 24px, 1180px); padding-top: 16px; } .topbar { align-items: flex-start; } .top-actions { flex-direction: column; align-items: flex-end; gap: 8px; } .hero, .form-card, .side-card { padding: 20px; border-radius: 17px; } .fields, .factor-grid, .actions { grid-template-columns: 1fr; } h1 { font-size: 34px; } }
        </style>
      </head>
      <body>
        <div class="shell">
          <header class="topbar">
            <a class="brand" href="/">
              <span class="brand-mark">ML</span>
              <span class="brand-copy"><span class="kicker">Loan approval MLOps</span><strong>Decision Console</strong></span>
            </a>
            <div class="top-actions"><span class="live-pill"><span class="live-dot"></span>Live API</span><a href="/docs">API documentation</a></div>
          </header>
          <section class="hero">
            <div>
              <span class="eyebrow">Cloud-deployed inference</span>
              <h1>Make a loan decision with evidence.</h1>
              <p>Submit an applicant profile to the champion XGBoost model. The response includes probability, confidence, and the strongest factors behind the decision.</p>
            </div>
            <div class="hero-meta" aria-label="Deployment summary">
              <div class="hero-stat"><span>Model</span><strong>XGBoost</strong></div>
              <div class="hero-stat"><span>Region</span><strong>ap-south-1</strong></div>
            </div>
          </section>
          <main class="layout">
            <section class="card form-card">
              <div class="card-head">
                <div><span class="eyebrow">Applicant profile</span><h2>Enter application details</h2></div>
                <span class="required">All fields required</span>
              </div>
              <form id="prediction-form">
                <div class="group-title">Personal details</div>
                <div class="fields">
                  <label>Gender<select name="Gender"><option>Male</option><option>Female</option></select></label>
                  <label>Married<select name="Married"><option>Yes</option><option>No</option></select></label>
                  <label>Dependents<select name="Dependents"><option>0</option><option>1</option><option>2</option><option>3+</option></select></label>
                  <label>Education<select name="Education"><option>Graduate</option><option>Not Graduate</option></select></label>
                  <label>Self employed<select name="Self_Employed"><option>No</option><option>Yes</option></select></label>
                  <label>Property area<select name="Property_Area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select></label>
                </div>
                <div class="group-title">Financial profile</div>
                <div class="fields">
                  <label>Applicant income<input name="ApplicantIncome" type="number" min="0" value="5000" required></label>
                  <label>Coapplicant income<input name="CoapplicantIncome" type="number" min="0" value="1500" required></label>
                  <label>Loan amount<input name="LoanAmount" type="number" min="0" value="150" required></label>
                  <label>Loan term (months)<input name="Loan_Amount_Term" type="number" min="1" value="360" required></label>
                  <label>Credit history<select name="Credit_History"><option value="1">Meets requirements</option><option value="0">Does not meet requirements</option></select></label>
                </div>
                <div class="actions">
                  <button class="primary" id="submit-button" type="submit"><span>Run prediction</span><span class="arrow">→</span></button>
                  <button class="secondary" id="reset-button" type="button">Reset</button>
                </div>
                <div id="result" role="status" aria-live="polite">
                  <div class="result-empty">Your prediction, probability, confidence, and explanation factors will appear here.</div>
                </div>
              </form>
            </section>
            <aside class="side-stack">
              <section class="card side-card">
                <span class="eyebrow">Connected services</span>
                <h3>Inspect the MLOps system</h3>
                <p>Use these links during the demo to show the API contract, readiness, and live monitoring output.</p>
                <div class="links">
                  <a class="service-link" href="/docs"><span>Swagger API docs</span><span>↗</span></a>
                  <a class="service-link" href="/health"><span>Health check</span><span>↗</span></a>
                  <a class="service-link" href="/ready"><span>Model readiness</span><span>↗</span></a>
                  <a class="service-link" href="/metrics"><span>Prometheus metrics</span><span>↗</span></a>
                  <a class="service-link" href="/grafana/"><span>Grafana dashboard</span><span>↗</span></a>
                </div>
              </section>
              <section class="card side-card">
                <span class="eyebrow">Champion model</span>
                <h3>Evidence before decision</h3>
                <p>The selected XGBoost classifier is loaded from the versioned model artifact and returns local feature contributions with every prediction.</p>
                <div class="mini-metrics">
                  <div class="mini-metric"><span>Accuracy</span><strong>95.0%</strong></div>
                  <div class="mini-metric"><span>F1 score</span><strong>95.8%</strong></div>
                  <div class="mini-metric"><span>ROC-AUC</span><strong>95.7%</strong></div>
                  <div class="mini-metric"><span>Endpoint</span><strong>POST API</strong></div>
                </div>
              </section>
            </aside>
          </main>
          <div class="footer">Decision support demonstration • Model outputs should be reviewed by a qualified human.</div>
        </div>
        <script>
          const form = document.getElementById('prediction-form');
          const result = document.getElementById('result');
          const button = document.getElementById('submit-button');
          const resetButton = document.getElementById('reset-button');
          const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[character]));
          const factorChips = (items) => (items || []).map((item) => `<span class="chip">${escapeHtml(item.feature)}</span>`).join('') || '<span class="chip">None identified</span>';
          const emptyResult = '<div class="result-empty">Your prediction, probability, confidence, and explanation factors will appear here.</div>';
          resetButton.addEventListener('click', () => { form.reset(); result.className = ''; result.innerHTML = emptyResult; });
          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            button.disabled = true;
            result.className = '';
            result.innerHTML = '<div class="result-empty">Calculating prediction and explanation...</div>';
            const values = Object.fromEntries(new FormData(form).entries());
            ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
              .forEach((key) => { values[key] = Number(values[key]); });
            try {
              const headers = { 'Content-Type': 'application/json' };
              const savedToken = window.localStorage.getItem('loan_prediction_token');
              if (savedToken) headers.Authorization = `Bearer ${savedToken}`;
              let response = await fetch('/prediction_api', {
                method: 'POST',
                headers,
                body: JSON.stringify(values)
              });
              if (response.status === 401) {
                const username = window.prompt('Username');
                const password = window.prompt('Password');
                const login = await fetch('/auth/login', {
                  method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ username, password })
                });
                if (!login.ok) throw new Error('Login failed');
                const auth = await login.json();
                window.localStorage.setItem('loan_prediction_token', auth.access_token);
                headers.Authorization = `Bearer ${auth.access_token}`;
                response = await fetch('/prediction_api', { method: 'POST', headers, body: JSON.stringify(values) });
              }
              const payload = await response.json();
              if (!response.ok) throw new Error(payload.detail || 'Prediction request failed');
              const probability = Math.max(0, Math.min(100, Number(payload.approval_probability || 0) * 100));
              const decision = escapeHtml(payload.status);
              result.innerHTML = `<div class="result-card ${payload.status === 'Approved' ? '' : 'rejected'}">
                <div class="result-head"><div><span class="eyebrow">Model decision</span><div class="decision">${decision}</div></div><span class="confidence">${escapeHtml(payload.confidence)} confidence</span></div>
                <div class="probability-row"><span>Approval probability</span><strong>${probability.toFixed(1)}%</strong></div>
                <div class="bar" aria-label="Approval probability"><span style="width:${probability}%"></span></div>
                <div class="factor-grid"><div class="factor-box support"><span class="factor-label">Supports decision</span>${factorChips(payload.positive_factors)}</div><div class="factor-box risk"><span class="factor-label">Works against decision</span>${factorChips(payload.negative_factors)}</div></div>
                <p class="explanation-note">Explanation method: ${escapeHtml(payload.explanation_method || 'Model feature contributions')}</p>
              </div>`;
            } catch (error) {
              result.innerHTML = `<div class="result-error">${escapeHtml(error.message)}</div>`;
            } finally {
              button.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth/status")
def auth_status():
    return {"enabled": config.AUTH_ENABLED}


@app.post("/auth/login")
def login(credentials: LoginRequest):
    if not config.AUTH_ENABLED:
        return {"enabled": False, "access_token": "", "token_type": "bearer"}
    if not hmac.compare_digest(credentials.username, config.AUTH_USERNAME) or not hmac.compare_digest(
        credentials.password, config.AUTH_PASSWORD
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _encode_token(
        {"sub": config.AUTH_USERNAME, "exp": int(time.time()) + config.AUTH_TOKEN_TTL_SECONDS}
    )
    return {"enabled": True, "access_token": token, "token_type": "bearer"}


@app.get("/ready")
def readiness():
    try:
        get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")
    return {"status": "ready"}

@app.post("/prediction_api")
def predict(
    loan_details: LoanPrediction,
    background_tasks: BackgroundTasks,
    _: bool | None = Depends(require_auth),
):
    data = loan_details.model_dump()
    details = generate_prediction_details([data])
    background_tasks.add_task(record_production_event, data, details)
    return {"status": details["decision"], **details}




@app.post("/prediction_ui")
def predict_gui(Gender: str,
    Married: str,
    Dependents: str,
    Education: str,
    Self_Employed: str,
    ApplicantIncome: float,
    CoapplicantIncome: float,
    LoanAmount: float,
    Loan_Amount_Term: float,
    Credit_History: float,
    Property_Area: str,
    background_tasks: BackgroundTasks,
    _: bool | None = Depends(require_auth)):

    input_data = [Gender, Married,Dependents, Education, Self_Employed,ApplicantIncome,
     CoapplicantIncome,LoanAmount, Loan_Amount_Term,Credit_History, Property_Area  ]

    cols = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    data_dict = dict(zip(cols,input_data))
    details = generate_prediction_details([data_dict])
    background_tasks.add_task(record_production_event, data_dict, details)
    return {"status": details["decision"], **details}



@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...), _: bool | None = Depends(require_auth)):

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content),index_col=False)
    print(df)

    # Ensure the CSV file contains the required features
    required_columns = config.FEATURES
    if not all(column in df.columns for column in required_columns):
        raise HTTPException(
            status_code=400,
            detail="CSV file does not contain the required columns.",
        )

    predictions = generate_predictions_batch(df)["prediction"]

    df['Prediction'] = predictions
    result = df.to_csv(index=False)

    if config.UPLOAD_PREDICTIONS:
        upload_to_s3(result.encode("utf-8"), file.filename or "predictions.csv")

    return StreamingResponse(io.BytesIO(result.encode('utf-8')), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=predictions.csv"})



if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0",port=8005)
