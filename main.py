import base64
import hashlib
import hmac
import json
import time

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
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
        <title>Loan Prediction API</title>
        <style>
          :root { color-scheme: light; font-family: system-ui, sans-serif; }
          body { margin: 0; background: #f3f7fa; color: #17324d; }
          main { max-width: 980px; margin: 36px auto; padding: 0 20px 48px; }
          header { margin-bottom: 22px; }
          h1 { margin: 0 0 8px; font-size: 32px; }
          p { color: #52677a; }
          .layout { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 22px; align-items: start; }
          section, aside { background: #fff; border: 1px solid #d9e4eb; border-radius: 10px; padding: 24px; box-shadow: 0 8px 24px rgba(23, 50, 77, .06); }
          h2 { margin-top: 0; font-size: 21px; }
          .fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }
          label { display: grid; gap: 6px; font-size: 14px; font-weight: 650; }
          input, select { width: 100%; box-sizing: border-box; padding: 10px 11px; border: 1px solid #b9cad6; border-radius: 6px; background: #fff; color: #17324d; font: inherit; }
          input:focus, select:focus { outline: 3px solid #cce8fb; border-color: #1769aa; }
          button { margin-top: 20px; width: 100%; padding: 12px 16px; border: 0; border-radius: 6px; background: #1769aa; color: white; font: inherit; font-weight: 700; cursor: pointer; }
          button:hover { background: #125689; }
          button:disabled { opacity: .6; cursor: wait; }
          #result { min-height: 24px; margin: 18px 0 0; padding: 12px; border-radius: 6px; font-weight: 700; }
          #result.approved { background: #e5f6ed; color: #146c3a; }
          #result.rejected { background: #fdeaea; color: #9a2525; }
          #result.error { background: #fff4df; color: #805b12; }
          aside { line-height: 1.7; }
          aside a { color: #1769aa; }
          code { background: #eef4f8; padding: 2px 5px; border-radius: 4px; }
          @media (max-width: 760px) { .layout { grid-template-columns: 1fr; } .fields { grid-template-columns: 1fr; } }
        </style>
      </head>
      <body>
        <main>
          <header>
            <h1>Loan Approval Prediction</h1>
            <p>Enter applicant details to receive a prediction from the deployed XGBoost model.</p>
          </header>
          <div class="layout">
            <section>
              <h2>Applicant Details</h2>
              <form id="prediction-form">
                <div class="fields">
                  <label>Gender<select name="Gender"><option>Male</option><option>Female</option></select></label>
                  <label>Married<select name="Married"><option>Yes</option><option>No</option></select></label>
                  <label>Dependents<select name="Dependents"><option>0</option><option>1</option><option>2</option><option>3+</option></select></label>
                  <label>Education<select name="Education"><option>Graduate</option><option>Not Graduate</option></select></label>
                  <label>Self Employed<select name="Self_Employed"><option>No</option><option>Yes</option></select></label>
                  <label>Applicant Income<input name="ApplicantIncome" type="number" min="0" value="5000" required></label>
                  <label>Coapplicant Income<input name="CoapplicantIncome" type="number" min="0" value="1500" required></label>
                  <label>Loan Amount<input name="LoanAmount" type="number" min="0" value="150" required></label>
                  <label>Loan Term (months)<input name="Loan_Amount_Term" type="number" min="1" value="360" required></label>
                  <label>Credit History<select name="Credit_History"><option value="1">1 - Meets requirements</option><option value="0">0 - Does not meet requirements</option></select></label>
                  <label>Property Area<select name="Property_Area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select></label>
                </div>
                <button id="submit-button" type="submit">Predict Loan Approval</button>
                <div id="result" role="status" aria-live="polite"></div>
              </form>
            </section>
            <aside>
              <h2>Service Links</h2>
              <p><a href="/docs">Swagger API documentation</a></p>
              <p><a href="/health">Health check</a></p>
              <p><a href="/ready">Model readiness</a></p>
              <p><a href="/metrics">Prometheus metrics</a></p>
              <p>API route: <code>POST /prediction_api</code></p>
            </aside>
          </div>
        </main>
        <script>
          const form = document.getElementById('prediction-form');
          const result = document.getElementById('result');
          const button = document.getElementById('submit-button');
          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            button.disabled = true;
            result.className = '';
            result.textContent = 'Calculating prediction...';
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
              const positive = (payload.positive_factors || []).map(item => item.feature).join(', ') || 'None';
              const negative = (payload.negative_factors || []).map(item => item.feature).join(', ') || 'None';
              result.innerHTML = `<strong>Prediction: ${payload.status}</strong><br>` +
                `Approval probability: ${(payload.approval_probability * 100).toFixed(1)}%<br>` +
                `Confidence: ${payload.confidence}<br>` +
                `Positive factors: ${positive}<br>Negative factors: ${negative}`;
              result.className = payload.status === 'Approved' ? 'approved' : 'rejected';
            } catch (error) {
              result.textContent = `Error: ${error.message}`;
              result.className = 'error';
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
def predict(loan_details: LoanPrediction, _: bool | None = Depends(require_auth)):
    data = loan_details.model_dump()
    details = generate_prediction_details([data])
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
    _: bool | None = Depends(require_auth)):

    input_data = [Gender, Married,Dependents, Education, Self_Employed,ApplicantIncome,
     CoapplicantIncome,LoanAmount, Loan_Amount_Term,Credit_History, Property_Area  ]

    cols = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    data_dict = dict(zip(cols,input_data))
    details = generate_prediction_details([data_dict])
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
