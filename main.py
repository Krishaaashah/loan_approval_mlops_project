from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import generate_predictions, generate_predictions_batch, get_model
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
              const response = await fetch('/prediction_api', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(values)
              });
              const payload = await response.json();
              if (!response.ok) throw new Error(payload.detail || 'Prediction request failed');
              result.textContent = `Prediction: ${payload.status}`;
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


@app.get("/ready")
def readiness():
    try:
        get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")
    return {"status": "ready"}

@app.post("/prediction_api")
def predict(loan_details: LoanPrediction):
    data = loan_details.model_dump()
    prediction = generate_predictions([data])["prediction"][0]
    if prediction == "Y":
        pred = "Approved"
    else:
        pred = "Rejected"
    return {"status":pred}




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
    Property_Area: str):

    input_data = [Gender, Married,Dependents, Education, Self_Employed,ApplicantIncome,
     CoapplicantIncome,LoanAmount, Loan_Amount_Term,Credit_History, Property_Area  ]

    cols = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    data_dict = dict(zip(cols,input_data))
    prediction = generate_predictions([data_dict])["prediction"][0]
    if prediction == "Y":
        pred = "Approved"
    else:
        pred = "Rejected"
    return {"status":pred}



@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):

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
