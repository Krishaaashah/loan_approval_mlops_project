from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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



@app.get("/")
def index():
    return {"message":"Welcome to the MLOps Loan Prediction app" }


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
