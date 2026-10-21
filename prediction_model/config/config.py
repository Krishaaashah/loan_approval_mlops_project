import os


current_directory = os.path.dirname(os.path.realpath(__file__))

PACKAGE_ROOT = os.path.dirname(current_directory)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.getenv("MLOPS_DATA_DIR", os.path.join(PACKAGE_ROOT, "datasets"))

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

TARGET = 'Loan_Status'

#Final features used in the model
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] # taking log of numerical columns

S3_BUCKET = os.getenv("MLOPS_S3_BUCKET", "loanprediction")

FOLDER = os.getenv("MLOPS_DRIFT_PREFIX", "datadrift")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./artifacts/mlruns")


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "loan_prediction_model")

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "Loanprediction-model")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "").strip()
MODEL_URI = os.getenv("MODEL_URI", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "model.joblib"))
PREPARED_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
PREPARED_TEST_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
BEST_PARAMS_PATH = os.path.join(ARTIFACTS_DIR, "best_params.json")
MIN_F1_SCORE = float(os.getenv("MIN_F1_SCORE", "0.0"))
UPLOAD_PREDICTIONS = os.getenv("MLOPS_UPLOAD_PREDICTIONS", "false").lower() in {
    "1",
    "true",
    "yes",
}

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("MLOPS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

# Authentication is opt-in so existing local demos remain usable. Production
# deployments should provide all three values through a secret manager.
AUTH_USERNAME = os.getenv("MLOPS_AUTH_USERNAME", "").strip()
AUTH_PASSWORD = os.getenv("MLOPS_AUTH_PASSWORD", "")
AUTH_SECRET = os.getenv("MLOPS_AUTH_SECRET", "")
AUTH_TOKEN_TTL_SECONDS = int(os.getenv("MLOPS_AUTH_TOKEN_TTL_SECONDS", "3600"))
AUTH_ENABLED = bool(AUTH_USERNAME and AUTH_PASSWORD and AUTH_SECRET)

DRIFT_REPORT_PATH = os.getenv(
    "MLOPS_DRIFT_REPORT_PATH", os.path.join(ARTIFACTS_DIR, "drift_report.json")
)

# Production feature logging is opt-in. The deployed EC2 instance enables it so
# drift checks can use real prediction inputs from the existing S3 bucket.
PRODUCTION_LOG_ENABLED = os.getenv("MLOPS_PRODUCTION_LOG_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
PRODUCTION_BUCKET = os.getenv("MLOPS_PRODUCTION_BUCKET", S3_BUCKET).strip()
PRODUCTION_PREFIX = os.getenv("MLOPS_PRODUCTION_PREFIX", "production/predictions").strip("/")
