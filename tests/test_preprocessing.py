import pandas as pd

from prediction_model.config import config
from prediction_model.pipeline import preprocessing_pipeline


def test_preprocessing_pipeline_returns_numeric_features():
    data = pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1000,
                "LoanAmount": 120,
                "Loan_Amount_Term": 360,
                "Credit_History": 1,
                "Property_Area": "Urban",
            },
            {
                "Gender": "Female",
                "Married": "No",
                "Dependents": "1",
                "Education": "Not Graduate",
                "Self_Employed": "Yes",
                "ApplicantIncome": 3000,
                "CoapplicantIncome": 0,
                "LoanAmount": 80,
                "Loan_Amount_Term": 180,
                "Credit_History": 0,
                "Property_Area": "Rural",
            },
        ]
    )

    transformed = preprocessing_pipeline.fit_transform(data)

    assert transformed.shape == (
        len(data),
        len(config.FEATURES) - len(config.DROP_FEATURES),
    )
    assert transformed.dtype.kind in "fc"
