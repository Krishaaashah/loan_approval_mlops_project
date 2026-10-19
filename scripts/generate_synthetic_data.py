"""Generate a deterministic synthetic loan dataset for local pipeline validation."""

import argparse
import csv
import os
import random
from pathlib import Path


FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


def make_row(rng, index):
    gender = rng.choice(["Male", "Female"])
    married = rng.choice(["Yes", "No"])
    dependents = rng.choice(["0", "1", "2", "3+"])
    education = rng.choice(["Graduate", "Not Graduate"])
    self_employed = rng.choice(["Yes", "No"])
    applicant_income = rng.randint(1800, 15000)
    coapplicant_income = rng.choice([0, rng.randint(500, 8000)])
    loan_amount = rng.randint(40, 480)
    loan_term = rng.choice([180, 240, 300, 360, 480])
    credit_history = rng.choice([0, 1, 1, 1])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"])

    total_income = applicant_income + coapplicant_income
    score = 0
    score += 4 if credit_history == 1 else -5
    score += 2 if total_income >= 7000 else -1
    score += 1 if education == "Graduate" else 0
    score += 1 if loan_amount <= total_income / 45 else -2
    score += 1 if property_area == "Semiurban" else 0
    score -= 1 if dependents == "3+" else 0
    if index % 17 == 0:
        score *= -1
    loan_status = "Y" if score >= 3 else "N"

    # A few missing values exercise the imputation stages without removing the
    # target label or creating an unusable row.
    if index % 29 == 0:
        self_employed = ""
    if index % 37 == 0:
        loan_term = ""
    if index % 43 == 0:
        credit_history = ""

    return {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    }


def generate_dataset(output_dir, train_size, test_size, seed):
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename, size, offset in [
        ("train.csv", train_size, 0),
        ("test.csv", test_size, train_size),
    ]:
        rows = [make_row(rng, offset + index) for index in range(size)]
        with (output_path / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FEATURES + ["Loan_Status"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join("prediction_model", "datasets"),
    )
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.train_size, args.test_size, args.seed)
