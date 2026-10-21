import numpy as np
import pytest

from prediction_model import predict


class FakeModel:
    def predict(self, data):
        return np.ones(len(data), dtype=int)


@pytest.fixture(autouse=True)
def local_model(monkeypatch):
    monkeypatch.setattr(predict, "get_model", lambda: FakeModel())


def test_single_prediction_returns_serializable_string():
    result = predict.generate_predictions([{"ApplicantIncome": 1000}])

    assert result == {"prediction": ["Y"]}


def test_batch_prediction_returns_one_result_per_row():
    result = predict.generate_predictions_batch(
        [{"ApplicantIncome": 1000}, {"ApplicantIncome": 2000}]
    )

    assert result["prediction"] == ["Y", "Y"]


def test_prediction_details_include_confidence_and_explanation():
    result = predict.generate_prediction_details([{"ApplicantIncome": 1000}])

    assert result["decision"] == "Approved"
    assert 0 <= result["approval_probability"] <= 1
    assert result["confidence"] in {"Low", "Medium", "High"}
    assert "positive_factors" in result
    assert "negative_factors" in result
