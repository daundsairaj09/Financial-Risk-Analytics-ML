from fastapi.testclient import TestClient
from src.api import app, load_models

# Initialize client
client = TestClient(app)

# Force load models for testing environment
load_models()

def test_health_check_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data

def test_prediction_success_valid_data():
    payload = {
        "person_age": 25,
        "person_income": 65000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 15000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.23,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "feature_importances" in data

def test_prediction_fails_invalid_age():
    payload = {
        "person_age": 12,  # Invalid: below 18
        "person_income": 65000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 15000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.23,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (Pydantic Validation Error)
    
def test_prediction_fails_missing_field():
    payload = {
        "person_age": 25,
        # Missing person_income entirely
        "person_home_ownership": "RENT",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
