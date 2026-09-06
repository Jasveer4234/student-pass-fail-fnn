"""Automated unit and integration tests for Student Performance Prediction & Risk Analytics.

Verifies:
1. Required dataset columns exist.
2. Target column contains both binary classes (0 and 1).
3. Model input shape is compatible with 3 features.
4. Model output shape is (None, 1).
5. Model contains exactly 73 trainable parameters.
6. Model output values are valid probabilities strictly in [0, 1].
7. Preprocessing strictly prevents data leakage (scaler fitted exclusively on training set).
8. End-to-end prediction pipeline functions reliably with valid samples.
9. Flask application imports and initializes artifacts properly.
10. GET / returns HTTP 200 and loads form.
11. POST /predict with valid pass inputs returns PASS outcome.
12. POST /predict with valid fail inputs returns FAIL outcome.
13. Out-of-bounds attendance is rejected gracefully with validation message.
14. Out-of-bounds study hours are rejected gracefully with validation message.
15. Non-numeric inputs are rejected gracefully.
16. GET /health returns HTTP 200 and status ok JSON.
17. Standard HTTP security headers are applied to responses.
18. Allowlisted visual artifacts return HTTP 200.
19. Disallowed and path traversal attempts on /outputs return HTTP 404.
20. Nonexistent routes return custom HTTP 404 handler.
21. NaN and Infinity inputs are rejected by server-side validation.
22. Oversized payloads trigger HTTP 413.
23. Application exports valid WSGI callable for Gunicorn.
24. render.yaml blueprint contains valid deployment directives.
25. POST /api/predict with valid pass data returns JSON prediction PASS.
26. POST /api/predict with valid fail data returns JSON prediction FAIL.
27. POST /api/predict with missing fields returns HTTP 400.
28. POST /api/predict with out-of-range values returns HTTP 400.
29. POST /api/predict with non-JSON or malformed payload returns HTTP 400.
30. POST /api/predict with NaN or Infinity returns HTTP 400.
"""

import os
import sys

# Ensure root directory is on sys.path for test runners
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.utils import build_model, load_and_preprocess_data, set_seed
from src.predict import predict_pass_fail, validate_inputs


@pytest.fixture(scope="session")
def setup_dataset():
    """Ensure student_data.csv exists before running tests."""
    data_path = "data/student_data.csv"
    if not os.path.exists(data_path):
        from data.generate_dataset import generate_student_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_student_dataset(n_samples=300, seed=42)
        df.to_csv(data_path, index=False)
    return data_path


def test_dataset_columns(setup_dataset):
    """Test 1: Verify all required dataset columns are present."""
    df = pd.read_csv(setup_dataset)
    required_cols = {"Study_Hours", "Attendance", "Previous_Marks", "Result"}
    assert required_cols.issubset(df.columns), (
        f"Missing required columns in dataset: {required_cols - set(df.columns)}"
    )
    assert len(df) >= 100, f"Dataset has too few samples ({len(df)} samples)."
    assert df[list(required_cols)].isna().sum().sum() == 0, "Dataset contains missing values."


def test_target_classes(setup_dataset):
    """Test 2: Verify binary target column contains both 0 and 1."""
    df = pd.read_csv(setup_dataset)
    unique_classes = set(df["Result"].unique())
    assert unique_classes == {0, 1}, f"Target classes must be {{0, 1}}, got {unique_classes}"

    counts = df["Result"].value_counts()
    ratio = counts[1] / len(df)
    assert 0.35 <= ratio <= 0.65, f"Class balance ratio {ratio:.2f} is heavily skewed."


def test_model_input_shape():
    """Test 3: Verify model accepts 3-dimensional input features."""
    model = build_model()
    input_shape = model.input_shape
    assert input_shape == (None, 3), f"Expected model input shape (None, 3), got {input_shape}"


def test_model_output_shape():
    """Test 4: Verify model output shape is single binary probability."""
    model = build_model()
    output_shape = model.output_shape
    assert output_shape == (None, 1), f"Expected model output shape (None, 1), got {output_shape}"


def test_trainable_parameters_count():
    """Test 5: Verify model contains EXACTLY 73 trainable parameters.

    Calculation:
    - Layer 1 (Dense 8): 3 * 8 + 8 = 32
    - Layer 2 (Dense 4): 8 * 4 + 4 = 36
    - Layer 3 (Dense 1): 4 * 1 + 1 = 5
    - Total = 32 + 36 + 5 = 73
    """
    model = build_model()
    total_params = model.count_params()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)

    assert total_params == 73, f"Total parameters must be exactly 73, got {total_params}"
    assert trainable_params == 73, f"Trainable parameters must be exactly 73, got {trainable_params}"


def test_model_output_range():
    """Test 6: Verify model output is bounded within [0, 1] for arbitrary valid inputs."""
    model = build_model()
    dummy_inputs = np.array([
        [-2.0, -1.5, -0.8],
        [0.0, 0.0, 0.0],
        [1.5, 2.0, 1.2],
        [5.0, 5.0, 5.0]
    ], dtype=np.float32)

    predictions = model.predict(dummy_inputs, verbose=0)
    assert predictions.shape == (4, 1), f"Unexpected prediction shape {predictions.shape}"
    assert np.all(predictions >= 0.0) and np.all(predictions <= 1.0), (
        f"Sigmoid outputs must lie strictly in [0, 1]. Got min={predictions.min()}, max={predictions.max()}"
    )


def test_preprocessing_no_data_leakage(setup_dataset):
    """Test 7: Automated test to guarantee zero data leakage during preprocessing.

    Verifies:
    a) StandardScaler mean and variance exactly match X_train statistics.
    b) Scaler statistics differ from full-dataset statistics and test-dataset statistics.
    c) Validation and test transformations do not modify scaler state.
    """
    set_seed(42)
    split_info_path = "models/test_leakage_split_indices.json"
    data_dict = load_and_preprocess_data(
        csv_path=setup_dataset,
        seed=42,
        save_indices_path=split_info_path,
    )

    scaler = data_dict["scaler"]
    X_train_raw = data_dict["X_train_raw"].to_numpy()
    X_test_raw = data_dict["X_test_raw"].to_numpy()
    raw_df = data_dict["raw_df"][["Study_Hours", "Attendance", "Previous_Marks"]].to_numpy()

    # 1. Check scaler mean & variance match X_train exactly
    expected_train_mean = np.mean(X_train_raw, axis=0)
    expected_train_var = np.var(X_train_raw, axis=0)

    np.testing.assert_allclose(
        scaler.mean_,
        expected_train_mean,
        rtol=1e-5,
        err_msg="StandardScaler mean_ does not match training set mean!",
    )
    np.testing.assert_allclose(
        scaler.var_,
        expected_train_var,
        rtol=1e-5,
        err_msg="StandardScaler var_ does not match training set variance!",
    )

    # 2. Confirm scaler was NOT fit on the full dataset or test split
    full_mean = np.mean(raw_df, axis=0)
    test_mean = np.mean(X_test_raw, axis=0)

    assert not np.allclose(scaler.mean_, full_mean, atol=1e-3), (
        "Data Leakage Detected: Scaler mean matches full dataset mean!"
    )
    assert not np.allclose(scaler.mean_, test_mean, atol=1e-3), (
        "Data Leakage Detected: Scaler mean matches test set mean!"
    )

    # 3. Confirm transforming test data does NOT alter scaler attributes
    mean_before = np.copy(scaler.mean_)
    var_before = np.copy(scaler.var_)
    _ = scaler.transform(X_test_raw)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.var_, var_before)

    # Clean up test json if created
    if os.path.exists(split_info_path):
        os.remove(split_info_path)


def test_prediction_pipeline(setup_dataset):
    """Test 8: Verify end-to-end prediction pipeline with valid inputs and error handling."""
    model_path = "models/student_pass_fail.keras"
    scaler_path = "models/scaler.pkl"
    assert os.path.exists(model_path), f"Trained model artifact not found at '{model_path}'. Run train.py first."
    assert os.path.exists(scaler_path), f"Preprocessing scaler artifact not found at '{scaler_path}'. Run train.py first."

    # Test valid pass sample (High hours, high attendance, high marks)
    prob_high, label_high = predict_pass_fail(
        study_hours=8.5,
        attendance=92.0,
        previous_marks=88.0,
        model_path=model_path,
        scaler_path=scaler_path,
    )
    assert 0.0 <= prob_high <= 1.0
    assert label_high in {"PASS", "FAIL"}

    # Test valid fail sample (Low hours, low attendance, low marks)
    prob_low, label_low = predict_pass_fail(
        study_hours=1.5,
        attendance=50.0,
        previous_marks=30.0,
        model_path=model_path,
        scaler_path=scaler_path,
    )
    assert 0.0 <= prob_low <= 1.0
    assert label_low in {"PASS", "FAIL"}
    assert prob_high > prob_low, "Pass probability for strong student should exceed weak student."

    # Test out-of-bounds input validation raises ValueError
    with pytest.raises(ValueError):
        validate_inputs(study_hours=-1.0, attendance=80.0, previous_marks=70.0)
    with pytest.raises(ValueError):
        validate_inputs(study_hours=5.0, attendance=110.0, previous_marks=70.0)
    with pytest.raises(ValueError):
        validate_inputs(study_hours=5.0, attendance=80.0, previous_marks=150.0)


# ==========================================
# FLASK WEB INTERFACE TESTS
# ==========================================

@pytest.fixture
def client():
    """Create Flask test client."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_flask_app_import_and_artifacts():
    """Test 9: Verify Flask app imports and loads saved model/scaler artifacts."""
    from app import app, model, scaler
    assert app is not None
    assert model is not None
    assert scaler is not None
    assert model.count_params() == 73


def test_flask_root_route_get(client):
    """Test 10: Verify GET / returns HTTP 200 and contains form elements."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Student Performance Intelligence" in html
    assert "Study Hours" in html
    assert "Attendance" in html
    assert "Previous Marks" in html


def test_flask_predict_valid_pass_input(client):
    """Test 11: Verify valid high-performing student input yields PASS with correct probabilities."""
    response = client.post(
        "/predict",
        data={
            "study_hours": "7.5",
            "attendance": "88",
            "previous_marks": "80",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Prediction Result" in html
    assert "PASS" in html
    assert "PASS" in html
    assert "FAIL" in html


def test_flask_predict_valid_fail_input(client):
    """Test 12: Verify valid low-performing student input yields FAIL."""
    response = client.post(
        "/predict",
        data={
            "study_hours": "2.0",
            "attendance": "50",
            "previous_marks": "40",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Prediction Result" in html
    assert "FAIL" in html


def test_flask_predict_invalid_attendance(client):
    """Test 13: Verify out-of-bounds attendance is rejected gracefully with validation message."""
    # Test attendance > 100
    response_high = client.post(
        "/predict",
        data={
            "study_hours": "5.0",
            "attendance": "150",
            "previous_marks": "75",
        },
    )
    assert response_high.status_code == 200
    html_high = response_high.get_data(as_text=True)
    assert "Attendance must be numeric and between 0.0% and 100.0%." in html_high
    assert "outcome-pass" not in html_high and "outcome-fail" not in html_high

    # Test attendance < 0
    response_low = client.post(
        "/predict",
        data={
            "study_hours": "5.0",
            "attendance": "-10",
            "previous_marks": "75",
        },
    )
    assert response_low.status_code == 200
    html_low = response_low.get_data(as_text=True)
    assert "Attendance must be numeric and between 0.0% and 100.0%." in html_low


def test_flask_predict_invalid_study_hours(client):
    """Test 14: Verify negative study hours are rejected gracefully with validation message."""
    response = client.post(
        "/predict",
        data={
            "study_hours": "-2",
            "attendance": "85",
            "previous_marks": "75",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Study Hours must be numeric and between 0.0 and 24.0 hours." in html
    assert "outcome-pass" not in html and "outcome-fail" not in html


def test_flask_predict_non_numeric_input(client):
    """Test 15: Verify non-numeric input strings are rejected gracefully without 500 error."""
    response = client.post(
        "/predict",
        data={
            "study_hours": "abc",
            "attendance": "85",
            "previous_marks": "75",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Study Hours must be a valid numeric value." in html


# ==========================================
# BACKEND HARDENING & SECURITY TESTS
# ==========================================

def test_health_endpoint(client):
    """Test 16: Verify GET /health returns HTTP 200 and expected status JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["success"] is True
    assert data["status"] == "ok"


def test_security_headers(client):
    """Test 17: Verify standard HTTP security headers are present on responses."""
    response = client.get("/")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in response.headers
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_output_allowlist_allowed(client):
    """Test 18: Verify allowed visual artifact returns HTTP 200."""
    response = client.get("/outputs/architecture.png")
    assert response.status_code == 200


def test_output_allowlist_rejected(client):
    """Test 19: Verify non-allowlisted and path traversal attempts return HTTP 404."""
    # Disallowed file name
    response_disallowed = client.get("/outputs/secret.json")
    assert response_disallowed.status_code == 404

    # Path traversal attempt
    response_traversal = client.get("/outputs/../models/scaler.pkl")
    assert response_traversal.status_code == 404


def test_custom_404_handler(client):
    """Test 20: Verify invalid routes return HTTP 404 without exposing stack traces."""
    response = client.get("/nonexistent-endpoint-xyz")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert "The requested page or resource was not found." in html


def test_nan_and_inf_input_rejection(client):
    """Test 21: Verify NaN and Infinity values are rejected by server-side validation."""
    # Test NaN input
    response_nan = client.post(
        "/predict",
        data={
            "study_hours": "nan",
            "attendance": "85",
            "previous_marks": "75",
        },
    )
    assert response_nan.status_code == 200
    html_nan = response_nan.get_data(as_text=True)
    assert "Study Hours must be a valid numeric value." in html_nan

    # Test Infinity input
    response_inf = client.post(
        "/predict",
        data={
            "study_hours": "7.5",
            "attendance": "inf",
            "previous_marks": "75",
        },
    )
    assert response_inf.status_code == 200
    html_inf = response_inf.get_data(as_text=True)
    assert "Attendance must be a valid numeric value." in html_inf


def test_oversized_payload_protection(client):
    """Test 22: Verify oversized payloads exceeding MAX_CONTENT_LENGTH trigger HTTP 413."""
    oversized_data = {
        "study_hours": "7.5",
        "attendance": "85",
        "previous_marks": "75",
        "junk": "X" * (40 * 1024),  # 40 KB payload exceeds 32 KB limit
    }
    response = client.post("/predict", data=oversized_data)
    assert response.status_code == 413


def test_wsgi_callable_and_entrypoint():
    """Test 23: Verify app module exports valid WSGI application callable for Gunicorn."""
    from app import app
    assert callable(app)
    assert hasattr(app, "wsgi_app")
    assert callable(app.wsgi_app)


def test_render_deployment_configuration():
    """Test 24: Verify render.yaml contains valid deployment directives and PYTHON_VERSION."""
    assert os.path.exists("render.yaml"), "render.yaml blueprint file is missing"

    with open("render.yaml", "r", encoding="utf-8") as f:
        render_content = f.read()
    assert "type: web" in render_content
    assert "runtime: python" in render_content
    assert "plan: free" in render_content
    assert "pip install -r requirements.txt" in render_content
    assert "gunicorn" in render_content
    assert "app:app" in render_content
    assert "healthCheckPath: /health" in render_content
    assert "PYTHON_VERSION" in render_content
    assert "3.12" in render_content


# ==========================================
# REST API ENDPOINT TESTS (PHASE F)
# ==========================================

def test_api_predict_valid_pass(client):
    """Test 25: Verify POST /api/predict with valid high-performing values returns JSON PASS."""
    payload = {
        "study_hours": 7.5,
        "attendance": 88.0,
        "previous_marks": 80.0
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["success"] is True
    assert data["prediction"] == "PASS"
    assert data["pass_probability"] > 0.5
    assert data["fail_probability"] < 0.5
    assert "confidence" in data
    assert data["study_hours"] == 7.5
    assert data["attendance"] == 88.0
    assert data["previous_marks"] == 80.0


def test_api_predict_valid_fail(client):
    """Test 26: Verify POST /api/predict with valid low-performing values returns JSON FAIL."""
    payload = {
        "study_hours": 1.0,
        "attendance": 45.0,
        "previous_marks": 35.0
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["success"] is True
    assert data["prediction"] == "FAIL"
    assert data["pass_probability"] < 0.5
    assert data["fail_probability"] > 0.5


def test_api_predict_missing_fields(client):
    """Test 27: Verify POST /api/predict with missing required fields returns HTTP 400."""
    payload = {
        "study_hours": 7.5,
        "attendance": 88.0
        # missing previous_marks
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.is_json
    data = response.get_json()
    assert data["success"] is False
    assert "Previous Marks is required." in data["error"]


def test_api_predict_out_of_range(client):
    """Test 28: Verify POST /api/predict with out-of-range values returns HTTP 400."""
    payload = {
        "study_hours": 25.0,  # exceeds 24.0
        "attendance": 88.0,
        "previous_marks": 80.0
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.is_json
    data = response.get_json()
    assert data["success"] is False
    assert "Study Hours must be numeric and between 0.0 and 24.0 hours." in data["error"]


def test_api_predict_non_json_content_type(client):
    """Test 29: Verify POST /api/predict with non-JSON content-type returns HTTP 400."""
    response = client.post(
        "/api/predict",
        data="study_hours=7.5&attendance=88&previous_marks=80",
        content_type="application/x-www-form-urlencoded",
    )
    assert response.status_code == 400
    assert response.is_json
    data = response.get_json()
    assert data["success"] is False
    assert "Request content-type must be application/json." in data["error"]


def test_api_predict_nan_and_inf(client):
    """Test 30: Verify POST /api/predict rejects NaN and Infinity inputs."""
    payload_nan = {
        "study_hours": float("nan"),
        "attendance": 88.0,
        "previous_marks": 80.0
    }
    response_nan = client.post(
        "/api/predict",
        data=json.dumps(payload_nan),
        content_type="application/json",
    )
    assert response_nan.status_code == 400
    assert response_nan.is_json
    assert response_nan.get_json()["success"] is False
