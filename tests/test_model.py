"""Automated unit and integration tests for Student Pass/Fail Prediction FNN.

Verifies:
1. Required dataset columns exist.
2. Target column contains both binary classes (0 and 1).
3. Model input shape is compatible with 3 features.
4. Model output shape is (None, 1).
5. Model contains exactly 73 trainable parameters.
6. Model output values are valid probabilities strictly in [0, 1].
7. Preprocessing strictly prevents data leakage (scaler fitted exclusively on training set).
8. End-to-end prediction pipeline functions reliably with valid samples.
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
