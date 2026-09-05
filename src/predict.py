"""Inference script for Student Pass/Fail Prediction FNN.

Accepts student features (Study Hours, Attendance, Previous Exam Marks),
validates inputs, applies the saved training StandardScaler, and outputs
the predicted pass probability and final binary classification.
"""

import argparse
import os
import sys

# Ensure root directory is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import tensorflow as tf


def validate_inputs(study_hours: float, attendance: float, previous_marks: float):
    """Validate numerical bounds of student feature inputs.
    
    Args:
        study_hours (float): Daily study hours (expected: 0.0 to 24.0).
        attendance (float): Attendance percentage (expected: 0.0 to 100.0).
        previous_marks (float): Previous exam marks (expected: 0.0 to 100.0).
        
    Raises:
        ValueError: If any input value is out of logical domain bounds.
    """
    if not (0.0 <= study_hours <= 24.0):
        raise ValueError(
            f"Invalid Study Hours: {study_hours}. Expected value between 0.0 and 24.0 hours."
        )
    if not (0.0 <= attendance <= 100.0):
        raise ValueError(
            f"Invalid Attendance: {attendance}%. Expected percentage between 0.0% and 100.0%."
        )
    if not (0.0 <= previous_marks <= 100.0):
        raise ValueError(
            f"Invalid Previous Marks: {previous_marks}. Expected marks between 0.0 and 100.0."
        )


def predict_pass_fail(
    study_hours: float,
    attendance: float,
    previous_marks: float,
    model_path: str = "models/student_pass_fail.keras",
    scaler_path: str = "models/scaler.pkl",
):
    """Predict pass/fail probability for a single student profile.
    
    Args:
        study_hours (float): Hours spent studying per day.
        attendance (float): Class attendance percentage.
        previous_marks (float): Marks from previous exams.
        model_path (str): Path to saved Keras model artifact.
        scaler_path (str): Path to saved StandardScaler artifact.
        
    Returns:
        tuple: (pass_probability: float, prediction_label: str)
    """
    validate_inputs(study_hours, attendance, previous_marks)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. Run 'python src/train.py' first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Preprocessing scaler artifact not found at '{scaler_path}'. Run 'python src/train.py' first."
        )

    # 1. Load fitted scaler and transform feature vector
    scaler = joblib.load(scaler_path)
    raw_features = np.array([[study_hours, attendance, previous_marks]], dtype=np.float32)
    scaled_features = scaler.transform(raw_features)

    # 2. Load model and compute prediction
    model = tf.keras.models.load_model(model_path)
    prob = float(model.predict(scaled_features, verbose=0)[0][0])

    label = "PASS" if prob >= 0.5 else "FAIL"

    return prob, label


def main():
    parser = argparse.ArgumentParser(
        description="Predict Student Pass/Fail using trained Feedforward Neural Network."
    )
    parser.add_argument(
        "--study-hours",
        type=float,
        help="Daily study hours (e.g. 7.5)",
    )
    parser.add_argument(
        "--attendance",
        type=float,
        help="Attendance percentage from 0 to 100 (e.g. 88.0)",
    )
    parser.add_argument(
        "--previous-marks",
        type=float,
        help="Previous exam marks out of 100 (e.g. 80.0)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/student_pass_fail.keras",
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/scaler.pkl",
        help="Path to fitted scaler .pkl",
    )

    args = parser.parse_args()

    # Interactive input fallback if CLI arguments omitted
    if args.study_hours is None or args.attendance is None or args.previous_marks is None:
        print("\nStudent Pass/Fail Interactive Predictor")
        print("-" * 45)
        try:
            sh = float(input("Enter Study Hours (0 - 24): ").strip())
            att = float(input("Enter Attendance % (0 - 100): ").strip())
            pm = float(input("Enter Previous Exam Marks (0 - 100): ").strip())
        except ValueError as e:
            print(f"Error: Invalid numeric input. {e}")
            sys.exit(1)
    else:
        sh = args.study_hours
        att = args.attendance
        pm = args.previous_marks

    try:
        prob, label = predict_pass_fail(
            study_hours=sh,
            attendance=att,
            previous_marks=pm,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
        )

        print("\n" + "=" * 45)
        print("INFERENCE RESULT")
        print("=" * 45)
        print(f"  Input Features:")
        print(f"    * Study Hours   : {sh:.2f} hrs/day")
        print(f"    * Attendance    : {att:.2f}%")
        print(f"    * Previous Marks: {pm:.2f} / 100")
        print("-" * 45)
        print(f"  Pass Probability: {prob * 100:.2f}%")
        print(f"  Prediction      : {label}")
        print("=" * 45 + "\n")

    except Exception as err:
        print(f"\n[ERROR] Prediction Error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
