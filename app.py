"""Flask Web Application for Student Pass/Fail Prediction.

Serves a clean, professional web interface for inference using the pre-trained
73-parameter Feedforward Neural Network (FNN) and fitted StandardScaler.
"""

import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = Flask(__name__)

# Model and scaler artifact paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_pass_fail.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler once at application startup
model = None
scaler = None


def load_artifacts():
    """Load the trained model and fitted StandardScaler once on startup."""
    global model, scaler
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run 'python src/train.py' first."
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Fitted scaler not found at {SCALER_PATH}. Run 'python src/train.py' first."
            )
        scaler = joblib.load(SCALER_PATH)


# Initialize artifacts at import/startup time
load_artifacts()


def validate_input_data(study_hours_raw, attendance_raw, previous_marks_raw):
    """Validate server-side student input values.
    
    Returns:
        tuple: (validated_values_dict_or_None, error_message_or_None)
    """
    if study_hours_raw is None or not str(study_hours_raw).strip():
        return None, "Study Hours is required."
    if attendance_raw is None or not str(attendance_raw).strip():
        return None, "Attendance is required."
    if previous_marks_raw is None or not str(previous_marks_raw).strip():
        return None, "Previous Marks is required."

    try:
        study_hours = float(study_hours_raw)
    except (ValueError, TypeError):
        return None, "Study Hours must be a valid numeric value."

    try:
        attendance = float(attendance_raw)
    except (ValueError, TypeError):
        return None, "Attendance must be a valid numeric value."

    try:
        previous_marks = float(previous_marks_raw)
    except (ValueError, TypeError):
        return None, "Previous Marks must be a valid numeric value."

    if study_hours < 0.0 or study_hours > 24.0:
        return None, "Study Hours must be numeric and between 0.0 and 24.0 hours."

    if attendance < 0.0 or attendance > 100.0:
        return None, "Attendance must be numeric and between 0.0% and 100.0%."

    if previous_marks < 0.0 or previous_marks > 100.0:
        return None, "Previous Marks must be numeric and between 0.0 and 100.0."

    return {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
    }, None


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle home interface and prediction submissions."""
    if request.method == "GET":
        return render_template(
            "index.html",
            study_hours="",
            attendance="",
            previous_marks="",
            error=None,
            result=None,
        )

    # Handle POST request
    study_hours_raw = request.form.get("study_hours", "")
    attendance_raw = request.form.get("attendance", "")
    previous_marks_raw = request.form.get("previous_marks", "")

    validated_vals, error_msg = validate_input_data(
        study_hours_raw, attendance_raw, previous_marks_raw
    )

    if error_msg is not None:
        return render_template(
            "index.html",
            study_hours=study_hours_raw,
            attendance=attendance_raw,
            previous_marks=previous_marks_raw,
            error=error_msg,
            result=None,
        )

    sh = validated_vals["study_hours"]
    att = validated_vals["attendance"]
    pm = validated_vals["previous_marks"]

    try:
        # Preprocess features: Study_Hours, Attendance, Previous_Marks
        raw_features = np.array([[sh, att, pm]], dtype=np.float32)
        scaled_features = scaler.transform(raw_features)

        # Model inference
        pass_prob = float(model.predict(scaled_features, verbose=0)[0][0])
        fail_prob = max(0.0, 1.0 - pass_prob)
        prediction = "PASS" if pass_prob >= 0.5 else "FAIL"

        result = {
            "prediction": prediction,
            "pass_prob_pct": f"{pass_prob * 100:.2f}%",
            "fail_prob_pct": f"{fail_prob * 100:.2f}%",
            "raw_pass_prob": pass_prob,
            "study_hours": f"{sh:.2f}",
            "attendance": f"{att:.2f}",
            "previous_marks": f"{pm:.2f}",
        }

        return render_template(
            "index.html",
            study_hours=study_hours_raw,
            attendance=attendance_raw,
            previous_marks=previous_marks_raw,
            error=None,
            result=result,
        )

    except Exception:
        # Do not expose internal traceback to the user
        return render_template(
            "index.html",
            study_hours=study_hours_raw,
            attendance=attendance_raw,
            previous_marks=previous_marks_raw,
            error="An unexpected error occurred during prediction. Please verify your inputs.",
            result=None,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
