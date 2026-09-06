"""Flask Web Application for Student Performance Prediction & Risk Analytics.

Serves a production-hardened web interface, What-If scenario inference, and
a RESTful prediction API using the pre-trained 73-parameter Feedforward
Neural Network (FNN) and fitted StandardScaler.
"""

import logging
import math
import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory

# Configure structured application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("student_predictor")

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = Flask(__name__)

# Request payload size limit (32 KB is ample for numeric form/JSON submissions)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

# Model and scaler artifact paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_pass_fail.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Allowed public visualization artifacts
ALLOWED_OUTPUT_FILES = {
    "architecture.png",
    "training_history.png",
    "confusion_matrix.png",
}

# Global references loaded once at application startup
model = None
scaler = None


def load_artifacts():
    """Load the trained model and fitted StandardScaler once on startup."""
    global model, scaler
    if model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model artifact not found at {MODEL_PATH}")
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run 'python src/train.py' first."
            )
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Successfully loaded pre-trained model from {MODEL_PATH} ({model.count_params()} parameters)")

    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            logger.error(f"Scaler artifact not found at {SCALER_PATH}")
            raise FileNotFoundError(
                f"Fitted scaler not found at {SCALER_PATH}. Run 'python src/train.py' first."
            )
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Successfully loaded fitted StandardScaler from {SCALER_PATH}")


# Initialize artifacts at import/startup time
load_artifacts()


@app.after_request
def apply_security_headers(response):
    """Apply standard HTTP security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self';"
    )
    return response


def validate_input_data(study_hours_raw, attendance_raw, previous_marks_raw):
    """Validate server-side student input values with strict numeric and boundary checks.

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
        if math.isnan(study_hours) or math.isinf(study_hours):
            return None, "Study Hours must be a valid numeric value."
    except (ValueError, TypeError):
        return None, "Study Hours must be a valid numeric value."

    try:
        attendance = float(attendance_raw)
        if math.isnan(attendance) or math.isinf(attendance):
            return None, "Attendance must be a valid numeric value."
    except (ValueError, TypeError):
        return None, "Attendance must be a valid numeric value."

    try:
        previous_marks = float(previous_marks_raw)
        if math.isnan(previous_marks) or math.isinf(previous_marks):
            return None, "Previous Marks must be a valid numeric value."
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


def run_inference(sh: float, att: float, pm: float) -> dict:
    """Execute single-sample inference through the exact locked preprocessing and model pipeline.

    Args:
        sh: Daily Study Hours in [0.0, 24.0]
        att: Classroom Attendance Percentage in [0.0, 100.0]
        pm: Previous Marks in [0.0, 100.0]

    Returns:
        dict: Standardized inference result payload.
    """
    raw_features = np.array([[sh, att, pm]], dtype=np.float32)
    scaled_features = scaler.transform(raw_features)

    pass_prob = float(model.predict(scaled_features, verbose=0)[0][0])
    fail_prob = max(0.0, 1.0 - pass_prob)
    prediction = "PASS" if pass_prob >= 0.5 else "FAIL"
    confidence_val = pass_prob if prediction == "PASS" else fail_prob
    confidence_pct = f"{confidence_val * 100:.2f}%"

    return {
        "prediction": prediction,
        "confidence": confidence_pct,
        "confidence_pct": confidence_pct,
        "pass_probability": round(pass_prob, 6),
        "fail_probability": round(fail_prob, 6),
        "pass_prob_pct": f"{pass_prob * 100:.2f}%",
        "fail_prob_pct": f"{fail_prob * 100:.2f}%",
        "raw_pass_prob": pass_prob,
        "raw_fail_prob": fail_prob,
        "study_hours": round(sh, 2),
        "attendance": round(att, 2),
        "previous_marks": round(pm, 2),
        "interpretation": (
            "The model predicts PASS for the supplied input values."
            if prediction == "PASS"
            else "The model predicts FAIL for the supplied input values."
        ),
    }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming service and model readiness."""
    if model is None or scaler is None:
        return jsonify({"success": False, "status": "unhealthy", "message": "Model or scaler artifact not loaded"}), 503
    return jsonify({"success": True, "status": "ok"}), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic single-sample Pass/Fail prediction.

    Expected JSON body:
    {
        "study_hours": 7.5,
        "attendance": 88.0,
        "previous_marks": 80.0
    }
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request content-type must be application/json."
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Malformed JSON payload provided."
        }), 400

    study_hours_raw = data.get("study_hours")
    attendance_raw = data.get("attendance")
    previous_marks_raw = data.get("previous_marks")

    validated_vals, error_msg = validate_input_data(
        study_hours_raw, attendance_raw, previous_marks_raw
    )

    if error_msg is not None:
        return jsonify({
            "success": False,
            "error": error_msg
        }), 400

    sh = validated_vals["study_hours"]
    att = validated_vals["attendance"]
    pm = validated_vals["previous_marks"]

    try:
        result = run_inference(sh, att, pm)
        response_payload = {
            "success": True,
            "prediction": result["prediction"],
            "pass_probability": result["pass_probability"],
            "fail_probability": result["fail_probability"],
            "confidence": result["confidence"],
            "study_hours": result["study_hours"],
            "attendance": result["attendance"],
            "previous_marks": result["previous_marks"],
            "interpretation": result["interpretation"],
        }
        return jsonify(response_payload), 200
    except Exception as exc:
        logger.error(f"API inference failure: {exc}", exc_info=False)
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred during inference."
        }), 500


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_outputs(filename):
    """Serve visual outputs strictly constrained to allowed image artifacts."""
    if filename not in ALLOWED_OUTPUT_FILES:
        return render_template(
            "index.html",
            study_hours="",
            attendance="",
            previous_marks="",
            error="The requested visual artifact was not found.",
            result=None,
        ), 404
    return send_from_directory(os.path.join(BASE_DIR, "outputs"), filename)


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle home dashboard rendering and prediction inference requests."""
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
        logger.warning(f"Prediction input validation rejected: {error_msg}")
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
        result = run_inference(sh, att, pm)
        # Format strings for HTML presentation
        result["study_hours"] = f"{sh:.2f}"
        result["attendance"] = f"{att:.2f}"
        result["previous_marks"] = f"{pm:.2f}"

        return render_template(
            "index.html",
            study_hours=study_hours_raw,
            attendance=attendance_raw,
            previous_marks=previous_marks_raw,
            error=None,
            result=result,
        )

    except Exception as exc:
        logger.error(f"Unexpected inference failure: {exc}", exc_info=False)
        return render_template(
            "index.html",
            study_hours=study_hours_raw,
            attendance=attendance_raw,
            previous_marks=previous_marks_raw,
            error="An unexpected error occurred during prediction. Please verify your inputs.",
            result=None,
        )


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request error gracefully."""
    if request.path.startswith("/api/") or request.path == "/health" or request.is_json:
        return jsonify({"success": False, "error": "Bad request. Please check the submitted data."}), 400
    return render_template(
        "index.html",
        study_hours="",
        attendance="",
        previous_marks="",
        error="Bad request. Please check the submitted data.",
        result=None,
    ), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found error gracefully."""
    if request.path.startswith("/api/") or request.path.startswith("/health"):
        return jsonify({"success": False, "error": "The requested API endpoint was not found."}), 404
    return render_template(
        "index.html",
        study_hours="",
        attendance="",
        previous_marks="",
        error="The requested page or resource was not found.",
        result=None,
    ), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 Payload Too Large error gracefully."""
    if request.path.startswith("/api/") or request.path.startswith("/health") or request.is_json:
        return jsonify({"success": False, "error": "Submitted payload is too large. Maximum allowed size is 32 KB."}), 413
    return render_template(
        "index.html",
        study_hours="",
        attendance="",
        previous_marks="",
        error="Submitted payload is too large. Please provide standard input values.",
        result=None,
    ), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server Error without exposing stack trace."""
    logger.error(f"Internal Server Error 500 on {request.path}")
    if request.path.startswith("/api/") or request.path == "/health" or request.is_json:
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500
    return render_template(
        "index.html",
        study_hours="",
        attendance="",
        previous_marks="",
        error="An unexpected server error occurred. Please try again later.",
        result=None,
    ), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
