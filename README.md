# Student Performance Prediction & Academic Risk Analytics

An end-to-end machine learning system and academic risk analytics platform that predicts student course outcomes (Pass/Fail) from study engagement and historical performance indicators. Built with a 73-parameter Feedforward Neural Network (FNN) in TensorFlow/Keras, fitted with leakage-free StandardScaler preprocessing, and served via a hardened Flask web application, an interactive What-If scenario simulator, and a programmatic RESTful inference API.

Developed strictly to engineering and academic standards with 100% deterministic reproducibility, verified zero data leakage, a 30-test automated test suite, and cloud deployment readiness on Render.

---

## 📌 Overview

Early detection of academic difficulty empowers educators and institutions to deliver timely, targeted pedagogical interventions. This platform models the non-linear relationship between student engagement (Study Hours and Classroom Attendance) and academic history (Previous Marks) to produce calibrated binary classification probabilities.

The platform provides three primary interfaces:
1. **Academic Web Dashboard**: A modern, responsive, accessible interface with synchronized numeric/range inputs, real-time quality validation, and visual outcome telemetry.
2. **What-If Scenario Simulation**: An interactive comparison workspace evaluating alternative input scenarios against current predictions without retraining the model.
3. **RESTful Inference API**: A low-latency JSON endpoint (`POST /api/predict`) and service health monitor (`GET /health`) for programmatic integrations.

---

## 🌐 Live Demo

> **Deployment Target**: Render Web Service
>
> **Live Application URL**: `https://student-pass-fail-fnn.onrender.com` *(Deployment placeholder)*

---

## ✨ Key Features

- **Strict Leakage-Free Preprocessing**: `StandardScaler` is fitted **strictly on the 70% training split** and persisted to `models/scaler.pkl` to transform validation, test, and live inference vectors without data leakage.
- **Deterministic 70/15/15 Stratified Partition**: Fixed random state (`seed=42`) with serialized partition indices (`models/split_indices.json`) guaranteeing exact evaluation reproducibility.
- **Compact Neural Architecture**: Exactly **73 trainable parameters** ($3 \rightarrow 8 \rightarrow 4 \rightarrow 1$) with zero unnecessary model bloat.
- **Interactive What-If Analysis**: Compare baseline and alternative student performance scenarios side-by-side using the identical saved model, scaler, and 0.50 decision threshold.
- **RESTful Prediction API**: High-performance `POST /api/predict` endpoint returning calibrated class probabilities, prediction verdicts, and honest interpretations.
- **Production Backend Hardening**: Server-side validation with NaN/Infinity rejection, 32 KB payload limit, standard HTTP security headers (`nosniff`, `SAMEORIGIN`, CSP), safe artifact allowlists, structured logging, and a dedicated `/health` check.
- **Cloud & WSGI Ready**: Production-configured with Gunicorn WSGI, single-worker multi-threading to prevent free-tier memory exhaustion, and a declarative `render.yaml` blueprint with explicit Python version pinning (`3.12.8`).
- **Comprehensive Automated Test Suite**: 30 pytest tests covering data hygiene, parameter count verification, leakage prevention, Flask routes, REST API contracts, boundary validation, WSGI callable, and Render blueprint schema.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT INTERFACES                              │
│                                                                             │
│   Web Dashboard (HTML5/CSS3/JS)       │    External Services / REST Clients │
│   [Sliders • Status • What-If Panel]   │    [cURL, Python Requests, etc.]    │
└───────────────────────┬───────────────────────────────────────┬─────────────┘
                        │ Form POST / GET                       │ JSON POST / GET
                        ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLASK WEB APPLICATION                              │
│                                                                             │
│  • Security Headers (nosniff, SAMEORIGIN, CSP)                              │
│  • Request Size Guard (MAX_CONTENT_LENGTH = 32 KB)                          │
│  • Strict Server-Side Validation (Range, Types, NaN, Infinity)              │
│  • Error Handlers (400 Bad Request, 404 Not Found, 413, 500)                │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ Validated Features [sh, att, pm]
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHARED INFERENCE PIPELINE                            │
│                                                                             │
│  1. Feature Standardization: models/scaler.pkl                              │
│     (Fitted strictly on 70% training partition; mean_ & var_ locked)        │
│                                                                             │
│  2. Neural Network Forward Pass: models/student_pass_fail.keras             │
│     Input(3) ──> Dense(8, ReLU) ──> Dense(4, ReLU) ──> Dense(1, Sigmoid)    │
│                                                                             │
│  3. Calibrated Output Probability: P(Pass) ∈ [0.0, 1.0]                     │
│                                                                             │
│  4. Decision Boundary: Threshold = 0.50 ──> PASS (≥ 0.5) or FAIL (< 0.5)    │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
            HTML Dashboard Render              JSON API Response
```

---

## 🧠 Machine Learning Model

### Model Architecture

| Layer | Type | Neurons | Activation | Weights | Biases | Total Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Input** | Feature Vector | 3 | — | — | — | — |
| **Hidden 1** | Dense | 8 | ReLU | $3 \times 8 = 24$ | 8 | **32** |
| **Hidden 2** | Dense | 4 | ReLU | $8 \times 4 = 32$ | 4 | **36** |
| **Output** | Dense | 1 | Sigmoid | $4 \times 1 = 4$ | 1 | **5** |
| **Total** | | | | **60** | **13** | **73** |

$$\text{Total Trainable Parameters} = 32 + 36 + 5 = \mathbf{73}$$

### Training Configuration
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam (Learning Rate = 0.01)
- **Batch Size**: 16
- **Regularization / Early Stopping**: `patience=20`, `restore_best_weights=True` (Restored optimal weights from Epoch 6)
- **Decision Threshold**: 0.50

---

## 📊 Dataset

> [!NOTE]
> The dataset (`data/student_data.csv`) is **synthetically generated** for academic demonstration and learning; it does not represent actual institutional records.

- **Total Samples**: 300 student records
- **Features**:
  1. `Study_Hours`: Average daily self-study hours ($0.0 – 24.0$ hrs)
  2. `Attendance`: Classroom attendance percentage ($0.0 – 100.0\%$)
  3. `Previous_Marks`: Score in previous evaluations ($0.0 – 100.0$)
- **Target Variable**: `Result` ($0 = \text{Fail}, 1 = \text{Pass}$)
- **Class Balance**: 136 Fail (45.3%) • 164 Pass (54.7%)

### Stratified 70 / 15 / 15 Partition
- **Training Set (70%)**: 210 samples (Fail: 95, Pass: 115)
- **Validation Set (15%)**: 45 samples (Fail: 21, Pass: 24)
- **Held-out Test Set (15%)**: 45 samples (Fail: 20, Pass: 25)

---

## ⚙️ Data Preprocessing & Zero Data Leakage

1. **Partition Precedes Preprocessing**: Dataset is split into train (70%), validation (15%), and test (15%) partitions first using stratified sampling (`random_state=42`).
2. **Fit on Training Set Only**: `StandardScaler.fit()` is computed **strictly on the 210 training samples**.
3. **Downstream Standardization**: Validation, test, and live prediction feature vectors are standardized using the training-fitted mean and variance without updating scaler state.
4. **Artifact Serialization**: Scaler parameters are persisted to `models/scaler.pkl` and loaded once at startup.

---

## 📈 Evaluation & Held-Out Test Metrics

Evaluated strictly on the **45 held-out synthetic test samples** (`src/evaluate.py`):

| Metric | Held-Out Test Score | Formula |
| :--- | :---: | :--- |
| **Accuracy** | **93.33%** | $\frac{24 + 18}{45}$ |
| **Precision** | **92.31%** | $\frac{24}{24 + 2}$ |
| **Recall** | **96.00%** | $\frac{24}{24 + 1}$ |
| **F1-Score** | **94.12%** | $2 \times \frac{0.9231 \times 0.9600}{0.9231 + 0.9600}$ |

### Confusion Matrix (45 Test Samples)

```
                     Predicted FAIL (0)     Predicted PASS (1)
 Actual FAIL (0)             18 (TN)                 2 (FP)
 Actual PASS (1)              1 (FN)                24 (TP)
```

---

## 🖥️ Web Dashboard & Interactive Scenario Analytics (Phase F.1)

The web application provides an interactive academic performance prediction and What-If scenario analysis workspace:
- **Synchronized Controls**: Bidirectional numeric inputs and smooth range sliders with live bound validation.
- **Dynamic Student Input Profile**: Real-time visual horizontal meters reflecting entered input values relative to valid domain bounds (Study Hours $X/24$, Attendance $X/100$, Previous Marks $X/100$) with descriptive profile categorization.
- **Dual-Inference What-If Scenario Simulation**: Compares Baseline Student against Simulated Scenario Student by querying the live `/api/predict` endpoint in parallel.
- **Comparative Probability Distribution**: Animated comparative progress meters showing PASS/FAIL distribution shifts between baseline and hypothetical scenarios.
- **Scenario Impact Summary**: Exact delta calculations reporting PASS probability shifts in percentage points (`+17.16 percentage points`), outcome transitions (`FAIL → PASS`), and feature deltas.
- **Deterministic Non-Causal Interpretation**: Objective explanations of model behavior without unfounded causal claims.
- **Model Architecture & Diagnostics**: Visual neural network topology flow diagram and held-out test set confusion matrix benchmarks.

---

## 🔌 REST API Documentation

The application exposes a lightweight, production-hardened REST API for programmatic predictions and system monitoring.

### 1. Health Check Endpoint
Confirm service and model artifact readiness.

- **Endpoint**: `GET /health`
- **Response (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "success": true
  }
  ```

### 2. Predict Endpoint
Execute single-sample student Pass/Fail inference.

- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "study_hours": 7.5,
    "attendance": 88.0,
    "previous_marks": 80.0
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "prediction": "PASS",
    "pass_probability": 0.999912,
    "fail_probability": 0.000088,
    "confidence": "99.99%",
    "study_hours": 7.5,
    "attendance": 88.0,
    "previous_marks": 80.0,
    "interpretation": "The model predicts PASS for the supplied input values."
  }
  ```

### 3. API Error Responses

| Status Code | Reason | Example Response |
| :---: | :--- | :--- |
| **`400 Bad Request`** | Missing fields, non-numeric values, or out-of-range inputs | `{"success": false, "error": "Study Hours must be numeric and between 0.0 and 24.0 hours."}` |
| **`404 Not Found`** | Unknown API endpoint | `{"success": false, "error": "The requested API endpoint was not found."}` |
| **`413 Payload Too Large`** | Request body exceeds 32 KB limit | `{"success": false, "error": "Submitted payload is too large. Maximum allowed size is 32 KB."}` |
| **`500 Server Error`** | Internal exception during inference | `{"success": false, "error": "An unexpected server error occurred."}` |

---

## 💻 API Usage Examples

### cURL
```bash
curl -X POST http://127.0.0.1:5000/api/predict   -H "Content-Type: application/json"   -d '{"study_hours": 7.5, "attendance": 88.0, "previous_marks": 80.0}'
```

### Python
```python
import requests

url = "http://127.0.0.1:5000/api/predict"
payload = {
    "study_hours": 7.5,
    "attendance": 88.0,
    "previous_marks": 80.0
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Prediction: {data['prediction']} (Confidence: {data['confidence']})")
print(f"Pass Probability: {data['pass_probability']:.4f}")
```

---

## 📁 Project Structure

```
student-pass-fail-fnn/
├── .gitignore                      # Git ignore rules for Python, caches, and virtualenvs
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive resume-grade documentation
├── render.yaml                     # Render Infrastructure-as-Code Blueprint specification
├── requirements.txt                # Pinned dependencies (TensorFlow, scikit-learn, Flask, Gunicorn)
├── app.py                          # Hardened Flask web application with REST API & WSGI
├── data/
│   ├── generate_dataset.py         # Realistic student dataset generator
│   └── student_data.csv            # 300 student records CSV
├── models/
│   ├── scaler.pkl                  # StandardScaler fitted strictly on training partition
│   ├── split_indices.json          # Deterministic train/val/test index splits
│   └── student_pass_fail.keras     # 73-parameter TensorFlow/Keras FNN model artifact
├── outputs/
│   ├── architecture.png            # Visual neural network architecture diagram
│   ├── confusion_matrix.png        # Annotated confusion matrix heatmap on test set
│   └── training_history.png        # Loss & Accuracy curves across training epochs
├── src/
│   ├── __init__.py                 # Python package initializer
│   ├── evaluate.py                 # Standalone test evaluation script
│   ├── predict.py                  # CLI & interactive inference script
│   ├── train.py                    # Training pipeline with early stopping & leakage prevention
│   └── utils.py                    # Core model builder, preprocessing, & visualization routines
├── static/
│   ├── script.js                   # Interactive client-side slider sync, validation, & What-If handler
│   └── style.css                   # Custom styling for Flask web dashboard & What-If panel
├── templates/
│   └── index.html                  # Semantic HTML template for browser prediction UI & Model Card
└── tests/
    └── test_model.py               # 30 automated unit, integration, web, and API tests
```

---

## 🧪 Testing & Validation

Execute the complete automated test suite:

```bash
pytest -v
```

The test suite in `tests/test_model.py` runs **35 automated tests**:
1. `test_dataset_columns`: Asserts required columns exist without missing values.
2. `test_target_classes`: Validates binary classes `{0, 1}` and balanced ratio.
3. `test_model_input_shape`: Validates model input shape is `(None, 3)`.
4. `test_model_output_shape`: Validates model output shape is `(None, 1)`.
5. `test_trainable_parameters_count`: Confirms total and trainable parameter counts are **strictly 73**.
6. `test_model_output_range`: Asserts model output is bounded in $[0, 1]$.
7. `test_preprocessing_no_data_leakage`: Verifies `StandardScaler` parameters equal training partition statistics and differ from test/full partitions.
8. `test_prediction_pipeline`: Tests CLI inference function and boundary error handling.
9. `test_flask_app_import_and_artifacts`: Asserts Flask app imports and pre-loads model/scaler.
10. `test_flask_root_route_get`: Asserts GET `/` returns HTTP 200 with form elements.
11. `test_flask_predict_valid_pass_input`: Validates high-scoring input yields `PASS` on web route.
12. `test_flask_predict_valid_fail_input`: Validates low-scoring input yields `FAIL` on web route.
13. `test_flask_predict_invalid_attendance`: Validates out-of-bounds attendance is rejected gracefully.
14. `test_flask_predict_invalid_study_hours`: Validates negative study hours are rejected gracefully.
15. `test_flask_predict_non_numeric_input`: Validates invalid non-numeric inputs do not crash the app.
16. `test_health_endpoint`: Asserts `GET /health` returns HTTP 200 and `{"success": true, "status": "ok"}`.
17. `test_security_headers`: Asserts `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and CSP headers are present.
18. `test_output_allowlist_allowed`: Asserts allowlisted visual artifacts return HTTP 200.
19. `test_output_allowlist_rejected`: Asserts non-allowlisted and path traversal attempts return HTTP 404.
20. `test_custom_404_handler`: Asserts nonexistent routes return HTTP 404 with custom error template.
21. `test_nan_and_inf_input_rejection`: Asserts `NaN` and `Infinity` inputs are rejected by server-side validation.
22. `test_oversized_payload_protection`: Asserts payloads exceeding `MAX_CONTENT_LENGTH` trigger HTTP 413.
23. `test_wsgi_callable_and_entrypoint`: Verifies WSGI `app` callable is exported for Gunicorn execution.
24. `test_render_deployment_configuration`: Verifies `render.yaml` blueprint contains required build, start, health check, and `PYTHON_VERSION` directives.
25. `test_api_predict_valid_pass`: Validates `POST /api/predict` with strong student inputs returns JSON `PASS`.
26. `test_api_predict_valid_fail`: Validates `POST /api/predict` with weak student inputs returns JSON `FAIL`.
27. `test_api_predict_missing_fields`: Validates missing payload keys return HTTP 400.
28. `test_api_predict_out_of_range`: Validates out-of-range API inputs return HTTP 400.
29. `test_api_predict_non_json_content_type`: Validates non-JSON content types return HTTP 400.
30. `test_api_predict_nan_and_inf`: Validates `NaN` and `Infinity` values are rejected by API validation.
31. `test_api_predict_boundary_inputs`: Asserts minimum (0.0) and maximum domain boundaries evaluate correctly.
32. `test_api_predict_scenario_comparison_shift`: Verifies scenario comparisons yield valid positive probability shifts on improved academic metrics.
33. `test_api_predict_string_numeric_coercion`: Verifies API gracefully parses string-encoded numbers in JSON payloads.
34. `test_static_assets_serving`: Verifies static CSS and JS are served with HTTP 200.
35. `test_index_contains_dynamic_elements`: Verifies dashboard DOM includes dynamic scenario simulator and live profile hooks.

---

## 🔒 Security & Backend Hardening

- **Strict Type & Range Validation**: All inputs undergo server-side numeric parsing and boundary checks before being converted into tensor arrays.
- **NaN & Infinity Rejection**: Explicit `math.isnan()` and `math.isinf()` checks guard against floating-point anomalies.
- **Request Size Limitation**: Enforces `MAX_CONTENT_LENGTH = 32 KB` to protect against denial-of-service payload attacks.
- **HTTP Security Headers**: Injected on all responses:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: SAMEORIGIN`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self';`
- **Output Artifact Allowlist**: Strict allowlist (`architecture.png`, `training_history.png`, `confusion_matrix.png`) prevents arbitrary file exposure or directory traversal via `/outputs`.
- **Zero Stack Trace Exposure**: Production error handlers (400, 404, 413, 500) log internal errors server-side without exposing internal paths or stack traces to clients.

---

## ☁️ Deployment Configuration (Render)

The application is cloud-ready and configured for continuous deployment on **Render**:

- **Blueprint File**: `render.yaml`
- **Runtime**: `Python 3.12` (`PYTHON_VERSION: 3.12.8`)
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 60 app:app
  ```
- **Health Check Path**: `/health`
- **Single Worker Architecture**: Single worker (`--workers 1 --threads 2`) prevents multi-process TensorFlow memory duplication on containerized cloud instances.

---

## ⚠️ Responsible Use & Limitations

- **Synthetic Dataset**: The dataset (`data/student_data.csv`) is synthetic and designed for educational modeling and demonstration.
- **Feature Scope**: The model uses only three features and cannot capture complex personal, health, socioeconomic, or institutional dynamics.
- **Probabilistic Interpretations**: Model outputs represent calibrated statistical probabilities, not absolute certainties or causal determinants.
- **Human Oversight**: Predictions are machine learning decision aids and must **never** be used as the sole basis for high-stakes academic decisions, grading, or disciplinary actions.

---

## 🔮 Future Improvements

- Integration with anonymized longitudinal institutional student databases.
- Multi-class academic outcome classification (letter grades A/B/C/D/F).
- Integration of Explainable AI (XAI) feature attribution algorithms (SHAP or Integrated Gradients).
- Real-time batch inference upload via CSV file.

---

## 🚀 Installation & Local Execution

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Jasveer4234/student-pass-fail-fnn.git
cd student-pass-fail-fnn
python -m venv .venv
.venv\Scripts\activate  # On Windows (.venv/bin/activate on Unix)
pip install -r requirements.txt
```

### 2. Run Local Web Server
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

### 3. Run CLI Inference
```bash
python src/predict.py --study-hours 7.5 --attendance 88 --previous-marks 80
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
