# Student Pass/Fail Prediction using Feedforward Neural Network (FNN)

An educational B.Tech machine learning project implementing a Feedforward Neural Network (FNN) in TensorFlow/Keras to predict student pass/fail academic outcomes from behavioral and historical performance metrics. Includes both a command-line interface (CLI) and a simple, local browser-based interface using Flask.

Developed strictly to educational standards with deterministic reproducibility, explicit zero data leakage enforcement, full test suite coverage, and an interactive local web dashboard.

---

## 📌 1. Project Overview

Predicting student academic standing early enables targeted academic interventions. This project models the non-linear relationship between student engagement and final course outcomes using a locked, lightweight Feedforward Neural Network.

The project provides dual access interfaces:
1. **Terminal Workflow (CLI & Scripts)**: Training, evaluation, and command-line predictions.
2. **Local Browser Web Interface**: A clean, responsive Flask web application for interactive single-student predictions.

---

## ✨ 2. Features

- **Strict Leakage-Free Preprocessing**: `StandardScaler` is fitted **strictly on the 70% training split** and applied downstream to validation, test, and inference.
- **Deterministic 70/15/15 Stratified Split**: Fixed random state (`seed=42`) with serialized split indices (`models/split_indices.json`) guaranteeing exact test evaluation reproducibility.
- **Strict Parameter Count**: Exactly **73 trainable parameters** with zero unneeded overhead.
- **Local Browser-Based UI**: Lightweight, responsive Flask web interface with server-side input validation and real-time pass/fail probability calculation.
- **Comprehensive Automated Tests**: 15 unit, integration, and web endpoint tests covering dataset hygiene, parameter counts, boundary checks, zero data leakage, and Flask routes.
- **Visual Artifacts**: Auto-generated architecture diagram, training history curves, and annotated confusion matrix heatmap.

---

## 🏛️ 3. Neural Network Architecture

The model is a 3-layer Feedforward Neural Network (FNN) configured with non-linear ReLU activations in hidden layers and a Sigmoid activation at the output neuron for binary classification probability $P(\text{Pass} \mid X)$.

```
Input (3 Features) 
       │
       ▼
┌────────────────────────────────────────┐
│  Dense Layer 1: 8 Neurons (ReLU)       │ ──> (3 × 8) weights + 8 biases = 32 params
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Dense Layer 2: 4 Neurons (ReLU)       │ ──> (8 × 4) weights + 4 biases = 36 params
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Output Layer: 1 Neuron (Sigmoid)      │ ──> (4 × 1) weights + 1 bias   =  5 params
└────────────────────────────────────────┘
       │
       ▼
  P(Pass) ∈ [0.0, 1.0]
```

### Parameter Calculation Verification
$$\text{Total Trainable Parameters} = 32 + 36 + 5 = \mathbf{73}$$

| Layer | Input Dimension | Output Dimension | Weights | Biases | Total Layer Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hidden Layer 1** | 3 | 8 | $3 \times 8 = 24$ | 8 | **32** |
| **Hidden Layer 2** | 8 | 4 | $8 \times 4 = 32$ | 4 | **36** |
| **Output Layer** | 4 | 1 | $4 \times 1 = 4$ | 1 | **5** |
| **Total** | | | **60** | **13** | **73** |

---

## 📊 4. Dataset Specifications

> [!NOTE]
> The dataset (`data/student_data.csv`) is **synthetically generated** for educational and demonstration purposes; it does not represent real student records.

The dataset comprises **300 student records** generated with realistic distributions and feature interactions:

| Feature Name | Description | Range | Data Type |
| :--- | :--- | :---: | :---: |
| `Study_Hours` | Average daily self-study hours | 1.0 – 11.0 hrs | Float |
| `Attendance` | Class attendance percentage | 42.0% – 98.0% | Float |
| `Previous_Marks`| Score in previous evaluations | 26.0 – 96.0 / 100 | Float |
| `Result` *(Target)* | Final student status (`0` = Fail, `1` = Pass) | {0, 1} | Binary Int |

### Class Distribution
- **Fail (`0`)**: 136 samples (45.3%)
- **Pass (`1`)**: 164 samples (54.7%)

### Stratified 70 / 15 / 15 Split Breakdown
- **Training Set (70%)**: 210 samples (Fail: 95, Pass: 115)
- **Validation Set (15%)**: 45 samples (Fail: 21, Pass: 24)
- **Held-out Test Set (15%)**: 45 samples (Fail: 20, Pass: 25)

---

## 📁 5. Repository Structure

```
student-pass-fail-fnn/
├── .gitignore                      # Git ignore patterns for Python, caches, and environments
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive project documentation and verified metrics
├── requirements.txt                # Pinned dependencies (TensorFlow, scikit-learn, Flask, etc.)
├── app.py                          # Local Flask web application for browser-based predictions
├── data/
│   ├── generate_dataset.py         # Realistic student dataset generator
│   └── student_data.csv            # 300 student records CSV
├── models/
│   ├── scaler.pkl                  # StandardScaler fitted strictly on training data
│   ├── split_indices.json          # Deterministic train/val/test index splits
│   └── student_pass_fail.keras     # Trained TensorFlow/Keras FNN model artifact
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
│   └── style.css                   # Custom CSS styling for Flask web dashboard
├── templates/
│   └── index.html                  # HTML template for browser prediction UI
└── tests/
    └── test_model.py               # 15 automated unit, integration, and web endpoint tests
```

---

## 🚀 6. Installation

Clone or navigate to the project directory:

```bash
cd C:\Users\JASVEER\Downloads\student-pass-fail-fnn
```

Activate your Python virtual environment (if using `.venv` on Windows):

```bash
.venv\Scripts\activate
```

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 💻 7. Terminal Usage

The CLI and pipeline scripts remain the source of truth for the machine learning workflow.

### A. Dataset Generation (Optional)
To regenerate the 300-sample educational dataset:
```bash
python data/generate_dataset.py
```

### B. Model Training
Run the complete training pipeline. This fits the scaler on the training set, trains the 73-parameter FNN, saves model artifacts, and exports visual plots:
```bash
python src/train.py
```

### C. Standalone Evaluation
Evaluate the trained model on the exact held-out test split:
```bash
python src/evaluate.py
```

### D. Predict via Command-Line
Run CLI inference with arguments:
```bash
python src/predict.py --study-hours 7.5 --attendance 88 --previous-marks 80
```

Or run interactively in the terminal:
```bash
python src/predict.py
```

---

## 🌐 8. Browser Usage (Local Web App)

The Flask web application provides an intuitive local browser interface. It loads the pre-trained model (`models/student_pass_fail.keras`) and scaler (`models/scaler.pkl`) once on startup and runs locally on your machine.

> [!NOTE]
> This application runs strictly on your **local machine** (`127.0.0.1`). It is not hosted on the public internet.

### Steps to Run:

1. Navigate to the project directory:
   ```bash
   cd C:\Users\JASVEER\Downloads\student-pass-fail-fnn
   ```
2. Activate the virtual environment:
   ```bash
   .venv\Scripts\activate
   ```
3. Install required packages (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
4. Start the local Flask server:
   ```bash
   python app.py
   ```
5. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
6. Enter student metrics (Study Hours, Attendance %, Previous Marks %) and click **Predict Outcome**.

---

## 🎯 9. Example Prediction

### CLI Example
```bash
python src/predict.py --study-hours 7.5 --attendance 88 --previous-marks 80
```

*Output:*
```
=============================================
INFERENCE RESULT
=============================================
  Input Features:
    * Study Hours   : 7.50 hrs/day
    * Attendance    : 88.00%
    * Previous Marks: 80.00 / 100
---------------------------------------------
  Pass Probability: 99.99%
  Prediction      : PASS
=============================================
```

### Web Interface Example
- **Inputs**: Study Hours: `7.5`, Attendance: `88%`, Previous Marks: `80%`
- **Result Display**:
  - **Prediction**: `PASS`
  - **Pass Probability**: `99.99%`
  - **Fail Probability**: `0.01%`
  - **Submitted Features Summary**: `7.50 hrs | 88.00% | 80.00%`

---

## 🧪 10. Testing

Execute the complete test suite:

```bash
pytest -v
```

The test suite in `tests/test_model.py` runs **15 automated tests**:

1. `test_dataset_columns`: Asserts required columns exist without missing values.
2. `test_target_classes`: Validates binary classes `{0, 1}` and distribution balance.
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

---

## ⚙️ 11. Technical Details & Verification

### Actual Experimental Results
- **Optimization**: Adam Optimizer ($\text{learning rate} = 0.01$)
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 16
- **Epochs Trained**: 26 (EarlyStopping patience=20 restored best weights from **Epoch 6**)
- **Restored Model Training Accuracy**: **91.43%** | Restored Model Training Loss: **0.1822**
- **Restored Model Validation Accuracy**: **88.89%** | Restored Model Validation Loss: **0.2689**

### Held-Out Test Evaluation Metrics (45 Samples)
Evaluated strictly using `evaluate.py` on the exact held-out split:
- **Test Accuracy**: **93.33%** ($\frac{24 + 18}{45}$)
- **Precision**: **92.31%** ($\frac{24}{24 + 2}$)
- **Recall**: **96.00%** ($\frac{24}{24 + 1}$)
- **F1-Score**: **94.12%**

### Confusion Matrix Breakdown
```
                     Predicted Fail (0)     Predicted Pass (1)
 Actual Fail (0)             18 (TN)                 2 (FP)
 Actual Pass (1)              1 (FN)                24 (TP)
```

---

## ⚠️ 12. Limitations

- **Educational Synthetic Dataset**: The dataset was synthetically generated to model academic interactions for demonstration purposes. It does not represent real student institutional data.
- **Scope**: Designed as an academic B.Tech project demonstration. It is not intended for real-world administrative or high-stakes academic decision-making.
- **Local Deployment**: The Flask application is intended for local execution and demonstration; it does not include production WSGI server configurations (e.g., Gunicorn), user authentication, or persistent database storage.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
