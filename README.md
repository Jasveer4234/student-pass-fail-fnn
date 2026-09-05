# Student Pass/Fail Prediction using Feedforward Neural Network (FNN)

An educational B.Tech machine learning project implementing a Feedforward Neural Network (FNN) in TensorFlow/Keras to predict student pass/fail academic outcomes from behavioral and historical performance metrics.

Developed strictly to educational standards with deterministic reproducibility, explicit zero data leakage enforcement, and full test suite coverage.

---

## 📌 Project Overview

Predicting student academic standing early enables targeted academic interventions. This project models the non-linear relationship between student engagement and final course outcomes using a locked, lightweight Feedforward Neural Network.

### Key Highlights
- **Strict Leakage-Free Preprocessing**: `StandardScaler` is fitted **strictly on the 70% training split** and applied downstream to validation, test, and inference.
- **Deterministic 70/15/15 Stratified Split**: Fixed random state (`seed=42`) with serialized split indices (`models/split_indices.json`) guaranteeing exact test evaluation reproducibility.
- **Strict Parameter Count**: Exactly **73 trainable parameters** with zero unneeded overhead.
- **Comprehensive Automated Tests**: 8 unit & integration tests covering dataset hygiene, parameter counts, boundary checks, and zero data leakage.
- **Visual Artifacts**: Auto-generated architecture diagram, training history curves, and annotated confusion matrix heatmap.

---

## 🏛️ Neural Network Architecture

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

## 📊 Dataset Specifications

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

## 📈 Actual Experimental Results

### Training Convergence
- **Optimization**: Adam Optimizer ($\text{learning rate} = 0.01$)
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 16
- **Epochs Trained**: 26 (EarlyStopping patience=20 restored best weights from **Epoch 6**)
- **Restored Model Training Accuracy**: **91.43%** | Restored Model Training Loss: **0.1822**
- **Restored Model Validation Accuracy**: **88.89%** | Restored Model Validation Loss: **0.2689**

### Held-Out Test Evaluation Metrics (45 Samples)
Evaluated strictly using `evaluate.py` on the exact held-out split:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{24 + 18}{45} = \mathbf{93.33\%}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{24}{24 + 2} = \mathbf{92.31\%}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{24}{24 + 1} = \mathbf{96.00\%}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{94.12\%}$$

### Confusion Matrix Breakdown
```
                     Predicted Fail (0)     Predicted Pass (1)
 Actual Fail (0)             18 (TN)                 2 (FP)
 Actual Pass (1)              1 (FN)                24 (TP)
```

---

## 📁 Repository Structure

```
student-pass-fail-fnn/
├── .gitignore                      # Git ignore patterns for Python, caches, and environments
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive project documentation and verified metrics
├── requirements.txt                # Fixed pinned dependencies
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
└── tests/
    └── test_model.py               # 8 automated unit & integration tests
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Dataset Generation (Optional)
To regenerate the 300-sample educational dataset:
```bash
python data/generate_dataset.py
```

### 3. Model Training
Run the complete training pipeline. This fits the scaler on the training set, trains the 73-parameter FNN, saves model artifacts, and exports visual plots:
```bash
python src/train.py
```

### 4. Standalone Evaluation
Evaluate the trained model on the exact held-out test split:
```bash
python src/evaluate.py
```

### 5. Predict on New Student Records
Run CLI inference:
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

Or run interactively:
```bash
python src/predict.py
```

### 6. Automated Testing
Execute the complete test suite:
```bash
pytest tests/test_model.py -v
```

---

## 🧪 Test Suite Summary

The automated test suite (`tests/test_model.py`) runs 8 comprehensive checks:

1. `test_dataset_columns`: Asserts required columns (`Study_Hours`, `Attendance`, `Previous_Marks`, `Result`) exist without missing values.
2. `test_target_classes`: Validates binary classes `{0, 1}` and balanced distribution.
3. `test_model_input_shape`: Validates model input shape is `(None, 3)`.
4. `test_model_output_shape`: Validates model output shape is `(None, 1)`.
5. `test_trainable_parameters_count`: Confirms total and trainable parameter counts are **strictly 73**.
6. `test_model_output_range`: Asserts model output is bounded in $[0, 1]$.
7. `test_preprocessing_no_data_leakage`: Verifies `StandardScaler` parameters equal training partition statistics and differ from test/full partitions.
8. `test_prediction_pipeline`: Tests end-to-end inference and input range validation errors.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
