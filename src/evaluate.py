"""Standalone evaluation script for Student Pass/Fail Prediction FNN.

Evaluates the trained model on the EXACT held-out test split created during training.
Calculates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
Generates and saves outputs/confusion_matrix.png.
"""

import json
import os
import sys

# Ensure root directory is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf

from src.utils import plot_confusion_matrix


def evaluate_model(
    data_path: str = "data/student_data.csv",
    indices_path: str = "models/split_indices.json",
    model_path: str = "models/student_pass_fail.keras",
    scaler_path: str = "models/scaler.pkl",
    output_cm_path: str = "outputs/confusion_matrix.png",
):
    print("=" * 65)
    print("STUDENT PASS/FAIL FNN - HELD-OUT TEST SET EVALUATION")
    print("=" * 65)

    # 1. Verify existence of artifacts
    for path, name in [
        (data_path, "Dataset"),
        (indices_path, "Split indices"),
        (model_path, "Trained model"),
        (scaler_path, "Preprocessing scaler"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} artifact not found at: {path}. Please run train.py first.")

    # 2. Load dataset and reproduce exact held-out test set
    df = pd.read_csv(data_path)
    with open(indices_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_indices = split_data["test_indices"]
    test_df = df.iloc[test_indices].copy()

    feature_cols = ["Study_Hours", "Attendance", "Previous_Marks"]
    target_col = "Result"

    X_test_raw = test_df[feature_cols].to_numpy()
    y_test = test_df[target_col].to_numpy()

    print(f"\n[1/4] Loaded exact held-out test split: {len(test_df)} samples")
    print(f"      - Random seed used in split: {split_data.get('random_seed', 42)}")
    print(f"      - Test Fail count (0)      : {np.sum(y_test == 0)}")
    print(f"      - Test Pass count (1)      : {np.sum(y_test == 1)}")

    # 3. Load preprocessing scaler and transform test data (No refitting)
    print(f"\n[2/4] Applying fitted StandardScaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test_raw)

    # 4. Load trained Keras model and run inference
    print(f"\n[3/4] Loading trained model from {model_path} and generating predictions...")
    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # 5. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # 6. Save confusion matrix visualization
    print(f"\n[4/4] Generating confusion matrix visualization in {output_cm_path}...")
    plot_confusion_matrix(y_test, y_pred, output_path=output_cm_path)
    print(f"      [OK] Saved {output_cm_path}")

    # Display clean formatted results
    print("\n" + "=" * 65)
    print("FINAL HELD-OUT TEST EVALUATION REPORT")
    print("=" * 65)
    print(f"  * Test Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  * Precision     : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  * Recall        : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  * F1-Score      : {f1:.4f}  ({f1 * 100:.2f}%)")
    print("-" * 65)
    print("  * Confusion Matrix Breakdown:")
    print(f"      - True Negatives  (TN - Correct Fail): {tn}")
    print(f"      - False Positives (FP - Incorrect Pass): {fp}")
    print(f"      - False Negatives (FN - Incorrect Fail): {fn}")
    print(f"      - True Positives  (TP - Correct Pass): {tp}")
    print("=" * 65)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


if __name__ == "__main__":
    evaluate_model()
