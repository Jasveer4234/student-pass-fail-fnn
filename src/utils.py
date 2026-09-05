"""Utility functions for data preprocessing, model building, and visualization.

This module enforces strict data leakage prevention by ensuring StandardScaler
is fitted exclusively on the training split, sets deterministic random seeds,
constructs the exact 73-parameter Feedforward Neural Network, and generates
clear visual artifacts.
"""

import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential, layers


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility.
    
    Args:
        seed (int): Random seed value. Defaults to 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess_data(
    csv_path: str = "data/student_data.csv",
    seed: int = 42,
    save_indices_path: str = "models/split_indices.json"
):
    """Load dataset and perform a strict, leakage-free stratified 70/15/15 split.
    
    Order of operations:
    1. Load raw CSV data.
    2. Separate feature matrix X and binary target vector y.
    3. Perform stratified split into Train (70%) and Temp (30%).
    4. Perform stratified split of Temp into Validation (15%) and Test (15%).
    5. Fit StandardScaler ONLY on X_train.
    6. Transform X_train, X_val, and X_test using the same fitted scaler.
    7. Persist split indices for exact reproducibility in standalone evaluation.

    Args:
        csv_path (str): Path to data/student_data.csv.
        seed (int): Random seed for stratified splitting.
        save_indices_path (str): Path to serialize train/val/test index splits.

    Returns:
        dict: Dictionary containing scaled feature splits, target splits,
              raw feature splits, fitted scaler, and indices.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = ["Study_Hours", "Attendance", "Previous_Marks"]
    target_col = "Result"

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    indices = np.arange(len(df))

    # Step 1: 70% Train, 30% Temp (stratified by y)
    (
        X_train_raw,
        X_temp_raw,
        y_train,
        y_temp,
        idx_train,
        idx_temp,
    ) = train_test_split(
        X,
        y,
        indices,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    # Step 2: Split 30% Temp into 15% Validation and 15% Test (50% of 30%, stratified)
    (
        X_val_raw,
        X_test_raw,
        y_val,
        y_test,
        idx_val,
        idx_test,
    ) = train_test_split(
        X_temp_raw,
        y_temp,
        idx_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )

    # Step 3: Fit scaler ONLY on X_train array (Zero Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw.to_numpy())
    X_val_scaled = scaler.transform(X_val_raw.to_numpy())
    X_test_scaled = scaler.transform(X_test_raw.to_numpy())

    # Save split indices for exact evaluation reproducibility
    if save_indices_path:
        os.makedirs(os.path.dirname(save_indices_path), exist_ok=True)
        split_data = {
            "random_seed": seed,
            "train_indices": idx_train.tolist(),
            "val_indices": idx_val.tolist(),
            "test_indices": idx_test.tolist(),
            "total_samples": len(df),
            "train_samples": len(idx_train),
            "val_samples": len(idx_val),
            "test_samples": len(idx_test),
        }
        with open(save_indices_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "raw_df": df,
    }


def build_model() -> tf.keras.Model:
    """Build the Feedforward Neural Network (FNN) according to the approved spec.
    
    Architecture:
        Input(3) -> Dense(8, relu) -> Dense(4, relu) -> Dense(1, sigmoid)
        
    Trainable Parameters:
        Layer 1 (Dense 8): (3 * 8) + 8 = 32
        Layer 2 (Dense 4): (8 * 4) + 4 = 36
        Layer 3 (Dense 1): (4 * 1) + 1 = 5
        Total = 32 + 36 + 5 = 73 parameters.

    Returns:
        tf.keras.Model: Uncompiled Keras Sequential model.
    """
    model = Sequential(
        [
            layers.Input(shape=(3,), name="input_layer"),
            layers.Dense(8, activation="relu", name="hidden_layer_1"),
            layers.Dense(4, activation="relu", name="hidden_layer_2"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="Student_Pass_Fail_FNN",
    )

    # Strict parameter count assertion
    param_count = model.count_params()
    if param_count != 73:
        raise ValueError(
            f"Expected exactly 73 trainable parameters, but got {param_count}."
        )

    return model


def plot_architecture(output_path: str = "outputs/architecture.png") -> None:
    """Generate a clean, professional architecture diagram of the 3 -> 8 -> 4 -> 1 FNN.
    
    Args:
        output_path (str): Destination path for the image artifact.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.axis("off")

    layer_sizes = [3, 8, 4, 1]
    layer_names = [
        "Input Layer\n(3 Features)",
        "Hidden Layer 1\n(8 Neurons, ReLU)",
        "Hidden Layer 2\n(4 Neurons, ReLU)",
        "Output Layer\n(1 Neuron, Sigmoid)",
    ]
    feature_labels = ["Study Hours", "Attendance", "Previous Marks"]

    v_spacing = 0.8
    h_spacing = 2.8

    # Calculate coordinates for each layer's neurons
    layer_coords = []
    for i, size in enumerate(layer_sizes):
        x = i * h_spacing
        y_offset = (max(layer_sizes) - size) * v_spacing / 2.0
        coords = [(x, y_offset + j * v_spacing) for j in range(size)]
        layer_coords.append(coords)

    # Draw connection lines
    for i in range(len(layer_sizes) - 1):
        for src_x, src_y in layer_coords[i]:
            for dst_x, dst_y in layer_coords[i + 1]:
                ax.plot(
                    [src_x, dst_x],
                    [src_y, dst_y],
                    color="#94a3b8",
                    alpha=0.4,
                    linewidth=0.8,
                    zorder=1,
                )

    # Colors for layers
    colors = ["#38bdf8", "#818cf8", "#a78bfa", "#34d399"]

    # Draw neurons and annotations
    for i, (coords, color, name) in enumerate(
        zip(layer_coords, colors, layer_names)
    ):
        for j, (x, y) in enumerate(coords):
            circle = plt.Circle(
                (x, y),
                0.22,
                color=color,
                ec="#0f172a",
                lw=1.5,
                zorder=2,
            )
            ax.add_patch(circle)

            # Neuron index label
            ax.text(
                x,
                y,
                f"N{j+1}" if i > 0 else f"X{j+1}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#0f172a",
                zorder=3,
            )

            # Feature name annotations for input layer
            if i == 0:
                ax.text(
                    x - 0.35,
                    y,
                    feature_labels[j],
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#1e293b",
                )
            elif i == len(layer_sizes) - 1:
                ax.text(
                    x + 0.35,
                    y,
                    "Pass Probability P(Pass)\n[0 = Fail, 1 = Pass]",
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#1e293b",
                )

        # Layer title at top
        max_y = max(y for _, y in coords)
        ax.text(
            coords[0][0],
            max_y + 0.65,
            name,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    # Parameter Breakdown Header & Box
    param_text = (
        "Trainable Parameters Breakdown:\n"
        "- Input -> Hidden 1:  3 x 8 + 8 = 32 weights & biases\n"
        "- Hidden 1 -> Hidden 2: 8 x 4 + 4 = 36 weights & biases\n"
        "- Hidden 2 -> Output:   4 x 1 + 1 = 5 weights & biases\n"
        "- Total Trainable Parameters: 73"
    )
    plt.figtext(
        0.5,
        0.05,
        param_text,
        ha="center",
        fontsize=9.5,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            lw=1.2,
        ),
    )

    plt.title(
        "Student Pass/Fail Prediction - Feedforward Neural Network Architecture (3 -> 8 -> 4 -> 1)",
        fontsize=12,
        fontweight="bold",
        pad=25,
        color="#0f172a",
    )

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_training_history(
    history, output_path: str = "outputs/training_history.png"
) -> None:
    """Plot training and validation loss and accuracy curves over epochs.
    
    Args:
        history (tf.keras.callbacks.History): Training history object.
        output_path (str): File destination for history plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Binary Crossentropy Loss
    ax1.plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax1.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
        color="#dc2626",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=3,
    )
    ax1.set_title("Model Loss (Binary Crossentropy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Classification Accuracy
    ax2.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
        color="#16a34a",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax2.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="#ea580c",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=3,
    )
    ax2.set_title("Model Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(
        "FNN Training & Validation History across Epochs",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "outputs/confusion_matrix.png",
) -> np.ndarray:
    """Generate and save an annotated confusion matrix heatmap.
    
    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels (threshold = 0.5).
        output_path (str): Destination file path for heatmap.
        
    Returns:
        np.ndarray: 2x2 confusion matrix array.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Annotation strings with count and percentage
    total = np.sum(cm)
    labels = [
        [f"True Neg (TN)\n{cm[0, 0]}\n({cm[0, 0]/total*100:.1f}%)",
         f"False Pos (FP)\n{cm[0, 1]}\n({cm[0, 1]/total*100:.1f}%)"],
        [f"False Neg (FN)\n{cm[1, 0]}\n({cm[1, 0]/total*100:.1f}%)",
         f"True Pos (TP)\n{cm[1, 1]}\n({cm[1, 1]/total*100:.1f}%)"]
    ]

    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=["Fail (0)", "Pass (1)"],
        yticklabels=["Fail (0)", "Pass (1)"],
        linewidths=1,
        linecolor="#cbd5e1",
    )

    ax.set_title("Held-out Test Set Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    return cm
