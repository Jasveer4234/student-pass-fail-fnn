"""Training pipeline for Student Pass/Fail Prediction Feedforward Neural Network.

This script executes the complete, leak-free ML workflow:
1. Sets deterministic seeds (seed = 42).
2. Loads student_data.csv and performs an exact stratified 70/15/15 split.
3. Fits StandardScaler ONLY on X_train and transforms splits.
4. Serializes the fitted scaler to models/scaler.pkl and split indices to models/split_indices.json.
5. Constructs the exact 73-parameter 3 -> 8 -> 4 -> 1 FNN.
6. Compiles with Adam optimizer and binary crossentropy loss.
7. Trains using EarlyStopping (monitoring val_loss, restoring best weights).
8. Saves the trained model to models/student_pass_fail.keras.
9. Generates outputs/architecture.png and outputs/training_history.png.
"""

import os
import sys

# Ensure root directory is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.utils import (
    set_seed,
    load_and_preprocess_data,
    build_model,
    plot_architecture,
    plot_training_history,
)


def train_model(
    data_path: str = "data/student_data.csv",
    model_save_path: str = "models/student_pass_fail.keras",
    scaler_save_path: str = "models/scaler.pkl",
    indices_save_path: str = "models/split_indices.json",
    seed: int = 42,
    epochs: int = 150,
    batch_size: int = 16,
):
    print("=" * 65)
    print("STUDENT PASS/FAIL FNN TRAINING PIPELINE")
    print("=" * 65)

    import sklearn
    import matplotlib
    import seaborn
    import pytest
    print("\nEnvironment Packages:")
    print(f"  - tensorflow == {tf.__version__}")
    print(f"  - pandas     == {pd.__version__}")
    print(f"  - numpy      == {np.__version__}")
    print(f"  - scikit-learn == {sklearn.__version__}")
    print(f"  - matplotlib == {matplotlib.__version__}")
    print(f"  - seaborn    == {seaborn.__version__}")
    print(f"  - joblib     == {joblib.__version__}")
    print(f"  - pytest     == {pytest.__version__}")

    # 1. Reproducibility
    print(f"\n[1/6] Setting random seed = {seed}...")
    set_seed(seed)

    # 2. Preprocessing & Leakage-Free Splitting
    print(f"\n[2/6] Loading data and executing stratified 70/15/15 split...")
    data_dict = load_and_preprocess_data(
        csv_path=data_path,
        seed=seed,
        save_indices_path=indices_save_path,
    )

    X_train_scaled = data_dict["X_train_scaled"]
    y_train = data_dict["y_train"]
    X_val_scaled = data_dict["X_val_scaled"]
    y_val = data_dict["y_val"]
    X_test_scaled = data_dict["X_test_scaled"]
    y_test = data_dict["y_test"]
    scaler = data_dict["scaler"]
    raw_df = data_dict["raw_df"]

    total_samples = len(raw_df)
    train_count = len(y_train)
    val_count = len(y_val)
    test_count = len(y_test)

    print(f"      - Total dataset size : {total_samples} samples")
    print(f"      - Training split     : {train_count} samples ({train_count/total_samples*100:.1f}%)")
    print(f"      - Validation split   : {val_count} samples ({val_count/total_samples*100:.1f}%)")
    print(f"      - Held-out Test split: {test_count} samples ({test_count/total_samples*100:.1f}%)")

    # Class distribution analysis
    print("\n      Class Distribution:")
    for split_name, targets in [("Overall", raw_df["Result"]), ("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = pd.Series(targets).value_counts().sort_index().index, pd.Series(targets).value_counts().sort_index().values
        dist_str = ", ".join([f"Class {u}: {c} ({c/len(targets)*100:.1f}%)" for u, c in zip(unique, counts)])
        print(f"        * {split_name:<8}: {dist_str}")

    # 3. Save Preprocessing Artifact
    print(f"\n[3/6] Persisting fitted StandardScaler to {scaler_save_path}...")
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print("      [OK] Scaler fitted strictly on X_train (Zero Data Leakage).")

    # 4. Model Architecture & Verification
    print(f"\n[4/6] Constructing FNN Model (3 -> 8 -> 4 -> 1)...")
    model = build_model()
    model.summary()

    param_count = model.count_params()
    print(f"\n      [OK] Verified Trainable Parameters: {param_count} (Expected: 73)")
    if param_count != 73:
        raise RuntimeError(f"Parameter count mismatch: got {param_count}, expected 73.")

    # 5. Compilation & Training
    print(f"\n[5/6] Compiling and training model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Save trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\n      [OK] Best model saved to: {model_save_path}")

    # 6. Generate Visual Artifacts
    print(f"\n[6/6] Generating visual artifacts in outputs/...")
    plot_architecture("outputs/architecture.png")
    plot_training_history(history, "outputs/training_history.png")
    print("      [OK] Saved outputs/architecture.png")
    print("      [OK] Saved outputs/training_history.png")

    # Summary of training performance using restored best weights
    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss_hist = val_losses[best_epoch - 1]
    total_epochs = len(val_losses)

    # Directly evaluate the restored best-weights model on train and val sets
    train_eval = model.evaluate(X_train_scaled, y_train, verbose=0)
    val_eval = model.evaluate(X_val_scaled, y_val, verbose=0)
    train_loss, train_acc = train_eval[0], train_eval[1]
    val_loss, val_acc = val_eval[0], val_eval[1]

    print("\n" + "=" * 65)
    print(f"Training Complete across {total_epochs} epochs")
    print(f"   Best weights restored from Epoch {best_epoch} (min val_loss: {best_val_loss_hist:.4f})")
    print(f"   Restored Model Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Restored Model Val Loss  : {val_loss:.4f} | Val Accuracy  : {val_acc*100:.2f}%")
    print("=" * 65)

    return model, history, scaler, data_dict


if __name__ == "__main__":
    train_model()
