"""Realistic educational student dataset generator.

Creates 300 student records with realistic feature distributions and balanced pass/fail outcomes.
Features:
- Study_Hours: Daily hours studied (1.0 to 11.0)
- Attendance: Class attendance percentage (42.0 to 98.0)
- Previous_Marks: Score in previous exams out of 100 (26.0 to 96.0)
- Result: 0 = Fail, 1 = Pass
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # 1. Study Hours: normal distribution clipped to [1.0, 11.0]
    study_hours = rng.normal(loc=5.0, scale=2.1, size=n_samples)
    study_hours = np.clip(study_hours, 1.0, 11.0).round(1)

    # 2. Attendance: normal distribution clipped to [42.0, 98.0]
    attendance = rng.normal(loc=74.0, scale=13.0, size=n_samples)
    attendance = np.clip(attendance, 42.0, 98.0).round(1)

    # 3. Previous Marks: correlated with attendance & study hours + noise
    marks_latent = 0.35 * attendance + 2.8 * study_hours + rng.normal(loc=20.0, scale=10.0, size=n_samples)
    previous_marks = np.clip(marks_latent, 26.0, 96.0).round(1)

    # 4. Realistic Decision Boundary
    norm_study = (study_hours - 1.0) / (11.0 - 1.0)
    norm_att = (attendance - 42.0) / (98.0 - 42.0)
    norm_marks = (previous_marks - 26.0) / (96.0 - 26.0)

    # Base composite metric
    composite = 0.38 * norm_study + 0.32 * norm_att + 0.30 * norm_marks

    # Non-linear interaction penalties and bonuses
    synergy = np.where((attendance >= 75.0) & (study_hours >= 5.0), 0.08, 0.0)
    penalty = np.where((attendance < 60.0) & (study_hours < 3.0), -0.12, 0.0)
    
    # Realistic borderline noise (exam day variance, difficult question luck)
    noise = rng.normal(loc=0.0, scale=0.04, size=n_samples)
    final_score = composite + synergy + penalty + noise

    # Threshold for balanced ~54% Pass and ~46% Fail
    threshold = 0.465
    result = (final_score >= threshold).astype(int)

    df = pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Result": result,
    })

    return df


if __name__ == "__main__":
    df = generate_student_dataset(n_samples=300, seed=42)
    os.makedirs("data", exist_ok=True)
    csv_path = "data/student_data.csv"
    df.to_csv(csv_path, index=False)
    
    counts = df["Result"].value_counts()
    print(f"Generated {len(df)} samples saved to {csv_path}")
    print(f"Class distribution:")
    print(f"  - Fail (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  - Pass (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)")
    print("\nSample records:")
    print(df.head(10))
