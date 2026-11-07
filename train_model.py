#!/usr/bin/env python3
"""
Model Training & Evaluation Script
=================================
Trains a RandomForestClassifier on motor_features.csv:
1. 80/20 Stratified train/test split
2. 5-Fold Stratified Cross-Validation
3. Evaluates Precision, Recall, F1-score
4. Generates confusion_matrix.png
5. Persists model to motor_model.pkl
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

FEATURES_FILE = "motor_features.csv"
MODEL_FILE = "motor_model.pkl"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"

FEATURE_COLUMNS = [
    "vibe_rms",
    "vibe_p2p",
    "vibe_kurtosis",
    "gyro_rms",
    "gyro_p2p",
    "temp_mean",
    "temp_slope"
]


def train():
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Missing '{FEATURES_FILE}'. Run feature extraction first.")

    df = pd.read_csv(FEATURES_FILE)
    print(f"[+] Loaded {len(df)} samples from '{FEATURES_FILE}'")
    print("Class Breakdown:\n", df["label"].value_counts().to_string(), "\n")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 1. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )

    # 3. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    print(f"[+] 5-Fold Stratified CV Accuracy: {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")

    # 4. Train on train split & evaluate on holdout test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"[✓] Holdout Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # 5. Confusion Matrix Heatmap
    unique_labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title("Motor State Classification - Confusion Matrix", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
    plt.close()
    print(f"[✓] Saved confusion matrix heatmap to '{CONFUSION_MATRIX_FILE}'.")

    # 6. Feature Importances
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nFeature Importances:\n", importances.to_string())

    # 7. Persist Model
    joblib.dump(model, MODEL_FILE)
    print(f"\n[✓] Saved trained model to '{MODEL_FILE}'.")


if __name__ == "__main__":
    train()
