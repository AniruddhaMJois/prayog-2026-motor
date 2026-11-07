#!/usr/bin/env python3
"""
Multi-Sensor Feature Extraction & Model Training Pipeline
=========================================================
Extracts physical and statistical features from ALL sensors:
1. MPU-6050: 3-axis Accelerometer & 3-axis Gyroscope
2. ACS712: Motor Current (Amps)
3. Internal & External DS18B20 Temperature (°C)
4. Tachometer / Encoder Speed (RPM)

Trains a Random Forest Classifier with 5-fold Stratified Cross-Validation
and saves the trained model to motor_model.pkl.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
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

RAW_DATA_FILE = "clean_motor_dataset.csv"
FEATURES_DATA_FILE = "motor_features.csv"
MODEL_OUTPUT_FILE = "motor_model.pkl"
CONFUSION_MATRIX_PNG = "confusion_matrix.png"

WINDOW_SIZE = 25  # 25 samples = 500 ms @ 50 Hz (2 records/sec)
DT = 0.02         # 20 ms interval

FEATURE_COLUMNS = [
    "vibe_rms",
    "vibe_p2p",
    "vibe_kurtosis",
    "gyro_rms",
    "gyro_p2p",
    "current_mean",
    "current_std",
    "current_peak",
    "temp_mean",
    "temp_slope",
    "temp_ext_mean",
    "rpm_mean"
]


def extract_window_features(window_df: pd.DataFrame) -> dict:
    """
    Computes statistical and physical features across a 25-sample window
    integrating MPU-6050, Current, Temperature, and Speed sensors.
    """
    ax = window_df["ax_g"].values
    ay = window_df["ay_g"].values
    az = window_df["az_g"].values
    gx = window_df["gx_ds"].values
    gy = window_df["gy_ds"].values
    gz = window_df["gz_ds"].values
    temp = window_df["temp_C"].values
    current = window_df["current_A"].values if "current_A" in window_df.columns else np.zeros(len(ax))
    temp_ext = window_df["temp_external_C"].values if "temp_external_C" in window_df.columns else temp
    rpm = window_df["rpm"].values if "rpm" in window_df.columns else np.zeros(len(ax))

    # 1. Vibration Dynamics (MPU-6050 Accelerometer)
    vibe_mag = np.sqrt(ax**2 + ay**2 + az**2)
    vibe_rms = float(np.sqrt(np.mean(vibe_mag**2)))
    vibe_p2p = float(np.ptp(vibe_mag))
    vibe_kurt = float(kurtosis(vibe_mag, fisher=True))

    # 2. Rotational Oscillation (MPU-6050 Gyroscope)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_rms = float(np.sqrt(np.mean(gyro_mag**2)))
    gyro_p2p = float(np.ptp(gyro_mag))

    # 3. Electrical Current Features (ACS712)
    current_mean = float(np.mean(current))
    current_std = float(np.std(current))
    current_peak = float(np.max(current))

    # 4. Thermal Dynamics (Internal + DS18B20 External)
    temp_mean = float(np.mean(temp))
    time_axis = np.arange(len(temp)) * DT
    if np.all(temp == temp[0]):
        temp_slope = 0.0
    else:
        temp_slope = float(np.polyfit(time_axis, temp, 1)[0])
    temp_ext_mean = float(np.mean(temp_ext))

    # 5. Rotational Speed (RPM)
    rpm_mean = float(np.mean(rpm))

    return {
        "vibe_rms": round(vibe_rms, 4),
        "vibe_p2p": round(vibe_p2p, 4),
        "vibe_kurtosis": round(vibe_kurt, 4),
        "gyro_rms": round(gyro_rms, 4),
        "gyro_p2p": round(gyro_p2p, 4),
        "current_mean": round(current_mean, 3),
        "current_std": round(current_std, 4),
        "current_peak": round(current_peak, 3),
        "temp_mean": round(temp_mean, 2),
        "temp_slope": round(temp_slope, 4),
        "temp_ext_mean": round(temp_ext_mean, 2),
        "rpm_mean": round(rpm_mean, 1)
    }


def process_rolling_features(raw_file: str, output_file: str) -> pd.DataFrame:
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Input file '{raw_file}' does not exist.")

    df_raw = pd.read_csv(raw_file)
    print(f"\n[+] Loaded {len(df_raw)} raw multi-sensor records from '{raw_file}'")

    feature_records = []
    classes = df_raw["label"].unique()
    print(f"[+] Classes detected: {classes.tolist()}")

    for label in classes:
        group_df = df_raw[df_raw["label"] == label].reset_index(drop=True)
        total_samples = len(group_df)
        num_windows = total_samples // WINDOW_SIZE

        if num_windows == 0:
            step = max(1, total_samples // 10)
            indices = range(0, max(1, total_samples - WINDOW_SIZE + 1), step)
        else:
            indices = [i * WINDOW_SIZE for i in range(num_windows)]

        for start_idx in indices:
            end_idx = start_idx + WINDOW_SIZE
            if end_idx <= total_samples:
                window_df = group_df.iloc[start_idx:end_idx]
            else:
                window_df = group_df.iloc[max(0, total_samples - WINDOW_SIZE):]

            feats = extract_window_features(window_df)
            feats["label"] = label
            feature_records.append(feats)

        print(f"  [✓] '{label}': {total_samples} raw samples -> {len([r for r in feature_records if r['label'] == label])} feature windows")

    features_df = pd.DataFrame(feature_records)
    features_df.to_csv(output_file, index=False)
    print(f"\n[✓] Exported {len(features_df)} multi-sensor feature records to '{output_file}'")
    return features_df


def train_and_evaluate(features_file: str):
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file '{features_file}' not found.")

    df = pd.read_csv(features_file)
    print(f"\n=======================================================")
    print(f"  Multi-Sensor Model Training & 5-Fold Cross-Validation")
    print(f"=======================================================")
    print(f"[+] Total feature instances: {len(df)}")
    print("Class distribution:\n", df["label"].value_counts().to_string())

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 1. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\n[+] Train set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 2. Balanced Random Forest Classifier
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
    print(f"\n[+] 5-Fold Stratified CV Accuracy: {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")
    for fold_idx, score in enumerate(cv_scores, 1):
        print(f"    Fold {fold_idx}: {score * 100:.2f}%")

    # 4. Train on train split & evaluate on holdout test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Holdout Test Accuracy (20% split): {test_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # 5. Feature Importances across All Sensors
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("Multi-Sensor Feature Importances:")
    for feat, imp in importances.items():
        print(f"  - {feat:15s}: {imp * 100:6.2f}%")

    # 6. Confusion Matrix Heatmap
    unique_labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title("Multi-Sensor Motor State Classification - Confusion Matrix", fontsize=12, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close()
    print(f"\n[✓] Saved labeled confusion matrix plot to '{CONFUSION_MATRIX_PNG}'")

    # 7. Model Serialization
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print(f"[✓] Successfully serialized multi-sensor model artifact to '{MODEL_OUTPUT_FILE}'\n")


def main():
    print("=" * 65)
    print(" All-Sensors Motor Fault Diagnostic Pipeline: Extract & Train")
    print("=" * 65)
    process_rolling_features(RAW_DATA_FILE, FEATURES_DATA_FILE)
    train_and_evaluate(FEATURES_DATA_FILE)


if __name__ == "__main__":
    main()
