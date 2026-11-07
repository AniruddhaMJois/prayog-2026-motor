#!/usr/bin/env python3
"""
Dual-Table Supabase Real-Time Diagnostic Engine
===============================================
Simultaneously fetches and correlates telemetry from BOTH Supabase tables:
1. 'mpu6050_telemetry' -> High-frequency 50 Hz Accelerometer, Gyroscope & Internal Temp
2. 'motor_telemetry'    -> Current Sensor (ACS712), External Temp (DS18B20), Speed (RPM)

Extracts combined multi-sensor features, runs inference with 'motor_model.pkl',
and logs the diagnosis to 'motor_predictions' table.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import joblib

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wyjalrrxuppksycuulzx.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Load from local .env if available
if (not SUPABASE_KEY or not SUPABASE_URL) and os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SUPABASE_KEY="):
                SUPABASE_KEY = line.split("=", 1)[1]
            elif line.startswith("SUPABASE_URL="):
                SUPABASE_URL = line.split("=", 1)[1]

MODEL_PATH = "motor_model.pkl"
WINDOW_SIZE = 25
DT = 0.02

FEATURE_NAMES = [
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

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def fetch_table_rows(table_name, limit=WINDOW_SIZE, order="id.desc", offset=0):
    """
    Fetches rows from a specific Supabase table using specified order and offset.
    - order='id.desc' for live hardware stream
    - order='id.asc' for replay from starting records
    """
    url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*&order={order}&limit={limit}&offset={offset}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data
    except Exception as e:
        return []


def post_prediction_to_supabase(record):
    """
    Logs prediction outcome to Supabase motor_predictions table.
    """
    url = f"{SUPABASE_URL}/rest/v1/motor_predictions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    req = urllib.request.Request(url, data=json.dumps(record).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status in (200, 201)
    except Exception as e:
        print(f"{YELLOW}[Warning]{RESET} Failed to log prediction to Supabase: {e}")
        return False


def extract_features(df_mpu, df_motor=None):
    """
    Extracts 12-dimensional multi-sensor feature vector combining BOTH tables:
    - MPU columns from 'mpu6050_telemetry': ax_g, ay_g, az_g, gx_ds, gy_ds, gz_ds, temp_c
    - Motor columns from 'motor_telemetry': current_a, temp_external_c, rpm
    """
    ax = df_mpu["ax_g"].values
    ay = df_mpu["ay_g"].values
    az = df_mpu["az_g"].values
    gx = df_mpu["gx_ds"].values
    gy = df_mpu["gy_ds"].values
    gz = df_mpu["gz_ds"].values
    temp = df_mpu["temp_c"].values

    # Electrical & Auxiliary sensors from motor_telemetry
    if df_motor is not None and not df_motor.empty:
        current = df_motor["current_a"].values if "current_a" in df_motor.columns else np.full(len(ax), 1.05)
        temp_ext = df_motor["temp_external_c"].values if "temp_external_c" in df_motor.columns and df_motor["temp_external_c"].notnull().any() else temp
        rpm = df_motor["rpm"].values if "rpm" in df_motor.columns and df_motor["rpm"].notnull().any() else np.full(len(ax), 1485.0)
    else:
        current = df_mpu["current_a"].values if "current_a" in df_mpu.columns else np.full(len(ax), 1.05)
        temp_ext = df_mpu["temp_external_c"].values if "temp_external_c" in df_mpu.columns else temp
        rpm = df_mpu["rpm"].values if "rpm" in df_mpu.columns else np.full(len(ax), 1485.0)

    # 1. Vibration
    vibe_mag = np.sqrt(ax**2 + ay**2 + az**2)
    vibe_rms = float(np.sqrt(np.mean(vibe_mag**2)))
    vibe_p2p = float(np.ptp(vibe_mag))
    vibe_kurtosis = float(kurtosis(vibe_mag, fisher=True))

    # 2. Gyroscope
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_rms = float(np.sqrt(np.mean(gyro_mag**2)))
    gyro_p2p = float(np.ptp(gyro_mag))

    # 3. Current
    current_mean = float(np.mean(current))
    current_std = float(np.std(current))
    current_peak = float(np.max(current))

    # 4. Temperature
    temp_mean = float(np.mean(temp))
    time_axis = np.arange(len(temp)) * DT
    if np.all(temp == temp[0]):
        temp_slope = 0.0
    else:
        temp_slope = float(np.polyfit(time_axis, temp, 1)[0])
    temp_ext_mean = float(np.mean(temp_ext))

    # 5. Speed (RPM)
    rpm_mean = float(np.mean(rpm))

    return {
        "vibe_rms": round(vibe_rms, 4),
        "vibe_p2p": round(vibe_p2p, 4),
        "vibe_kurtosis": round(vibe_kurtosis, 4),
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


def analyze_telemetry(model, mode="live", offset=0, verbose=True):
    """
    Analyzes telemetry from Supabase:
    - mode="live": Fetches latest incoming records from connected hardware (id.desc)
    - mode="start": Fetches sequential records starting from beginning (id.asc)
    """
    order = "id.desc" if mode == "live" else "id.asc"
    mpu_rows = fetch_table_rows("mpu6050_telemetry", limit=WINDOW_SIZE, order=order, offset=offset)
    if not mpu_rows:
        mpu_rows = fetch_table_rows("mpu_telemetry", limit=WINDOW_SIZE, order=order, offset=offset)

    if not mpu_rows:
        if verbose:
            print(f"{YELLOW}[!] No data in 'mpu6050_telemetry' (mode={mode}, offset={offset}). Waiting for stream...{RESET}")
        return None

    if len(mpu_rows) < WINDOW_SIZE:
        if verbose:
            print(f"{YELLOW}[i] Buffering... {len(mpu_rows)} / {WINDOW_SIZE} samples in 'mpu6050_telemetry'.{RESET}")
        return None

    motor_rows = fetch_table_rows("motor_telemetry", limit=WINDOW_SIZE, order=order, offset=offset)

    if mode == "live":
        df_mpu = pd.DataFrame(mpu_rows).iloc[::-1].reset_index(drop=True)
        df_motor = pd.DataFrame(motor_rows).iloc[::-1].reset_index(drop=True) if motor_rows else None
    else:
        df_mpu = pd.DataFrame(mpu_rows).reset_index(drop=True)
        df_motor = pd.DataFrame(motor_rows).reset_index(drop=True) if motor_rows else None

    # Extract 12-dimensional features
    feats = extract_features(df_mpu, df_motor)

    feature_df = pd.DataFrame([[
        feats["vibe_rms"],
        feats["vibe_p2p"],
        feats["vibe_kurtosis"],
        feats["gyro_rms"],
        feats["gyro_p2p"],
        feats["current_mean"],
        feats["current_std"],
        feats["current_peak"],
        feats["temp_mean"],
        feats["temp_slope"],
        feats["temp_ext_mean"],
        feats["rpm_mean"]
    ]], columns=FEATURE_NAMES)

    pred_label = model.predict(feature_df)[0]
    proba = model.predict_proba(feature_df)[0]
    class_idx = list(model.classes_).index(pred_label)
    confidence = float(proba[class_idx] * 100.0)

    # Multi-Sensor Diagnostics & Verdicts
    if pred_label == "Normal":
        alert_level = "NOMINAL"
        status_banner = f"{GREEN}🟢 HEALTHY / NOMINAL{RESET}"
        summary_verdict = (
            f"Vibration ({feats['vibe_rms']:.2f}g) and Current ({feats['current_mean']:.2f}A) are within safe baseline tolerances. "
            f"Motor speed is nominal at {feats['rpm_mean']:.0f} RPM with stable temperature ({feats['temp_mean']:.1f}°C)."
        )
    elif pred_label == "Imbalance":
        alert_level = "WARNING"
        status_banner = f"{YELLOW}🟠 IMBALANCE WARNING{RESET}"
        summary_verdict = (
            f"Excessive rotor vibration ({feats['vibe_rms']:.2f}g RMS, Peak-to-Peak {feats['vibe_p2p']:.2f}g) "
            f"with elevated current draw ({feats['current_mean']:.2f}A ± {feats['current_std']:.2f}A). "
            f"Inspect rotor shaft balance and mechanical coupling."
        )
    else:  # Stall
        alert_level = "CRITICAL"
        status_banner = f"{RED}🔴 CRITICAL FAULT: MOTOR STALLED{RESET}"
        summary_verdict = (
            f"CRITICAL OVERLOAD: Locked-rotor current surge ({feats['current_mean']:.2f}A, Peak {feats['current_peak']:.2f}A) "
            f"with ZERO rotation ({feats['rpm_mean']:.0f} RPM) and external thermal buildup ({feats['temp_ext_mean']:.1f}°C). "
            f"CUT MOTOR POWER IMMEDIATELY to prevent winding damage!"
        )

    # Log to Supabase motor_predictions table
    prediction_record = {
        "motor_id": df_mpu["device_id"].iloc[-1] if "device_id" in df_mpu.columns else "MOTOR_01",
        "predicted_state": pred_label,
        "confidence": round(confidence, 2),
        "vibe_rms": feats["vibe_rms"],
        "vibe_p2p": feats["vibe_p2p"],
        "vibe_kurtosis": feats["vibe_kurtosis"],
        "gyro_rms": feats["gyro_rms"],
        "temp_mean": feats["temp_mean"],
        "temp_slope": feats["temp_slope"],
        "alert_level": alert_level
    }
    post_prediction_to_supabase(prediction_record)

    diagnosis = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "state": pred_label,
        "confidence": confidence,
        "alert_level": alert_level,
        "verdict": summary_verdict,
        "features": feats,
        "mpu_id_range": f"#{mpu_rows[0]['id']} -> #{mpu_rows[-1]['id']}" if mpu_rows else None,
        "motor_id_range": f"#{motor_rows[0]['id']} -> #{motor_rows[-1]['id']}" if motor_rows else None
    }

    if verbose:
        mode_title = "CONNECTED LIVE HARDWARE (Real-Time)" if mode == "live" else f"REPLAY FROM START (Offset: {offset})"
        print(f"\n{BOLD}{'=' * 85}{RESET}")
        print(f" [{mode_title}] {status_banner}  |  Confidence: {BOLD}{confidence:.1f}%{RESET}")
        print(f"{BOLD}{'=' * 85}{RESET}")
        print(f" [1] Table 'mpu6050_telemetry' (IDs: {diagnosis['mpu_id_range']}):")
        print(f"     • Vibration (RMS / P2P / Kurt): {feats['vibe_rms']:.3f} g  /  {feats['vibe_p2p']:.3f} g  /  {feats['vibe_kurtosis']:.2f}")
        print(f"     • Gyroscope (RMS / P2P)       : {feats['gyro_rms']:.2f} °/s  /  {feats['gyro_p2p']:.2f} °/s")
        print(f"     • Internal Temp (MPU-6050)    : {feats['temp_mean']:.1f} °C (Slope: {feats['temp_slope']:+.4f} °C/s)")
        print(f" [2] Table 'motor_telemetry'   (IDs: {diagnosis['motor_id_range']}):")
        print(f"     • Current (Mean / Peak / Std) : {feats['current_mean']:.2f} A  /  {feats['current_peak']:.2f} A  /  {feats['current_std']:.3f} A")
        print(f"     • Motor Speed (Tachometer)    : {feats['rpm_mean']:.0f} RPM")
        print(f"     • External Temp (DS18B20)     : {feats['temp_ext_mean']:.1f} °C")
        print(f" -----------------------------------------------------------------------------------")
        print(f" • DIAGNOSTIC VERDICT : {summary_verdict}")
        print(f" • Supabase Database  : Prediction recorded to 'motor_predictions' table [✓]")
        print(f"{'=' * 85}\n", flush=True)

    return diagnosis


def main():
    parser = argparse.ArgumentParser(description="Fetch from both mpu6050_telemetry and motor_telemetry")
    parser.add_argument("--mode", choices=["live", "start"], default="live",
                        help="'live' = Connected live hardware (newest rows); 'start' = Replay from starting rows (id.asc)")
    parser.add_argument("--offset", type=int, default=0, help="Starting row offset when using --mode start (default: 0)")
    parser.add_argument("--loop", action="store_true", help="Continuously poll Supabase")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"{RED}[Error]{RESET} Model '{MODEL_PATH}' not found. Run 'python3 feature_extraction_and_train.py' first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print(f"{GREEN}[✓]{RESET} Loaded Multi-Sensor Model: '{MODEL_PATH}'")
    print(f"{GREEN}[✓]{RESET} Connected to Supabase: {SUPABASE_URL}")
    print(f"{CYAN}[i]{RESET} Ingestion Option: {BOLD}{'🟢 Connected Live Hardware' if args.mode == 'live' else '⏪ Replay from Starting (Row #1)'}{RESET}")
    print(f"{CYAN}[i]{RESET} Fetching from BOTH tables: 'mpu6050_telemetry' and 'motor_telemetry'")

    if args.loop:
        print(f"{CYAN}[i]{RESET} Running continuous monitor (interval: {args.interval}s). Press Ctrl+C to stop.\n")
        offset = args.offset
        try:
            while True:
                analyze_telemetry(model, mode=args.mode, offset=offset, verbose=True)
                if args.mode == "start":
                    offset += 5  # Advance playback window
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{CYAN}[i] Stopped monitoring.{RESET}")
    else:
        analyze_telemetry(model, mode=args.mode, offset=args.offset, verbose=True)


if __name__ == "__main__":
    main()
