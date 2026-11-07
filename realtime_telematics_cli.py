#!/usr/bin/env python3
"""
Real-Time Terminal Telematics & Fault Alert Monitor (CLI)
=========================================================
Runs real-time edge inference directly in the terminal:
1. Connects to ESP32 over serial (or runs simulation if hardware disconnected)
2. Maintains a 25-sample rolling buffer (500 ms @ 50 Hz = 2 inferences/sec)
3. Computes vibration, gyroscope, and thermal slope features
4. Classifies using 'motor_model.pkl'
5. Emits real-time clock (RTC) color-coded status warnings in the terminal
"""

import os
import sys
import time
import argparse
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import joblib

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

MODEL_PATH = "motor_model.pkl"
BUFFER_SIZE = 25
SAMPLING_FREQ = 50.0
DT = 1.0 / SAMPLING_FREQ

FEATURE_NAMES = [
    "vibe_rms",
    "vibe_p2p",
    "vibe_kurtosis",
    "gyro_rms",
    "gyro_p2p",
    "temp_mean",
    "temp_slope"
]

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"


def extract_features(buf):
    arr = np.array(buf)
    ax, ay, az = arr[:, 0], arr[:, 1], arr[:, 2]
    gx, gy, gz = arr[:, 3], arr[:, 4], arr[:, 5]
    temp = arr[:, 6]

    vibe_mag = np.sqrt(ax**2 + ay**2 + az**2)
    vibe_rms = float(np.sqrt(np.mean(vibe_mag**2)))
    vibe_p2p = float(np.ptp(vibe_mag))
    vibe_kurtosis = float(kurtosis(vibe_mag, fisher=True))

    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_rms = float(np.sqrt(np.mean(gyro_mag**2)))
    gyro_p2p = float(np.ptp(gyro_mag))

    temp_mean = float(np.mean(temp))
    time_axis = np.arange(len(temp)) * DT
    if np.all(temp == temp[0]):
        temp_slope = 0.0
    else:
        temp_slope = float(np.polyfit(time_axis, temp, 1)[0])

    return [vibe_rms, vibe_p2p, vibe_kurtosis, gyro_rms, gyro_p2p, temp_mean, temp_slope]


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) == 8:
        parts = parts[1:]
    if len(parts) == 7:
        try:
            return [float(x.strip()) for x in parts]
        except ValueError:
            return None
    return None


def simulate_sample(cycle_counter):
    # Cycles through Normal -> Imbalance -> Stall every 30 seconds
    mode_idx = (cycle_counter // 300) % 3
    if mode_idx == 0:
        return [
            np.random.normal(0.0, 0.05),
            np.random.normal(0.0, 0.05),
            np.random.normal(1.0, 0.04),
            np.random.normal(0.0, 2.0),
            np.random.normal(0.0, 2.0),
            np.random.normal(0.0, 2.0),
            41.8 + np.random.normal(0.0, 0.05)
        ]
    elif mode_idx == 1:
        t = time.time()
        return [
            -1.2 + 0.8 * np.sin(2 * np.pi * 15 * t) + np.random.normal(0.0, 0.1),
            0.6 * np.cos(2 * np.pi * 15 * t) + np.random.normal(0.0, 0.1),
            0.95 + np.random.normal(0.0, 0.08),
            40.0 * np.sin(2 * np.pi * 15 * t) + np.random.normal(0.0, 5.0),
            -60.0 * np.cos(2 * np.pi * 15 * t) + np.random.normal(0.0, 5.0),
            np.random.normal(0.0, 8.0),
            43.6 + np.random.normal(0.0, 0.08)
        ]
    else:
        return [
            -1.8 + np.random.normal(0.0, 0.15),
            0.8 + np.random.normal(0.0, 0.15),
            0.85 + np.random.normal(0.0, 0.05),
            np.random.normal(0.0, 15.0),
            np.random.normal(0.0, 20.0),
            np.random.normal(0.0, 6.0),
            44.5 + np.random.normal(0.0, 0.12)
        ]


def main():
    parser = argparse.ArgumentParser(description="Real-Time Terminal Telematics Monitor")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--sim", action="store_true", help="Force simulation mode")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"{RED}[Error]{RESET} Model '{MODEL_PATH}' not found. Run 'python3 feature_extraction_and_train.py' first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print(f"{GREEN}[✓]{RESET} Loaded ML model: {MODEL_PATH}")

    # Check serial port
    ser = None
    use_sim = args.sim

    if not use_sim and SERIAL_AVAILABLE:
        try:
            ser = serial.Serial(args.port, args.baud, timeout=0.1)
            print(f"{GREEN}[✓]{RESET} Connected to hardware serial port: {args.port} @ {args.baud} baud")
        except Exception as e:
            print(f"{YELLOW}[!] Serial port {args.port} not available ({e}).{RESET}")
            print(f"{CYAN}[i] Falling back to Real-Time Simulation / Demo Mode.{RESET}\n")
            use_sim = True
    else:
        use_sim = True
        print(f"{CYAN}[i] Running in Real-Time Simulation Mode.{RESET}\n")

    print(f"{BOLD}{'=' * 85}{RESET}")
    print(f"{BOLD} REAL-TIME MOTOR HEALTH TELEMATICS & FAULT ALERT STREAM{RESET}")
    print(f"{BOLD}{'=' * 85}{RESET}")
    print(f"{'RTC Timestamp':<24} | {'Status':<16} | {'Vibe RMS':<10} | {'Gyro RMS':<11} | {'Temp':<8} | {'Confidence':<10}")
    print("-" * 85)

    buffer = deque(maxlen=BUFFER_SIZE)
    cycle_counter = 0

    try:
        while True:
            if not use_sim and ser is not None:
                line = ser.readline().decode("utf-8", errors="ignore")
                sample = parse_line(line)
                if sample:
                    buffer.append(sample)
            else:
                sample = simulate_sample(cycle_counter)
                buffer.append(sample)
                cycle_counter += 1
                time.sleep(0.02)  # 50 Hz loop interval

            # When buffer fills, compute features and run inference
            if len(buffer) == BUFFER_SIZE and (cycle_counter % 25 == 0 or not use_sim):
                feats = extract_features(buffer)
                feature_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
                pred_label = model.predict(feature_df)[0]
                proba = model.predict_proba(feature_df)[0]
                class_idx = list(model.classes_).index(pred_label)
                confidence = proba[class_idx] * 100.0

                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                if pred_label == "Normal":
                    status_fmt = f"{GREEN}🟢 NOMINAL   {RESET}"
                elif pred_label == "Imbalance":
                    status_fmt = f"{YELLOW}🟠 IMBALANCE {RESET}"
                else:  # Stall
                    status_fmt = f"{RED}🔴 STALL CRIT{RESET}"

                vibe_rms = f"{feats[0]:.3f} g"
                gyro_rms = f"{feats[3]:.1f} °/s"
                temp_str = f"{feats[5]:.1f} °C"
                conf_str = f"{confidence:.1f} %"

                print(f"[{now_str}] | {status_fmt} | {vibe_rms:<10} | {gyro_rms:<11} | {temp_str:<8} | {conf_str:<10}", flush=True)

                if not use_sim:
                    buffer.clear()

    except KeyboardInterrupt:
        print(f"\n{CYAN}[i] Monitoring stopped by user.{RESET}")
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
