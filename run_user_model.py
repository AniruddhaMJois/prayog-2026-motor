import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load your dataset (Replace with your actual CSV path)
df = pd.read_csv("mpu6050_telemetry.csv")

# 2. Features and Target
X = df[["ax_g", "ay_g", "az_g", "gx_ds", "gy_ds", "gz_ds", "temp_C"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Accuracy & Report
acc = model.score(X_test, y_test)
print(f"Model Accuracy: {acc * 100:.2f}%\n")
y_pred = model.predict(X_test)
y_repr = y_pred
print("Classification Report:")
print(classification_report(y_test, y_repr if "y_repr" in locals() else y_test))

# ---------------- GRAPH 1: CONFUSION MATRIX ----------------
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
)
plt.title(
    f"Confusion Matrix (Overall Accuracy: {acc*100:.1f}%)",
    fontsize=12,
    pad=10,
)
plt.xlabel("Predicted State")
plt.ylabel("Actual State")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ---------------- GRAPH 2: FEATURE IMPORTANCE ----------------
plt.figure(figsize=(7, 4))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=True
)
importances.plot(kind="barh", color="#2b5c8f")
plt.title("Sensor Feature Importance for Fault Detection", fontsize=12, pad=10)
plt.xlabel("Relative Gini Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("✓ Saved 'confusion_matrix.png' and 'feature_importance.png' successfully!")

# Copy to Antigravity brain artifact directory
ARTIFACT_DIR = "/home/anijois/.gemini/antigravity/brain/8d8a74c0-6f8b-4dfe-84e6-f91c0249af76"
if os.path.exists(ARTIFACT_DIR):
    for fname in ["confusion_matrix.png", "feature_importance.png"]:
        shutil.copy(fname, os.path.join(ARTIFACT_DIR, fname))
