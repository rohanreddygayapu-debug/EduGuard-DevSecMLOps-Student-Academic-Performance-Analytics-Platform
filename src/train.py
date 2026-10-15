"""
train.py
MLOps Pipeline:
Model: Logistic Regression + StandardScaler; L2, C=1.0, lbfgs, max_iter=1000, random_state=42
- Scikit-learn Pipeline
- Evaluation Metrics (ROC-AUC, F1, Accuracy)
- Cryptographic SHA-256 Signing
- Baseline Drift Profile Generation
"""

import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import get_train_test_splits, load_student_data
from security_scanner import sign_model_artifact


def train_student_analytics_model():
    base_dir = os.path.dirname(CURRENT_DIR)
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 65)
    print("DevSecMLOps Pipeline: Student Academic Performance Analytics")
    print("=" * 65)

    # 1. Load Data
    print("[1/5] Loading and splitting student dataset...")
    X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.20, random_state=42)
    print(f"      Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"      Features: {list(X_train.columns)}")

    # 2. Build Pipeline verbatim to reference:
    # Logistic Regression + StandardScaler; L2, C=1.0, lbfgs, max_iter=1000, random_state=42
    print("[2/5] Initializing StandardScaler + LogisticRegression Pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])

    # 3. Train
    print("[3/5] Fitting model on training cohort...")
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    print("[4/5] Evaluating performance metrics...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "model_architecture": "Logistic Regression + StandardScaler; L2, C=1.0, lbfgs, max_iter=1000, random_state=42"
    }

    print("      Evaluation Results:")
    for k, v in metrics.items():
        print(f"      - {k.upper()}: {v}")

    # MLOps Quality Gate
    if metrics["accuracy"] < 0.75:
        raise ValueError(f"[GATE FAILED] Accuracy {metrics['accuracy']} below required threshold 0.75")
    print("      [PASSED] MLOps Quality Gate: Model meets deployment standards.")

    # 5. Serialization and Cryptographic Signing
    model_path = os.path.join(models_dir, "student_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    baseline_path = os.path.join(models_dir, "baseline_stats.json")

    print("[5/5] Saving model and generating cryptographic SHA-256 signature...")
    joblib.dump(pipeline, model_path)
    sha256_digest = sign_model_artifact(model_path)

    print(f"      Model artifact: {model_path}")
    print(f"      SHA-256 Digest: {sha256_digest}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save baseline drift statistics
    raw_df = load_student_data()
    baseline_stats = {
        col: {"mean": float(raw_df[col].mean()), "std": float(raw_df[col].std())}
        for col in ["Attendance_Percentage", "Study_Hours_per_Week", "LMS_Hours", "Internal_Marks"]
    }
    with open(baseline_path, "w") as f:
        json.dump(baseline_stats, f, indent=4)

    print("=" * 65)
    print("Pipeline Execution Complete: Artifacts Ready for Secure Serving")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    train_student_analytics_model()
