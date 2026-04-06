"""
src/models/train_model.py
--------------------------
Trains a Random Forest classifier to predict SaaS customer churn.
Logs all experiments (parameters + metrics) to mlruns/ folder
so you can open them in MLflow UI later.

WHAT THIS FILE TEACHES:
- Train/test split (why we do it)
- Handling class imbalance with class_weight
- Evaluation metrics that matter for churn (Precision, Recall, F1, ROC-AUC)
- Saving the trained model with pickle
- MLflow experiment tracking (every run is logged separately)
"""

import pandas as pd
import numpy as np
import pickle, os, json, time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
)

# ── Paths ──────────────────────────────────────────────────
FEATURES_PATH = "data/features.csv"
MODEL_PATH    = "src/models/best_model.pkl"
MLRUNS_DIR    = "mlruns/experiment_1"

os.makedirs(MLRUNS_DIR, exist_ok=True)


def log_run(run_name: str, params: dict, metrics: dict):
    """
    Mimics MLflow run logging — saves params + metrics as JSON.
    In production you would use: mlflow.log_params() and mlflow.log_metrics()
    """
    run_id   = f"run_{int(time.time())}"
    run_dir  = os.path.join(MLRUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/params.json", "w") as f:
        json.dump({"run_name": run_name, **params}, f, indent=2)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  📁 Run logged → {run_dir}")
    return run_id


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Run all evaluation metrics and print a clean report."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
        "roc_auc"   : round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'─'*50}")
    print(f"  Model: {model_name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<12}  {v:.4f}  {bar}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

    return metrics


def train():
    # ── 1. Load features ───────────────────────────────────
    print("📂 Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    X  = df.drop(columns=["churned"])
    y  = df["churned"]
    print(f"   X shape: {X.shape}  |  Churn rate: {y.mean():.1%}")

    # ── 2. Train / Test split ──────────────────────────────
    # 80% train, 20% test — we NEVER let the model see test data during training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️  Split: {len(X_train)} train  |  {len(X_test)} test")

    # ── 3. Define models to compare ───────────────────────
    # This is the "experiment" part MLflow tracks for us
    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
    }

    # ── 4. Train, evaluate, and log each model ─────────────
    results = {}
    trained_models = {}

    for name, model in candidates.items():
        print(f"\n🔄 Training: {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test, name)
        results[name] = metrics
        trained_models[name] = model

        # Log to mlruns/ (same concept as mlflow.start_run())
        params = model.get_params()
        log_run(name, params, metrics)

    # ── 5. Pick best model by ROC-AUC ─────────────────────
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\n{'='*50}")
    print(f"🏆 Best Model: {best_name}")
    print(f"   ROC-AUC : {best_metrics['roc_auc']}")
    print(f"   F1 Score: {best_metrics['f1_score']}")
    print(f"   Recall  : {best_metrics['recall']}  ← most important for churn")
    print(f"{'='*50}")

    # ── 6. Feature Importance (Random Forest / GB only) ───
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        print("\n📊 Top 10 Feature Importances:")
        for feat, score in importances.head(10).items():
            bar = "█" * int(score * 100)
            print(f"   {feat:<28} {score:.4f}  {bar}")

    # ── 7. Save best model ─────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "model_name": best_name,
                     "metrics": best_metrics, "features": list(X.columns),
                     "trained_at": datetime.now().isoformat()}, f)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    # ── 8. Save summary ────────────────────────────────────
    summary = {"best_model": best_name, "metrics": best_metrics, "all_results": results}
    with open("mlruns/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Summary saved → mlruns/summary.json")

    return best_model, best_metrics


if __name__ == "__main__":
    train()
