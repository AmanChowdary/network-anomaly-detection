"""
Network Anomaly Detection — Training Pipeline
XGBoost + SMOTE + MLflow + GridSearchCV
"""
import os, json, warnings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, f1_score, roc_auc_score,
                              confusion_matrix, precision_score, recall_score)
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import mlflow, mlflow.xgboost
    MLFLOW_AVAILABLE = True
    _mlflow_db = os.path.abspath('mlruns/mlflow_tracking.db')
    mlflow.set_tracking_uri(f"sqlite:///{_mlflow_db}")
except ImportError:
    MLFLOW_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.extract import prepare_features

def train(data_path="data/network_traffic.csv", model_dir="model/artifacts"):
    os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path)
    X, y = prepare_features(df)
    print(f"  {len(df):,} events | {X.shape[1]} features | anomaly rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # SMOTE oversampling to address class imbalance
    if SMOTE_AVAILABLE:
        print("Applying SMOTE oversampling...")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {len(X_train):,} samples")
    else:
        print("  ⚠ imbalanced-learn not installed — skipping SMOTE (install with: pip install imbalanced-learn)")

    # Hyperparameter tuning via GridSearchCV
    print("\nRunning GridSearchCV (3-fold)...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base = xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc",
                               random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV F1:  {gs.best_score_:.4f}")

    model = gs.best_estimator_

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1":        round(f1_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "best_params": gs.best_params_,
        "trained_at": datetime.now().isoformat(),
        "smote_applied": SMOTE_AVAILABLE,
        "n_features": X.shape[1],
    }

    print("\n── Evaluation ──────────────────────────────────────────")
    for k in ["f1", "auc_roc", "precision", "recall"]:
        print(f"  {k:12s}: {metrics[k]:.4f}")
    print("\n── Classification Report ────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Normal","Anomaly"]))

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\n── Top 10 Features ──────────────────────────────────────")
    print(importances.head(10).to_string())

    # Save artifacts
    joblib.dump(model, f"{model_dir}/anomaly_model.joblib")
    joblib.dump(list(X.columns), f"{model_dir}/feature_names.joblib")
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    importances.head(20).to_json(f"{model_dir}/feature_importance.json")

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("network_anomaly_detection")
            with mlflow.start_run():
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                     if isinstance(v, (int, float))})
                mlflow.xgboost.log_model(model, "model")
                mlflow.log_artifact(f"{model_dir}/metrics.json")
                print("  ✓ Logged to MLflow")
        except Exception as e:
            print(f"  ⚠ MLflow tracking skipped ({type(e).__name__}: {e})")

    print(f"\n  ✓ Model saved → {model_dir}/")
    return model, metrics

if __name__ == "__main__":
    if not os.path.exists("data/network_traffic.csv"):
        print("Generating data first...")
        from data.generate_data import generate_network_data
        os.makedirs("data", exist_ok=True)
        generate_network_data(n=50000).to_csv("data/network_traffic.csv", index=False)
    train()
