"""
utils/ml.py
Handles the full ML pipeline: dataset generation, model training,
and prediction. If saved model files aren't found (e.g. missed on
a git push), the app trains a fresh model automatically on first
load — so the deployed app can never show "model not found."
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
FEATURES = [
    "study_hours_per_day", "attendance_percent", "previous_exam_score",
    "sleep_hours", "extracurricular_activities", "part_time_job",
]


def _generate_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    study_hours = np.clip(rng.normal(4, 2, n), 0, 10)
    attendance = np.clip(rng.normal(80, 12, n), 40, 100)
    previous_score = np.clip(rng.normal(65, 15, n), 20, 100)
    sleep_hours = np.clip(rng.normal(6.5, 1.3, n), 3, 10)
    extracurricular = rng.choice([0, 1], n, p=[0.55, 0.45])
    part_time_job = rng.choice([0, 1], n, p=[0.7, 0.3])

    final_score = (
        study_hours * 3.0 + attendance * 0.30 + previous_score * 0.25
        + sleep_hours * 1.0 - part_time_job * 6 + extracurricular * 1.5
        - 10 + rng.normal(0, 9, n)
    )
    final_score = np.clip(final_score, 0, 100)
    passed = (final_score >= 50).astype(int)

    return pd.DataFrame({
        "study_hours_per_day": np.round(study_hours, 1),
        "attendance_percent": np.round(attendance, 1),
        "previous_exam_score": np.round(previous_score, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "extracurricular_activities": extracurricular,
        "part_time_job": part_time_job,
        "final_exam_score": np.round(final_score, 1),
        "passed": passed,
    })


def _train_all_models(df):
    X, y = df[FEATURES], df["passed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "y_test": y_test,
            "y_pred": preds,
        }

    best_name = max(results, key=lambda k: results[k]["f1"])
    return best_name, results, scaler


@st.cache_resource(show_spinner="Preparing the ML model...")
def load_or_train():
    """Loads saved model artifacts if present; otherwise trains fresh
    ones on the fly. Always returns a working model — never crashes
    the app with a missing-file error."""
    paths = {k: os.path.join(MODEL_DIR, f"{k}.pkl") for k in
              ["model", "scaler", "features", "model_name"]}

    if all(os.path.exists(p) for p in paths.values()):
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        features = joblib.load(paths["features"])
        model_name = joblib.load(paths["model_name"])
        df = pd.read_csv(os.path.join(MODEL_DIR, "student_data.csv"))
        # Rebuild comparison results for the Model Insights page
        best_name, results, _ = _train_all_models(df)
        return {
            "model": model, "scaler": scaler, "features": features,
            "model_name": model_name, "df": df, "results": results,
        }

    # Fallback: generate data + train from scratch (deployment-safe)
    df = _generate_dataset()
    best_name, results, scaler = _train_all_models(df)
    return {
        "model": results[best_name]["model"], "scaler": scaler,
        "features": FEATURES, "model_name": best_name,
        "df": df, "results": results,
    }


def predict(bundle, profile: dict):
    input_df = pd.DataFrame([profile])[bundle["features"]]
    scaled = bundle["scaler"].transform(input_df)
    pred = bundle["model"].predict(scaled)[0]
    prob = bundle["model"].predict_proba(scaled)[0][1]
    return pred, prob


def predict_batch(bundle, df: pd.DataFrame):
    input_df = df[bundle["features"]]
    scaled = bundle["scaler"].transform(input_df)
    preds = bundle["model"].predict(scaled)
    probs = bundle["model"].predict_proba(scaled)[:, 1]
    out = df.copy()
    out["predicted_outcome"] = np.where(preds == 1, "Pass", "Fail")
    out["pass_probability_%"] = np.round(probs * 100, 1)
    return out


def get_feature_importance(bundle):
    model = bundle["model"]
    features = bundle["features"]
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        return None
    return pd.Series(importance, index=features).sort_values(ascending=False)


def get_confusion_matrix(bundle):
    r = bundle["results"][bundle["model_name"]]
    return confusion_matrix(r["y_test"], r["y_pred"])


def get_classification_report(bundle):
    r = bundle["results"][bundle["model_name"]]
    return classification_report(r["y_test"], r["y_pred"], target_names=["Fail", "Pass"])
