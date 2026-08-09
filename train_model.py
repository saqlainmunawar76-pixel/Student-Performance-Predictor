"""
train_model.py
Trains and compares 3 classification models on the student performance
dataset, then saves the best-performing model + scaler for the app.
"""
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("/home/claude/Student-Performance-Predictor/model/student_data.csv")

FEATURES = [
    "study_hours_per_day", "attendance_percent", "previous_exam_score",
    "sleep_hours", "extracurricular_activities", "part_time_job"
]
X = df[FEATURES]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {"model": model, "accuracy": acc, "f1": f1}
    print(f"{name}: accuracy={acc:.3f}, f1={f1:.3f}")

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (f1={results[best_name]['f1']:.3f})")

print("\nClassification report for best model:")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

joblib.dump(best_model, "/home/claude/Student-Performance-Predictor/model/model.pkl")
joblib.dump(scaler, "/home/claude/Student-Performance-Predictor/model/scaler.pkl")
joblib.dump(FEATURES, "/home/claude/Student-Performance-Predictor/model/features.pkl")
joblib.dump(best_name, "/home/claude/Student-Performance-Predictor/model/model_name.pkl")

print("\nModel, scaler, and feature list saved.")
