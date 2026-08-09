"""
Generates a realistic synthetic student performance dataset and builds
the complete ML pipeline that will also be documented in the Colab notebook.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

study_hours = np.clip(np.random.normal(4, 2, n), 0, 10)
attendance = np.clip(np.random.normal(80, 12, n), 40, 100)
previous_score = np.clip(np.random.normal(65, 15, n), 20, 100)
sleep_hours = np.clip(np.random.normal(6.5, 1.3, n), 3, 10)
extracurricular = np.random.choice([0, 1], n, p=[0.55, 0.45])
part_time_job = np.random.choice([0, 1], n, p=[0.7, 0.3])

# Final score built from a realistic weighted combination + noise
final_score = (
    study_hours * 3.0
    + attendance * 0.30
    + previous_score * 0.25
    + sleep_hours * 1.0
    - part_time_job * 6
    + extracurricular * 1.5
    - 10  # baseline difficulty offset
    + np.random.normal(0, 9, n)
)
final_score = np.clip(final_score, 0, 100)
passed = (final_score >= 50).astype(int)

df = pd.DataFrame({
    "study_hours_per_day": np.round(study_hours, 1),
    "attendance_percent": np.round(attendance, 1),
    "previous_exam_score": np.round(previous_score, 1),
    "sleep_hours": np.round(sleep_hours, 1),
    "extracurricular_activities": extracurricular,
    "part_time_job": part_time_job,
    "final_exam_score": np.round(final_score, 1),
    "passed": passed,
})

df.to_csv("/home/claude/Student-Performance-Predictor/model/student_data.csv", index=False)
print(df.head())
print("\nShape:", df.shape)
print("\nPass rate:", df["passed"].mean())
