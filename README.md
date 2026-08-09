# 🎓 Student Performance Predictor

**Final AI & ML Project** — Module 6, Day 21–24
**Internship:** AI & ML Intern @ Codomax Digital Solutions
**Intern:** Saqlain Munawar

## 📌 About This Project

A complete, end-to-end Machine Learning project that predicts whether a student will **pass or fail** their final exam based on study habits and lifestyle factors — including a Jupyter/Colab notebook covering the full ML workflow, and a **live interactive Streamlit app** where anyone can enter a student profile and get an instant prediction.

This is the capstone project of my AI & ML Internship, bringing together everything learned across all previous modules:
- Python fundamentals (Module 1)
- Data analysis & visualization with Pandas/NumPy/Matplotlib (Module 3)
- Machine Learning with Scikit-learn (Module 4)
- Building real, deployed AI-powered applications (Module 5)

## 🎯 Problem Statement

Given a student's study hours, attendance, previous exam score, sleep habits, extracurricular involvement, and whether they hold a part-time job — predict whether they will **pass (≥50%)** or **fail** their final exam.

## 📊 Dataset

A synthetic dataset of 500 students, generated with realistic relationships between features and outcomes (more study hours and better attendance increase pass likelihood; a part-time job reduces it), plus random noise so patterns aren't artificially perfect.

| Feature | Description |
|---|---|
| `study_hours_per_day` | Average daily study hours |
| `attendance_percent` | Class attendance percentage |
| `previous_exam_score` | Score on the previous exam |
| `sleep_hours` | Average nightly sleep hours |
| `extracurricular_activities` | Involved in extracurriculars (0/1) |
| `part_time_job` | Has a part-time job (0/1) |
| `final_exam_score` | Final exam score (target basis) |
| `passed` | Pass/Fail label (target) |

## 🧠 ML Workflow

1. **Data creation & exploration** — shape, summary stats, class balance
2. **Data cleaning check** — missing values, duplicates
3. **Visualization** — scatter plots (study hours & attendance vs score), pass/fail distribution
4. **Feature preparation** — train/test split (80/20), feature scaling
5. **Model training & comparison** — Logistic Regression, Decision Tree, Random Forest
6. **Evaluation** — accuracy, F1 score, classification report, confusion matrix
7. **Prediction** — tested on a new, unseen student profile

**Best model:** Logistic Regression (accuracy: 71%, F1: 0.67 on the test set)

## 🖥️ Live App

The trained model, scaler, and feature list are saved (`model/*.pkl`) and loaded directly into a **Streamlit app** (`app.py`) — enter a student's profile in the sidebar and get an instant Pass/Fail prediction with probability.

## 📂 Project Structure
```
Student-Performance-Predictor/
├── app.py                     # Streamlit prediction app
├── generate_data.py           # Synthetic dataset generation
├── train_model.py             # Model training & comparison script
├── requirements.txt
├── notebook/
│   └── Student_Performance_Predictor.ipynb   # Full ML workflow (Colab-ready)
└── model/
    ├── student_data.csv       # Generated dataset
    ├── model.pkl              # Trained model (best performer)
    ├── scaler.pkl             # Feature scaler
    ├── features.pkl           # Feature list
    └── model_name.pkl         # Name of the best model
```

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Deployment
Deployed on **Streamlit Community Cloud** — no API keys required, since this app runs entirely on a locally-trained Scikit-learn model.

## 📓 Notebook
Open `notebook/Student_Performance_Predictor.ipynb` in **Google Colab** (Upload notebook) to see the complete workflow with all outputs and charts — or re-run it end-to-end (it's fully self-contained, no file uploads needed).

## 🔗 Deliverables
- **GitHub Repository:** this repo
- **LinkedIn Post:** sharing this capstone project
