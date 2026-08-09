# 🎓 PredictEd — Student Performance Predictor

**Final AI & ML Project** — Module 6, Day 21–24
**Internship:** AI & ML Intern @ Codomax Digital Solutions
**Intern:** Saqlain Munawar

## 📌 About This Project

A complete, end-to-end Machine Learning project — upgraded into a **premium SaaS-style dashboard app** — that predicts whether a student will **pass or fail** their final exam based on study habits and lifestyle factors.

This is the capstone project of my AI & ML Internship, bringing together everything learned across all previous modules:
- Python fundamentals (Module 1)
- Data analysis & visualization with Pandas/NumPy/Matplotlib (Module 3)
- Machine Learning with Scikit-learn (Module 4)
- Building real, deployed, SaaS-quality applications (Module 5)

## ✨ Features

| Page | What it does |
|---|---|
| 🏠 **Dashboard** | Greeting, quick actions, model accuracy, prediction stats, recent activity |
| 🔮 **Predict** | Enter a single student's profile and get an instant Pass/Fail prediction with probability |
| 📄 **Batch Predict** | Upload a CSV of many students and download predictions for all of them at once |
| 📊 **Model Insights** | Compares all 3 trained models, shows feature importance, confusion matrix & classification report |
| 🕓 **History** | Every prediction made is saved and browsable |

**Design:** a custom SaaS design system (gradient hero banners, cards, badges, full dark/light mode) — the same design language used across my other internship projects.

**Self-healing model loading:** if the saved model files aren't found (e.g. missed in a deployment), the app automatically regenerates the dataset and retrains the model on the fly — so it can never show a "model not found" error.

## 🎯 Problem Statement

Given a student's study hours, attendance, previous exam score, sleep habits, extracurricular involvement, and whether they hold a part-time job — predict whether they will **pass (≥50%)** or **fail** their final exam.

## 📊 Dataset

A synthetic dataset of 500 students, generated with realistic relationships between features and outcomes (more study hours and better attendance increase pass likelihood; a part-time job reduces it), plus random noise so patterns aren't artificially perfect.

## 🧠 ML Workflow

1. **Data creation & exploration**
2. **Data cleaning check** — missing values, duplicates
3. **Visualization** — Matplotlib charts (notebook) + live charts (app)
4. **Feature preparation** — train/test split (80/20), feature scaling
5. **Model training & comparison** — Logistic Regression, Decision Tree, Random Forest
6. **Evaluation** — accuracy, F1 score, classification report, confusion matrix
7. **Prediction** — single & batch predictions on new data

**Best model:** Logistic Regression (accuracy: ~71%, F1: ~0.67 on the test set)

## 📂 Project Structure
```
Student-Performance-Predictor/
├── app.py                     # SaaS-style Streamlit dashboard app
├── generate_data.py           # Synthetic dataset generation (standalone script)
├── train_model.py             # Model training & comparison (standalone script)
├── requirements.txt
├── utils/
│   ├── ml.py                  # ML pipeline: load/train model, predict, insights
│   ├── storage.py             # Local JSON storage for prediction history
│   └── styles.py              # SaaS design system CSS
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
Deployed on **Streamlit Community Cloud** — no API keys required, since this app runs entirely on a locally-trained Scikit-learn model. Thanks to the self-healing model loader, deployment works even if the `model/` folder isn't fully pushed.

## 📓 Notebook
Open `notebook/Student_Performance_Predictor.ipynb` in **Google Colab** to see the complete workflow with all outputs and charts — it's fully self-contained, no file uploads needed.

## 🔗 Deliverables
- **GitHub Repository:** this repo
- **LinkedIn Post:** sharing this capstone project
