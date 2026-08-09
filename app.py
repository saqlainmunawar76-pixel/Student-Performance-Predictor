"""
app.py
Student Performance Predictor — a Streamlit app that uses a trained
Scikit-learn model to predict whether a student will pass or fail,
based on study habits and lifestyle factors.

Internship: AI & ML Intern @ Codomax Digital Solutions
Module 6 — Day 21-24: Final AI & ML Project
Author: Saqlain Munawar
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Predictor")
st.caption("Predict whether a student is likely to pass or fail their final exam, "
           "based on study habits and lifestyle — powered by a trained Machine Learning model.")


@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    features = joblib.load("model/features.pkl")
    model_name = joblib.load("model/model_name.pkl")
    return model, scaler, features, model_name


try:
    model, scaler, FEATURES, MODEL_NAME = load_model()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Make sure `model/model.pkl`, `scaler.pkl`, "
             "`features.pkl`, and `model_name.pkl` are in the `model/` folder.")
    st.stop()

st.sidebar.header("📋 Student Profile")
study_hours = st.sidebar.slider("Study hours per day", 0.0, 10.0, 4.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 40, 100, 80)
previous_score = st.sidebar.slider("Previous exam score", 20, 100, 65)
sleep_hours = st.sidebar.slider("Sleep hours per night", 3.0, 10.0, 6.5, 0.5)
extracurricular = st.sidebar.selectbox("Involved in extracurricular activities?", ["No", "Yes"])
part_time_job = st.sidebar.selectbox("Has a part-time job?", ["No", "Yes"])

predict_button = st.sidebar.button("🔮 Predict Outcome", type="primary", use_container_width=True)

st.markdown(f"**Model in use:** `{MODEL_NAME}`")
st.divider()

if predict_button:
    input_data = pd.DataFrame([{
        "study_hours_per_day": study_hours,
        "attendance_percent": attendance,
        "previous_exam_score": previous_score,
        "sleep_hours": sleep_hours,
        "extracurricular_activities": 1 if extracurricular == "Yes" else 0,
        "part_time_job": 1 if part_time_job == "Yes" else 0,
    }])[FEATURES]

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.success(f"### ✅ Predicted: PASS")
    else:
        st.error(f"### ❌ Predicted: FAIL")

    st.metric("Probability of Passing", f"{probability * 100:.1f}%")
    st.progress(float(probability))

    st.markdown("#### 📋 Profile Summary")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    with st.expander("💡 What influences this prediction?"):
        st.markdown("""
        - **Study hours** and **attendance** are strong positive factors
        - A higher **previous exam score** tends to predict continued success
        - Adequate **sleep** supports better academic performance
        - Having a **part-time job** can reduce available study time
        - **Extracurricular activities** show a mild positive association
        """)
else:
    st.info("👈 Adjust the student profile in the sidebar and click **Predict Outcome**.")

st.divider()
st.caption("Final AI & ML Project — Module 6, AI & ML Internship at Codomax Digital Solutions.")
