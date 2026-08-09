"""
app.py
PredictEd — Student Performance Predictor, SaaS-style edition.

Internship: AI & ML Intern @ Codomax Digital Solutions
Module 6 — Day 21-24: Final AI & ML Project
Author: Saqlain Munawar
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils import ml, storage
from utils.styles import inject_css, card, empty_state

st.set_page_config(page_title="PredictEd", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

inject_css(st.session_state.theme)

bundle = ml.load_or_train()

NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Predict", "🔮"),
    ("Batch Predict", "📄"),
    ("Model Insights", "📊"),
    ("History", "🕓"),
]

with st.sidebar:
    st.markdown("## 🎓 PredictEd")
    st.caption("AI-powered student performance predictor")
    st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)

    for label, icon in NAV_ITEMS:
        active = st.session_state.page == label
        btn_type = "primary" if active else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True, type=btn_type):
            st.session_state.page = label
            st.rerun()

    st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)
    theme_label = "🌙 Dark Mode" if st.session_state.theme == "light" else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)
    st.caption(f"Model: **{bundle['model_name']}**")
    st.caption("Final AI & ML Project — Codomax Digital Solutions")


# ==================================================
# DASHBOARD
# ==================================================
def render_dashboard():
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    st.markdown(f"""
    <div class="sa-hero">
        <h1>{greeting}! 🎓</h1>
        <p>Predict student outcomes instantly with a trained Machine Learning model.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    actions = [("🔮 Predict a Student", "Predict"), ("📄 Batch Predict (CSV)", "Batch Predict"),
               ("📊 View Model Insights", "Model Insights")]
    for col, (label, target) in zip(cols, actions):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.page = target
                st.rerun()

    st.write("")
    stats = storage.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Total Predictions", str(stats["total_predictions"]))
    with c2:
        card("Predicted Pass", str(stats["pass_count"]), badge="success" if stats["pass_count"] else "")
    with c3:
        card("Predicted Fail", str(stats["fail_count"]))
    with c4:
        card("Avg. Pass Probability", f"{stats['avg_probability']}%")

    r = bundle["results"][bundle["model_name"]]
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        card("Model Accuracy", f"{r['accuracy']*100:.1f}%", sub=f"Best model: {bundle['model_name']}")
    with c2:
        card("Model F1 Score", f"{r['f1']:.2f}", sub="Balances precision & recall")

    st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### 🕓 Recent Predictions")
    history = storage.get_history()[:5]
    if not history:
        empty_state("🔮", "No predictions yet", "Run your first prediction to see it here.")
    else:
        for h in history:
            badge = "success" if h["outcome"] == "Pass" else "error"
            st.markdown(f"""
            <div class="sa-card" style="padding:14px 18px;">
                <span class="sa-badge {badge}">{h['outcome']}</span>
                &nbsp; {h['probability']}% probability
                <div style="color:var(--text-muted); font-size:0.8rem; margin-top:4px;">{h['created_at']}</div>
            </div>
            """, unsafe_allow_html=True)


# ==================================================
# PREDICT
# ==================================================
def render_predict():
    st.markdown("### 🔮 Predict a Student's Outcome")
    st.caption("Enter a student profile and get an instant Pass/Fail prediction.")

    c1, c2 = st.columns(2)
    with c1:
        study_hours = st.slider("Study hours per day", 0.0, 10.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 40, 100, 80)
        previous_score = st.slider("Previous exam score", 20, 100, 65)
    with c2:
        sleep_hours = st.slider("Sleep hours per night", 3.0, 10.0, 6.5, 0.5)
        extracurricular = st.selectbox("Extracurricular activities?", ["No", "Yes"])
        part_time_job = st.selectbox("Has a part-time job?", ["No", "Yes"])

    if st.button("🚀 Predict Outcome", type="primary"):
        profile = {
            "study_hours_per_day": study_hours,
            "attendance_percent": attendance,
            "previous_exam_score": previous_score,
            "sleep_hours": sleep_hours,
            "extracurricular_activities": 1 if extracurricular == "Yes" else 0,
            "part_time_job": 1 if part_time_job == "Yes" else 0,
        }
        pred, prob = ml.predict(bundle, profile)
        outcome = "Pass" if pred == 1 else "Fail"
        storage.add_prediction(profile, outcome, prob)

        badge = "success" if outcome == "Pass" else "error"
        icon = "✅" if outcome == "Pass" else "❌"
        st.markdown(f"""
        <div class="sa-hero">
            <h1>{icon} Predicted: {outcome}</h1>
            <p>Probability of passing: {prob*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob))

        with st.expander("💡 What influences this prediction?"):
            st.markdown("""
            - **Study hours** and **attendance** are strong positive factors
            - A higher **previous exam score** tends to predict continued success
            - Adequate **sleep** supports better performance
            - A **part-time job** can reduce available study time
            - **Extracurricular activities** show a mild positive association
            """)


# ==================================================
# BATCH PREDICT
# ==================================================
def render_batch():
    st.markdown("### 📄 Batch Predict from CSV")
    st.caption("Upload a CSV with columns: " + ", ".join(ml.FEATURES))

    template = pd.DataFrame([{
        "study_hours_per_day": 5.0, "attendance_percent": 85, "previous_exam_score": 70,
        "sleep_hours": 7, "extracurricular_activities": 1, "part_time_job": 0,
    }])
    st.download_button("⬇️ Download CSV Template", template.to_csv(index=False), file_name="student_template.csv")

    uploaded = st.file_uploader("Upload student data (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            missing = [f for f in ml.FEATURES if f not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                results_df = ml.predict_batch(bundle, df)
                st.success(f"✅ Predicted outcomes for {len(results_df)} students.")
                st.dataframe(results_df, use_container_width=True)
                st.download_button("⬇️ Download Results", results_df.to_csv(index=False),
                                    file_name="prediction_results.csv", type="primary")
        except Exception as e:
            st.error(f"Couldn't process file: {e}")


# ==================================================
# MODEL INSIGHTS
# ==================================================
def render_insights():
    st.markdown("### 📊 Model Insights")

    r = bundle["results"][bundle["model_name"]]
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Best Model", bundle["model_name"])
    with c2:
        card("Accuracy", f"{r['accuracy']*100:.1f}%")
    with c3:
        card("F1 Score", f"{r['f1']:.2f}")

    st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Model Comparison")
    comp_df = pd.DataFrame({
        name: {"Accuracy": res["accuracy"], "F1 Score": res["f1"]}
        for name, res in bundle["results"].items()
    }).T
    st.bar_chart(comp_df)

    st.markdown("#### Feature Importance")
    importance = ml.get_feature_importance(bundle)
    if importance is not None:
        st.bar_chart(importance)
    else:
        st.info("Feature importance not available for this model type.")

    st.markdown("#### Classification Report")
    st.code(ml.get_classification_report(bundle))

    with st.expander("📁 View Training Dataset"):
        st.dataframe(bundle["df"], use_container_width=True)


# ==================================================
# HISTORY
# ==================================================
def render_history():
    st.markdown("### 🕓 Prediction History")

    history = storage.get_history()
    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("🗑️ Clear History", use_container_width=True):
            storage.clear_history()
            st.rerun()

    if not history:
        empty_state("🕓", "No predictions yet", "Run a prediction to see it appear here.")
        return

    for h in history:
        badge = "success" if h["outcome"] == "Pass" else "error"
        p = h["profile"]
        st.markdown(f"""
        <div class="sa-card">
            <span class="sa-badge {badge}">{h['outcome']}</span> &nbsp; {h['probability']}% probability
            <span style="color:var(--text-muted); font-size:0.8rem;"> · {h['created_at']}</span>
            <div style="margin-top:8px; font-size:0.85rem; color:var(--text-muted);">
                Study: {p['study_hours_per_day']}h/day · Attendance: {p['attendance_percent']}% ·
                Previous score: {p['previous_exam_score']} · Sleep: {p['sleep_hours']}h
            </div>
        </div>
        """, unsafe_allow_html=True)


PAGES = {
    "Dashboard": render_dashboard,
    "Predict": render_predict,
    "Batch Predict": render_batch,
    "Model Insights": render_insights,
    "History": render_history,
}
PAGES[st.session_state.page]()
