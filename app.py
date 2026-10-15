"""
app.py
Student Academic Performance Analytics System
DevSecMLOps Interactive Demonstration & RBAC Audit Dashboard
"""

import sys
import os
import json
import streamlit as st
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import StudentAnalyticsService
from security_scanner import compute_file_sha256
from rbac_auth import RBACSecurityManager
from drift_detector import CohortDriftDetector

st.set_page_config(
    page_title="Student Analytics | DevSecMLOps",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Academic Performance Analytics Platform")
st.markdown(
    "**DevSecMLOps Architecture**: Logistic Regression + StandardScaler Pipeline • "
    "Role-Based Access Control (RBAC) • SHA-256 Integrity Verification • Cohort Drift Detection"
)

# Sidebar: RBAC Role & Security Status
st.sidebar.header("👤 User Authentication & RBAC")
active_role = st.sidebar.selectbox("Active User Role", ["faculty", "student", "admin", "auditor"], index=0)

model_path = os.path.join(CURRENT_DIR, "models", "student_model.joblib")
sig_path = model_path + ".sha256"

st.sidebar.divider()
st.sidebar.subheader("🔐 Cryptographic Model Status")
if os.path.exists(model_path) and os.path.exists(sig_path):
    with open(sig_path, "r") as f:
        stored_hash = f.read().strip()
    current_hash = compute_file_sha256(model_path)

    if stored_hash == current_hash:
        st.sidebar.success("✅ Model Cryptographic Signature Verified")
        st.sidebar.caption(f"**SHA-256 Digest:** `{current_hash[:16]}...`")
    else:
        st.sidebar.error("🚨 ALERT: Model Hash Mismatch! Potential Tampering.")
else:
    st.sidebar.warning("⚠️ Model or Signature file missing. Run `python src/train.py`.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Academic Risk Assessment",
    "📈 Cohort Data Drift Monitor",
    "📊 Model Performance & Card",
    "🛡️ RBAC & DevSecOps Controls"
])

with tab1:
    st.subheader("Student Academic Profile Evaluation")
    st.caption(f"Currently acting under role: **{active_role.upper()}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_id = st.text_input("Student Roll / ID", value="23WU0101001")
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        attendance = st.slider("Attendance Percentage (%)", min_value=10.0, max_value=100.0, value=78.5, step=0.5)
        study_hours = st.slider("Study Hours / Week", min_value=1.0, max_value=40.0, value=12.0, step=0.5)
    with col3:
        lms_hours = st.slider("LMS Platform Hours / Semester", min_value=5.0, max_value=150.0, value=50.0, step=1.0)
        internal_marks = st.slider("Internal Marks (out of 50)", min_value=5.0, max_value=50.0, value=32.0, step=0.5)

    run_pred = st.button("🚀 Evaluate Academic Risk", type="primary")

    if run_pred:
        try:
            service = StudentAnalyticsService()
            payload = {
                "student_id": student_id,
                "gender": gender,
                "attendance_percentage": attendance,
                "study_hours_per_week": study_hours,
                "lms_hours": lms_hours,
                "internal_marks": internal_marks
            }
            res = service.predict_student_risk(payload, caller_role=active_role)

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Academic At-Risk Probability", f"{res['risk_probability'] * 100:.1f}%")
            with c2:
                st.info(f"Status: **{res['risk_badge']}**")
            with c3:
                st.success(f"Caller Role: **{res['caller_role'].upper()}** (Authorized)")
            with c4:
                st.success("Integrity: SHA-256 Verified")

            st.markdown("### Recommended Academic Action")
            if res["risk_probability"] >= 0.5:
                st.warning(f"⚠️ **Action Required:** {res['recommended_intervention']}")
            else:
                st.success(f"✅ **On Track:** {res['recommended_intervention']}")

        except PermissionError as pe:
            st.error(f"⛔ {str(pe)}")
        except Exception as e:
            st.error(f"Evaluation Error: {str(e)}")

with tab2:
    st.subheader("MLOps Continuous Monitoring: Cohort Data Drift Detection")
    st.markdown("Simulate a semester shift to test whether student behavior has drifted from the training baseline.")

    baseline_path = os.path.join(CURRENT_DIR, "models", "baseline_stats.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            base_stats = json.load(f)

        drift_sim_col1, drift_sim_col2 = st.columns(2)
        with drift_sim_col1:
            sim_attendance = st.slider("Simulated Cohort Mean Attendance (%)", 40.0, 100.0, 78.0)
            sim_study = st.slider("Simulated Cohort Mean Study Hours", 1.0, 30.0, 14.0)
        with drift_sim_col2:
            sim_lms = st.slider("Simulated Cohort Mean LMS Hours", 10.0, 120.0, 55.0)
            sim_internals = st.slider("Simulated Cohort Mean Internal Marks", 10.0, 50.0, 30.0)

        # Mock current cohort
        curr_df = pd.DataFrame({
            "Attendance_Percentage": [sim_attendance],
            "Study_Hours_per_Week": [sim_study],
            "LMS_Hours": [sim_lms],
            "Internal_Marks": [sim_internals]
        })

        detector = CohortDriftDetector()
        detector.baseline_stats = base_stats
        drift_report = detector.evaluate_cohort_drift(curr_df, threshold=0.30)

        st.divider()
        if drift_report["overall_drift_detected"]:
            st.error("🚨 MLOps Alert: Statistical Covariate Shift / Data Drift Detected in Student Cohort! Retraining Recommended.")
        else:
            st.success("🟢 MLOps Health Check: Cohort distribution is stable. No drift detected.")

        drift_table = []
        for feature, vals in drift_report["feature_metrics"].items():
            drift_table.append({
                "Feature": feature,
                "Baseline Mean": vals["baseline_mean"],
                "Cohort Mean": vals["current_mean"],
                "Normalized Shift": vals["normalized_shift"],
                "Drift Triggered": "🔴 YES" if vals["drift_flag"] else "🟢 NO"
            })
        st.dataframe(pd.DataFrame(drift_table), use_container_width=True)

with tab3:
    st.subheader("Model Performance & MLOps Card")
    metrics_path = os.path.join(CURRENT_DIR, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
        m2.metric("Precision", f"{metrics.get('precision', 0.0) * 100:.1f}%")
        m3.metric("Recall", f"{metrics.get('recall', 0.0) * 100:.1f}%")
        m4.metric("F1-Score", f"{metrics.get('f1_score', 0.0) * 100:.1f}%")
        m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")

        st.json(metrics)

with tab4:
    st.subheader("DevSecOps Security Controls & RBAC Matrix")
    st.markdown("""
    ### Role-Based Access Control (RBAC) Enforcement Policy
    | Role | Single Prediction | Batch Analytics | Retrain Model | View Drift Reports | View Audit Trail |
    |---|:---:|:---:|:---:|:---:|:---:|
    | **Admin** | ✅ | ✅ | ✅ | ✅ | ✅ |
    | **Faculty** | ✅ | ✅ | ❌ | ✅ | ❌ |
    | **Student** | ✅ *(Self)* | ❌ | ❌ | ❌ | ❌ |
    | **Auditor** | ❌ | ❌ | ❌ | ✅ | ✅ |
    """)

    st.markdown("### Interactive RBAC Action Test")
    test_action = st.selectbox("Action to Test", ["retrain_model", "predict_batch", "view_audit_logs"])
    if st.button("Test Permission"):
        if RBACSecurityManager.is_authorized(active_role, test_action):
            st.success(f"Access Granted: Role `{active_role}` can execute `{test_action}`.")
        else:
            st.error(f"Access Denied: Role `{active_role}` is blocked from `{test_action}`.")
