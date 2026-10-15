"""
predict.py
Secure Student Academic Performance Inference Engine:
- Role-Based Access Control (RBAC) Enforcement
- SHA-256 Model Integrity Verification
- Student Academic Risk Scoring & Intervention Recommendations
"""

import sys
import os
import pandas as pd
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from security_scanner import safe_load_model
from rbac_auth import RBACSecurityManager


class StudentAnalyticsService:
    def __init__(self, model_path: str = None):
        base_dir = os.path.dirname(CURRENT_DIR)
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "student_model.joblib")
        self.model_path = model_path
        self.pipeline = safe_load_model(self.model_path)
        self.audit_log = []

    def predict_student_risk(self, student_payload: dict, caller_role: str = "faculty") -> dict:
        """
        Runs RBAC-governed inference.
        """
        # 1. DevSec: Enforce Role-Based Access Control
        RBACSecurityManager.enforce_permission(caller_role, "predict_single")

        # 2. Input sanitization
        gender_code = 1 if str(student_payload.get("gender", "male")).lower() == "male" else 0
        attendance = float(student_payload.get("attendance_percentage", 75.0))
        study_hours = float(student_payload.get("study_hours_per_week", 10.0))
        lms_hours = float(student_payload.get("lms_hours", 40.0))
        internal_marks = float(student_payload.get("internal_marks", 30.0))

        if not (0.0 <= attendance <= 100.0):
            raise ValueError(f"Attendance {attendance}% out of valid range [0, 100].")
        if not (0.0 <= internal_marks <= 50.0):
            raise ValueError(f"Internal marks {internal_marks} out of valid range [0, 50].")

        X_input = pd.DataFrame([{
            "Gender_Code": gender_code,
            "Attendance_Percentage": attendance,
            "Study_Hours_per_Week": study_hours,
            "LMS_Hours": lms_hours,
            "Internal_Marks": internal_marks
        }])

        # 3. Model Inference
        risk_class = int(self.pipeline.predict(X_input)[0])  # 1 = At-Risk, 0 = On-Track
        risk_proba = float(self.pipeline.predict_proba(X_input)[0][1])

        if risk_proba >= 0.70:
            academic_status = "CRITICAL AT-RISK"
            recommended_intervention = "Mandatory Academic Counseling & Remedial Coaching"
            badge = "🔴 High Risk"
        elif risk_proba >= 0.40:
            academic_status = "MODERATE RISK"
            recommended_intervention = "Peer Tutoring & LMS Engagement Monitoring"
            badge = "🟡 Moderate Risk"
        else:
            academic_status = "ON TRACK / EXCELLENT"
            recommended_intervention = "Standard Academic Progress / Advanced Projects"
            badge = "🟢 Safe / On Track"

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "caller_role": caller_role,
            "student_id": student_payload.get("student_id", "ANONYMOUS"),
            "risk_probability": round(risk_proba, 4),
            "academic_status": academic_status,
            "risk_badge": badge,
            "recommended_intervention": recommended_intervention,
            "integrity_verified": True
        }

        self.audit_log.append(result)
        return result


if __name__ == "__main__":
    service = StudentAnalyticsService()
    sample_student = {
        "student_id": "WU2026-0105",
        "gender": "Female",
        "attendance_percentage": 52.0,
        "study_hours_per_week": 4.0,
        "lms_hours": 15.0,
        "internal_marks": 18.0
    }
    output = service.predict_student_risk(sample_student, caller_role="faculty")
    print("Inference Result for At-Risk Student:")
    print(output)
