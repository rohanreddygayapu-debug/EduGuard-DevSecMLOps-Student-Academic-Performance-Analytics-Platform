"""
test_pipeline.py
Unit tests for data loader, feature engineering, and inference accuracy.
"""

import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_student_data, preprocess_student_data, get_train_test_splits
from predict import StudentAnalyticsService


def test_student_data_loading():
    df = load_student_data()
    assert df is not None
    assert len(df) == 3000
    assert "Student_ID" in df.columns
    assert "Attendance_Percentage" in df.columns
    assert "Final_Grade" in df.columns


def test_preprocessing_shapes():
    df = load_student_data()
    X, y = preprocess_student_data(df)
    assert len(X) == len(y)
    assert "Gender_Code" in X.columns
    assert "Attendance_Percentage" in X.columns
    assert "Internal_Marks" in X.columns


def test_student_inference():
    service = StudentAnalyticsService()
    student = {
        "student_id": "WU2026-9999",
        "gender": "Male",
        "attendance_percentage": 92.0,
        "study_hours_per_week": 25.0,
        "lms_hours": 90.0,
        "internal_marks": 45.0
    }
    res = service.predict_student_risk(student, caller_role="faculty")
    assert "risk_probability" in res
    assert "academic_status" in res
    assert res["integrity_verified"] is True
    assert 0.0 <= res["risk_probability"] <= 1.0
    # Strong student should be safe
    assert res["risk_probability"] < 0.40


if __name__ == "__main__":
    pytest.main(["-v", __file__])
