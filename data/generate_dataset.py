"""
generate_dataset.py
Generates the benchmark student academic performance dataset (3,000 students)
matching the exact reference columns:
Student_ID, Gender, Attendance_Percentage, Study_Hours_per_Week, LMS_Hours, Internal_Marks, Final_Grade
"""

import os
import random
import numpy as np
import pandas as pd


def generate_student_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    random.seed(random_state)

    # 1. Student IDs (WU2026-0001 ...)
    student_ids = [f"WU2026-{i:04d}" for i in range(1, n_samples + 1)]

    # 2. Gender (Male, Female)
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])

    # 3. Attendance Percentage (typically between 50% and 100%)
    attendance = np.clip(np.random.normal(loc=78.0, scale=12.0, size=n_samples), 40.0, 100.0)

    # 4. Study Hours per Week (typically between 2 and 35 hours)
    study_hours = np.clip(np.random.normal(loc=14.0, scale=6.0, size=n_samples), 1.0, 40.0)

    # 5. LMS Hours (portal activity hours per semester, 10 to 120 hours)
    lms_hours = np.clip(np.random.normal(loc=55.0, scale=20.0, size=n_samples), 5.0, 150.0)

    # 6. Internal Marks (out of 50, correlated with attendance and study hours)
    latent_ability = (
        0.35 * (attendance / 100.0) +
        0.40 * (study_hours / 40.0) +
        0.25 * (lms_hours / 150.0) +
        np.random.normal(0, 0.08, size=n_samples)
    )
    internal_marks = np.clip(latent_ability * 50.0, 10.0, 50.0)

    # 7. Final Grade (Categorical classification or Binary Pass/Fail & Grade Level):
    # Grade criteria:
    # Score composite out of 100: internal_marks (50) + final exam component (50)
    final_score = internal_marks + np.clip(latent_ability * 50.0 + np.random.normal(0, 4, size=n_samples), 5.0, 50.0)

    final_grades = []
    for s in final_score:
        if s >= 85:
            final_grades.append("A")
        elif s >= 70:
            final_grades.append("B")
        elif s >= 55:
            final_grades.append("C")
        elif s >= 40:
            final_grades.append("D")
        else:
            final_grades.append("F")

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": genders,
        "Attendance_Percentage": np.round(attendance, 2),
        "Study_Hours_per_Week": np.round(study_hours, 2),
        "LMS_Hours": np.round(lms_hours, 2),
        "Internal_Marks": np.round(internal_marks, 2),
        "Final_Grade": final_grades
    })

    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(out_dir, "student_academic_performance_dataset.csv")
    df = generate_student_data(n_samples=3000, random_state=42)
    df.to_csv(data_path, index=False)
    print(f"Dataset generated successfully at: {data_path}")
    print(f"Shape: {df.shape}")
    print("Grade Distribution:\n", df["Final_Grade"].value_counts())
