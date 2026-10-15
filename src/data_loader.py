"""
data_loader.py
Loads, cleans, validates, and encodes student academic performance data.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_student_data(data_path: str = None) -> pd.DataFrame:
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "student_academic_performance_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Student dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df


def preprocess_student_data(df: pd.DataFrame):
    """
    Features:
    - Gender (encoded: 0 for Female, 1 for Male)
    - Attendance_Percentage
    - Study_Hours_per_Week
    - LMS_Hours
    - Internal_Marks
    Target:
    - Target: Academic Risk / Low Performance (1 if Grade is 'F' or 'D', 0 if 'A', 'B', 'C')
      or multi-class Final_Grade. Here we model Academic At-Risk Prediction (Binary Classification)
      or Grade Classification. Binary At-Risk target is standard for early academic intervention!
    """
    df = df.copy()

    # Encode Gender (Binary: Female=0, Male=1)
    df["Gender_Code"] = df["Gender"].apply(lambda g: 1 if str(g).lower() == "male" else 0)

    # At-Risk Target: 1 if Final_Grade in ['F', 'D'], 0 otherwise
    df["At_Risk"] = df["Final_Grade"].apply(lambda g: 1 if g in ["F", "D"] else 0)

    feature_cols = [
        "Gender_Code",
        "Attendance_Percentage",
        "Study_Hours_per_Week",
        "LMS_Hours",
        "Internal_Marks"
    ]

    X = df[feature_cols]
    y = df["At_Risk"]
    return X, y


def get_train_test_splits(test_size: float = 0.20, random_state: int = 42):
    df = load_student_data()
    X, y = preprocess_student_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
