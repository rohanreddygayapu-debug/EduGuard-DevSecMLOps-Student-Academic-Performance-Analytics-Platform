"""
drift_detector.py
MLOps Continuous Monitoring:
Detects covariate shift / data drift across incoming student cohorts
comparing mean & standard deviation against baseline statistics.
"""

import numpy as np
import pandas as pd


class CohortDriftDetector:
    def __init__(self, baseline_df: pd.DataFrame = None):
        self.numeric_features = [
            "Attendance_Percentage",
            "Study_Hours_per_Week",
            "LMS_Hours",
            "Internal_Marks"
        ]
        self.baseline_stats = {}
        if baseline_df is not None:
            self.fit_baseline(baseline_df)

    def fit_baseline(self, df: pd.DataFrame):
        for col in self.numeric_features:
            if col in df.columns:
                self.baseline_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

    def evaluate_cohort_drift(self, current_df: pd.DataFrame, threshold: float = 0.25) -> dict:
        """
        Calculates normalized feature mean deviation from baseline:
        |mean_current - mean_baseline| / std_baseline
        """
        results = {}
        drift_detected = False

        for col in self.numeric_features:
            if col in current_df.columns and col in self.baseline_stats:
                base_m = self.baseline_stats[col]["mean"]
                base_s = max(self.baseline_stats[col]["std"], 1e-5)
                curr_m = float(current_df[col].mean())

                z_shift = abs(curr_m - base_m) / base_s
                is_drift = z_shift >= threshold
                if is_drift:
                    drift_detected = True

                results[col] = {
                    "baseline_mean": round(base_m, 2),
                    "current_mean": round(curr_m, 2),
                    "normalized_shift": round(z_shift, 4),
                    "drift_flag": is_drift
                }

        return {
            "overall_drift_detected": drift_detected,
            "threshold": threshold,
            "feature_metrics": results
        }
