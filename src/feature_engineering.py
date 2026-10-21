import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Reusable feature engineering transformer
    for customer churn prediction.

    All transformations use only existing
    customer attributes, without using churn labels.

    Compatible with scikit-learn Pipelines.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        required_columns = {
            "tenure_months",
            "late_payments_last_12m",
            "complaints_last_12m",
            "monthly_charges",
            "monthly_usage_gb",
            "contract_type",
            "satisfaction_score"
        }

        missing_columns = required_columns - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        df = X.copy()

        # ====================================
        # Feature 1: Customer Tenure in Years
        # ====================================

        df["tenure_years"] = (
            df["tenure_months"] / 12
        )

        # ====================================
        # Feature 2: Payment Risk Indicator
        # ====================================

        df["payment_risk_flag"] = (
            df["late_payments_last_12m"] >= 2
        ).astype(int)

        # ====================================
        # Feature 3: Complaint Intensity
        # ====================================

        observation_months = df[
            "tenure_months"
        ].clip(lower=1, upper=12)

        df["complaint_intensity"] = (
            df["complaints_last_12m"]
            / observation_months
        )

        # ====================================
        # Feature 4: Total Service Issues
        # ====================================

        df["total_service_issues"] = (
            df["complaints_last_12m"]
            + df["late_payments_last_12m"]
        )

        # ====================================
        # Feature 5: Charge-to-Usage Ratio
        # ====================================

        df["monthly_charge_per_gb"] = (
            df["monthly_charges"]
            / (df["monthly_usage_gb"] + 1)
        )

        # ====================================
        # Feature 6: Contract-Payment Interaction
        # ====================================

        df["month_to_month_late_payments"] = (
            np.where(
                df["contract_type"] == "Month-to-month",
                df["late_payments_last_12m"],
                0
            )
        )

        # ====================================
        # Feature 7: Low Satisfaction Indicator
        # ====================================

        df["low_satisfaction_flag"] = np.where(
            df["satisfaction_score"].isna(),
            np.nan,
            (
                df["satisfaction_score"] <= 4
            ).astype(int)
        )

        # ====================================
        # Feature 8: Charge-Satisfaction Interaction
        # ====================================

        df["charge_satisfaction_interaction"] = (
            df["monthly_charges"]
            * (10 - df["satisfaction_score"])
            / 10
        )

        return df