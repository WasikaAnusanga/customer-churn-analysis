# ==========================================
# Customer Churn Prediction
# Machine Learning Model Pipeline
# ==========================================

from sklearn.pipeline import Pipeline

from src.feature_engineering import ChurnFeatureEngineer
from src.preprocessing import build_preprocessor


def build_model_pipeline(model):
    """
    Build a complete machine learning pipeline.

    Steps:
    1. Feature engineering
    2. Data preprocessing
    3. Machine learning model

    This pipeline ensures preprocessing is
    performed within cross-validation folds.
    """

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                ChurnFeatureEngineer()
            ),
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline