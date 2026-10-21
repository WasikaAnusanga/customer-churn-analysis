# ==========================================
# Customer Churn Prediction
# Preprocessing Pipeline
# ==========================================

import numpy as np

from sklearn.compose import (
    ColumnTransformer,
    make_column_selector
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from src.feature_engineering import ChurnFeatureEngineer


def build_preprocessor():
    """
    Build preprocessing steps for
    numerical and categorical features.
    """

    # ======================================
    # Numerical Feature Pipeline
    # ======================================

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ======================================
    # Categorical Feature Pipeline
    # ======================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # ======================================
    # Combine Preprocessing Steps
    # ======================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                make_column_selector(
                    dtype_include=np.number
                )
            ),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(
                    dtype_exclude=np.number
                )
            )
        ],
        remainder="drop",
        sparse_threshold=0
    )

    return preprocessor


def build_preprocessing_pipeline():
    """
    Create the complete preprocessing
    pipeline, including feature engineering.
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
            )
        ]
    )

    return pipeline