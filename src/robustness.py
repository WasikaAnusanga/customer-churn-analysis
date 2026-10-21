# ==========================================
# Customer Churn Prediction
# Feature Removal Transformer
# ==========================================

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(
    BaseEstimator,
    TransformerMixin
):
    """
    Remove selected features from a DataFrame.

    Compatible with Scikit-learn pipelines.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):

        if self.columns is None:
            raise ValueError(
                "Columns to remove must be specified."
            )

        missing = set(self.columns) - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expected a pandas DataFrame."
            )

        return X.drop(
            columns=self.columns
        ).copy()