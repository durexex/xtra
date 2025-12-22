import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(0, np.nan)
        return X_copy

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not self.columns:
            return X_copy
            
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = np.log1p(X_copy[col])
        return X_copy