import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
        self.column_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        selected_type = X.select_dtypes(include=[self.dtype])
        self.column_names = selected_type.columns
        return selected_type

    def get_feature_names(self):
        return self.column_names


def clean_ohe_cols(ohe_cols, cat_cols):
    for i, col in enumerate(cat_cols):
        ohe_cols = [_.replace(f"x{i}", f"{col}") for _ in ohe_cols]
    return ohe_cols
