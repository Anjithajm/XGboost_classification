import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_raw_data():
    """Simulates loading raw data from a database or CSV."""
    return pd.DataFrame({
        'age': [25, 47, np.nan, 38, 22, 55, 33, np.nan],
        'income': [50000, 120000, 85000, np.nan, 42000, 150000, np.nan, 62000],
        'education': ['Bachelor', 'PhD', 'Master', 'Bachelor', 'High School', 'PhD', 'Master', 'Bachelor'],
        'city': ['NY', 'SF', 'NY', 'LA', 'Chicago', 'SF', 'LA', 'Chicago'],
        'churned': [0, 1, 0, 0, 1, 1, 0, 1]
    })

def build_preprocessor():
    """Creates and returns the Scikit-Learn ColumnTransformer object."""
    numeric_features = ['age', 'income']
    categorical_features = ['education', 'city']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor