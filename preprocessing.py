# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(csv_path, label_col):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    
    X_raw = df.drop(columns=[label_col])
    y_raw = df[label_col].astype(str)
    
    return X_raw, y_raw

def build_preprocessor(X_raw):
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols

def encode_labels(y_raw):
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le
