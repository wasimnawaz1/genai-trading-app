import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from typing import Tuple
from pathlib import Path

def build_sequences_multi(features: pd.DataFrame, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    arr = features.values
    idx = features.index
    n = len(features)
    X, y, y_idx = [], [], []
    for i in range(n - seq_len - horizon + 1):
        seq = arr[i:i+seq_len]
        price_t = features['Adj Close'].iloc[i + seq_len - 1]
        price_t_h = features['Adj Close'].iloc[i + seq_len - 1 + horizon]
        cum_ret = (price_t_h / price_t) - 1.0
        X.append(seq)
        y.append(cum_ret)
        y_idx.append(idx[i + seq_len - 1 + horizon])
    return np.array(X), np.array(y), pd.DatetimeIndex(y_idx)

def scale_sequences(X_train: np.ndarray, X_test: np.ndarray):
    ns, seq_len, nfeat = X_train.shape
    reshaped = X_train.reshape(-1, nfeat)
    scaler = MinMaxScaler()
    scaler.fit(reshaped)
    X_train_s = scaler.transform(reshaped).reshape(ns, seq_len, nfeat)
    X_test_s = scaler.transform(X_test.reshape(-1, nfeat)).reshape(X_test.shape)
    X_train_flat = X_train_s.reshape(X_train_s.shape[0], -1)
    X_test_flat = X_test_s.reshape(X_test_s.shape[0], -1)
    return X_train_flat, X_test_flat, scaler

def train_sklearn_baseline(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model_if_exists(path):
    p = Path(path)
    if p.exists():
        return joblib.load(path)
    return None

def time_based_train_test_splits(X, y, n_splits=5):
    n = len(X)
    splits = []
    step = n // (n_splits + 1)
    for i in range(1, n_splits+1):
        end_train = step * i
        start_test = end_train
        end_test = min(end_train + step, n)
        splits.append((slice(0, end_train), slice(start_test, end_test)))
    return splits
