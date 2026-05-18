import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LAG = 4


def load_ees(normalize: bool = True):
    path = os.path.join(DATA_DIR, "EES_data.csv")
    df = pd.read_csv(path, sep=";", decimal=",")
    X = df.values.astype(np.float64)
    scaler = {"min": X.min(axis=0), "max": X.max(axis=0)}
    if normalize:
        X = (X - scaler["min"]) / (scaler["max"] - scaler["min"] + 1e-9)
    return X, scaler, list(df.columns)


def load_nems(normalize: bool = False):
    path = os.path.join(DATA_DIR, "NEMS_data.csv")
    df = pd.read_csv(path, sep=";", decimal=",")
    X = df.values.astype(np.float64)
    scaler = {"min": X.min(axis=0), "max": X.max(axis=0)}
    if normalize:
        X = (X - scaler["min"]) / (scaler["max"] - scaler["min"] + 1e-9)
    return X, scaler, list(df.columns)


def split_train_test(X: np.ndarray, test_ratio: float = 0.2):
    n_train = int(len(X) * (1 - test_ratio))
    return X[:n_train], X[n_train:]


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1e-3
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.any() else np.nan)
    r2 = r2_score(y_true, y_pred)
    rng = y_true.max() - y_true.min()
    accuracy = max(0.0, (1.0 - mae / (rng + 1e-9)) * 100) if rng > 0 else 0.0
    return {
        "Модель":     name,
        "MAE":        round(mae, 5),
        "RMSE":       round(rmse, 5),
        "MAPE,%":     round(mape, 3),
        "R²":         round(r2, 4),
        "Точность,%": round(accuracy, 2),
    }


def print_results_table(results: list, title: str = "Сводная таблица"):
    df = pd.DataFrame(results)
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    return df
