import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_nems, split_train_test, evaluate, LAG
from nems.lstm import LSTM_NEMS
from nems.anfis import ANFIS_NEMS
from nems.arima import ARIMA_NEMS


def rolling_multistep_nn(model, X_train, X_test, horizon, n_windows=20):
    full = np.vstack([X_train, X_test])
    start = len(X_train)
    all_preds, all_true = [], []
    available = len(X_test) - horizon
    step = max(1, available // n_windows)
    for i in range(0, available, step):
        if len(all_preds) >= n_windows:
            break
        history = full[start + i - model.lag - 1:start + i]
        preds = model.predict(history, horizon)
        truth = full[start + i:start + i + horizon]
        all_preds.append(preds)
        all_true.append(truth)
    return np.vstack(all_preds), np.vstack(all_true)


def static_multistep_arima(arima_models, X_test, horizon, n_windows=20):
    all_preds, all_true = [], []
    available = len(X_test) - horizon
    step = max(1, available // n_windows)
    cursor = 0
    while cursor + horizon <= len(X_test) and len(all_preds) < n_windows:
        preds = np.zeros((horizon, X_test.shape[1]))
        for j in range(X_test.shape[1]):
            fc = arima_models[j].forecast(steps=horizon)
            preds[:, j] = np.clip(np.asarray(fc), 0.0, 1.0)
        all_preds.append(preds)
        all_true.append(X_test[cursor:cursor + horizon])
        cursor = min(cursor + step, len(X_test))
    return np.vstack(all_preds), np.vstack(all_true)


def main():
    print("=" * 70)
    print("  СРАВНЕНИЕ MULTI-STEP ПРОГНОЗА — НЭМС")
    print("=" * 70)

    X, scaler, cols = load_nems(normalize=False)
    print(f"Выборка: {X.shape}, концепты: {cols}")

    X_train_full, X_test = split_train_test(X, test_ratio=0.2)
    X_train = X_train_full[-1500:]
    print(f"Train: {len(X_train)}")
    print(f"Test: {len(X_test)}\n")

    print("[1/3] Обучение LSTM ...")
    lstm = LSTM_NEMS(lag=LAG, n_concepts=6).fit(X_train)

    print("\n[2/3] Обучение ANFIS ...")
    anfis = ANFIS_NEMS(lag=LAG, n_concepts=6).fit(X_train)

    print("\n[3/3] Обучение ARIMA ...")
    arima = ARIMA_NEMS().fit(X_train)

    horizons = [1, 5, 10, 15, 20]
    n_windows = 20
    rows = []
    for H in horizons:
        p_lstm, truth = rolling_multistep_nn(lstm, X_train, X_test, H, n_windows)
        p_anfis, _    = rolling_multistep_nn(anfis, X_train, X_test, H, n_windows)
        p_arima, _    = static_multistep_arima(arima.models, X_test, H, n_windows)

        r_lstm  = evaluate("LSTM",  truth, p_lstm)
        r_anfis = evaluate("ANFIS", truth, p_anfis)
        r_arima = evaluate("ARIMA", truth, p_arima)

        rows.append({
            "Горизонт": H,
            "LSTM, %":  r_lstm["Точность,%"],
            "ANFIS, %": r_anfis["Точность,%"],
            "ARIMA, %": r_arima["Точность,%"],
        })

    df = pd.DataFrame(rows)
    print()
    print("=" * 70)
    print("  ТОЧНОСТЬ ПО ГОРИЗОНТАМ ПРОГНОЗА — НЭМС")
    print("=" * 70)
    print(df.to_string(index=False))
    print()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_nems")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "nems_multistep.csv"), index=False, sep=";")
    print(f"Результаты сохранены: {out_dir}/nems_multistep.csv")


if __name__ == "__main__":
    main()
