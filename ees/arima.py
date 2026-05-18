import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class ARIMA_EES:

    NAME = "ARIMA"
    LAG = 1
    N_CONCEPTS = 4
    N_INPUTS = (LAG + 0)
    N_OUTPUTS = N_CONCEPTS
    HYPERPARAMS = {
        "framework":       "StatsModels",
        "lag":             1,
        "n_concepts":      4,
        "order_p":         1,
        "order_d":         0,
        "order_q":         0,
        "n_inputs_per_model": 1,
        "n_inputs_total":  4,
        "n_outputs":       4,
        "training_method": "MLE (Maximum Likelihood Estimation)",
        "ensemble_type":   "независимые одномерные модели",
        "n_models":        4,
        "forecast_mode":   "rolling one-step-ahead",
    }

    def __init__(self, lag: int = 1, order_d: int = 0, order_q: int = 0):
        self.lag = lag
        self.order = (lag, order_d, order_q)
        self.models = []

    def fit(self, X: np.ndarray):
        self.models = []
        n_components = X.shape[1]
        for j in range(n_components):
            m = ARIMA(X[:, j], order=self.order).fit(
                method_kwargs={"warn_convergence": False}
            )
            self.models.append(m)
            print(f"    [ARIMA] C{j+1}: AIC = {m.aic:.2f}")
        params_per_model = sum(self.order) + 2
        print(f"    [ARIMA] параметров на модель: {params_per_model}, "
              f"всего: {params_per_model * n_components}")
        return self

    def predict(self, X_history: np.ndarray, n_steps: int) -> np.ndarray:
        n_outputs = len(self.models)
        preds = np.zeros((n_steps, n_outputs))
        for j, m in enumerate(self.models):
            fc = m.forecast(steps=n_steps)
            preds[:, j] = np.clip(np.asarray(fc), 0.0, 1.0)
        return preds

    def rolling_forecast(self, X_test: np.ndarray) -> np.ndarray:
        n_test, n_outputs = X_test.shape
        preds = np.zeros((n_test, n_outputs))
        for j in range(n_outputs):
            model = self.models[j]
            for i in range(n_test):
                fc = model.forecast(steps=1)
                preds[i, j] = np.clip(fc.iloc[0] if hasattr(fc, "iloc") else fc[0],
                                      0.0, 1.0)
                model = model.append([X_test[i, j]], refit=False)
        return preds


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import load_ees, split_train_test, evaluate, print_results_table

    print("=" * 70)
    print(f"  ARIMA(1,0,0) (StatsModels) — ЭЭС")
    print("=" * 70)

    X, scaler, cols = load_ees(normalize=True)
    print(f"Выборка: {X.shape}, концепты: {cols}")

    X_train, X_test = split_train_test(X, test_ratio=0.2)
    X_train = X_train[-1500:]
    X_test = X_test[:200]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = ARIMA_EES(lag=1, order_d=0, order_q=0)
    model.fit(X_train)

    preds = model.rolling_forecast(X_test)
    result = evaluate("ARIMA", X_test, preds)
    print_results_table([result], "Результат ARIMA — ЭЭС")
