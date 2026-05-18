import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)


class LSTMNet(nn.Module):

    def __init__(self, n_features: int = 6, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTM_NEMS:

    NAME = "LSTM"
    LAG = 4
    N_CONCEPTS = 6
    N_INPUTS = LAG * N_CONCEPTS
    N_OUTPUTS = N_CONCEPTS
    HYPERPARAMS = {
        "framework":      "PyTorch",
        "lag":            4,
        "n_concepts":     6,
        "n_inputs":       24,
        "n_outputs":      6,
        "hidden_size":    64,
        "num_layers":     2,
        "dropout":        0.1,
        "target":         "first-order differences (delta x)",
        "loss":           "MSELoss",
        "optimizer":      "Adam",
        "learning_rate":  0.001,
        "epochs":         60,
        "batch_size":     64,
    }

    def __init__(self, lag: int = 4, n_concepts: int = 6,
                 hidden_size: int = 64, num_layers: int = 2,
                 lr: float = 0.001, epochs: int = 60, batch_size: int = 64,
                 device: str = "cpu"):
        self.lag = lag
        self.n_concepts = n_concepts
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.net = LSTMNet(n_concepts, hidden_size, num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def _make_diff_sequences(self, X: np.ndarray):
        dX = np.diff(X, axis=0)
        Xs, Ys = [], []
        for t in range(len(dX) - self.lag):
            Xs.append(dX[t:t + self.lag])
            Ys.append(dX[t + self.lag])
        Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
        Ys = torch.tensor(np.array(Ys), dtype=torch.float32)
        return Xs, Ys

    def fit(self, X: np.ndarray):
        Xs, Ys = self._make_diff_sequences(X)
        loader = DataLoader(TensorDataset(Xs, Ys),
                            batch_size=self.batch_size, shuffle=True)
        print(f"    [LSTM] параметров: {self.n_params:,}")
        print(f"    [LSTM] обучающих последовательностей: {len(Xs)}")

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.net(xb)
                loss = self.criterion(y_pred, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(xb)
            if epoch % 10 == 0 or epoch == 1:
                print(f"    [LSTM] epoch {epoch:3d}/{self.epochs}  "
                      f"loss = {total_loss / len(Xs):.6f}")
        return self

    @torch.no_grad()
    def predict(self, X_history: np.ndarray, n_steps: int) -> np.ndarray:
        self.net.eval()
        dX_hist = list(np.diff(X_history, axis=0))
        last_abs = X_history[-1].copy()
        preds = []
        for _ in range(n_steps):
            seq = np.array(dX_hist[-self.lag:])
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            dy = self.net(x).cpu().numpy()[0]
            last_abs = np.clip(last_abs + dy, 0.0, 1.0)
            preds.append(last_abs.copy())
            dX_hist.append(dy)
        return np.array(preds)

    def predict_step(self, history: np.ndarray) -> np.ndarray:
        return self.predict(history, 1)[0]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import load_nems, split_train_test, evaluate, print_results_table, LAG

    print("=" * 70)
    print(f"  LSTM (PyTorch, обучение на разностях) — НЭМС (LAG={LAG})")
    print("=" * 70)

    X, scaler, cols = load_nems(normalize=False)
    print(f"Выборка: {X.shape}, концепты: {cols}")

    X_train, X_test = split_train_test(X, test_ratio=0.2)
    X_train = X_train[-1500:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = LSTM_NEMS(lag=LAG, n_concepts=6)
    model.fit(X_train)

    full = np.vstack([X_train, X_test])
    preds = np.array([
        model.predict_step(full[len(X_train) + i - model.lag - 1:len(X_train) + i])
        for i in range(len(X_test))
    ])
    result = evaluate("LSTM", X_test, preds)
    print_results_table([result], "Результат LSTM — НЭМС")
