import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)


def subtractive_clustering(X: np.ndarray, r_a: float = 0.1,
                           eps_upper: float = 0.5, eps_lower: float = 0.15,
                           max_clusters: int = 30,
                           sample_size: int = 2000):
    X = X.astype(np.float32)
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X[idx]

    r_b = 1.5 * r_a
    alpha = 4.0 / (r_a ** 2)
    beta = 4.0 / (r_b ** 2)

    D = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    P = np.sum(np.exp(-alpha * D), axis=1)
    P0 = P.max()

    centers = []
    while len(centers) < max_clusters:
        idx_max = int(P.argmax())
        P_star = P[idx_max]
        if P_star >= eps_upper * P0:
            accept = True
        elif P_star <= eps_lower * P0:
            break
        else:
            if len(centers) == 0:
                accept = True
            else:
                d_min = min(np.linalg.norm(X[idx_max] - c) for c in centers)
                accept = (d_min / r_a + P_star / P0) >= 1
                if not accept:
                    P[idx_max] = 0
                    continue
        if accept:
            centers.append(X[idx_max].copy())
            D_new = np.sum((X - X[idx_max]) ** 2, axis=1)
            P = P - P_star * np.exp(-beta * D_new)
            P = np.clip(P, 0, None)
    return np.array(centers)


class ANFISNet(nn.Module):

    def __init__(self, init_centers: np.ndarray, n_outputs: int,
                 init_width: float = 0.3):
        super().__init__()
        n_rules, n_in = init_centers.shape
        self.n_rules = n_rules
        self.n_in = n_in
        self.n_outputs = n_outputs

        self.centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))
        self.widths = nn.Parameter(
            torch.full((n_rules, n_in), init_width, dtype=torch.float32)
        )
        self.consequent_p = nn.Parameter(
            torch.randn(n_rules, n_outputs, n_in) * 0.1
        )
        self.consequent_q = nn.Parameter(torch.zeros(n_rules, n_outputs))

    def forward(self, x):
        x_exp = x.unsqueeze(1)
        c = self.centers.unsqueeze(0)
        s = torch.abs(self.widths).unsqueeze(0) + 1e-6
        sq = ((x_exp - c) / s) ** 2
        w = torch.exp(-0.5 * sq.sum(dim=2))
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        linear = torch.einsum('roi,bi->bro', self.consequent_p, x) + \
                 self.consequent_q.unsqueeze(0)
        weighted = linear * w_norm.unsqueeze(2)
        return weighted.sum(dim=1)


class ANFIS_NEMS:

    NAME = "ANFIS"
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
        "rule_init":      "Subtractive Clustering (Chiu, 1994)",
        "r_a":            0.15,
        "mf_type":        "Multivariate Gaussian",
        "consequent":     "Takagi–Sugeno 1st order (linear)",
        "target":         "first-order differences (delta x)",
        "loss":           "MSELoss",
        "optimizer":      "Adam",
        "learning_rate":  0.003,
        "epochs":         120,
        "batch_size":     128,
    }

    def __init__(self, lag: int = 4, n_concepts: int = 6,
                 r_a: float = 0.15, lr: float = 0.003,
                 epochs: int = 120, batch_size: int = 128,
                 device: str = "cpu"):
        self.lag = lag
        self.n_concepts = n_concepts
        self.r_a = r_a
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.net = None
        self.n_rules = None
        self.n_params = None
        self._diff_min = None
        self._diff_max = None

    def _make_diff_features(self, X: np.ndarray):
        dX = np.diff(X, axis=0)
        self._diff_min = dX.min()
        self._diff_max = dX.max()
        rng = self._diff_max - self._diff_min + 1e-9
        dX_norm = (dX - self._diff_min) / rng
        Xs, Ys = [], []
        for t in range(self.lag, len(dX_norm)):
            Xs.append(dX_norm[t - self.lag:t].flatten())
            Ys.append(dX_norm[t])
        Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
        Ys = torch.tensor(np.array(Ys), dtype=torch.float32)
        return Xs, Ys

    def _denormalize_diff(self, d_norm):
        rng = self._diff_max - self._diff_min + 1e-9
        return d_norm * rng + self._diff_min

    def fit(self, X: np.ndarray):
        Xs, Ys = self._make_diff_features(X)

        print(f"    [ANFIS] subtractive clustering (r_a={self.r_a}) ...")
        centers = subtractive_clustering(Xs.numpy(), r_a=self.r_a)
        self.n_rules = len(centers)
        print(f"    [ANFIS] выявлено правил: {self.n_rules}")

        self.net = ANFISNet(centers, n_outputs=self.n_concepts).to(self.device)
        self.n_params = sum(p.numel() for p in self.net.parameters())
        print(f"    [ANFIS] параметров: {self.n_params:,}")
        print(f"    [ANFIS] обучающих пар: {len(Xs)}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(Xs, Ys),
                            batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                y_pred = self.net(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            if epoch % 10 == 0 or epoch == 1:
                print(f"    [ANFIS] epoch {epoch:3d}/{self.epochs}  "
                      f"loss = {total_loss / len(Xs):.6f}")
        return self

    @torch.no_grad()
    def predict(self, X_history: np.ndarray, n_steps: int) -> np.ndarray:
        self.net.eval()
        rng = self._diff_max - self._diff_min + 1e-9
        dX_hist = list((np.diff(X_history, axis=0) - self._diff_min) / rng)
        last_abs = X_history[-1].copy()
        preds = []
        for _ in range(n_steps):
            feat = np.array(dX_hist[-self.lag:]).flatten()
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            d_norm = self.net(x).cpu().numpy()[0]
            dy = self._denormalize_diff(d_norm)
            last_abs = np.clip(last_abs + dy, 0.0, 1.0)
            preds.append(last_abs.copy())
            dX_hist.append(d_norm)
        return np.array(preds)

    def predict_step(self, history: np.ndarray) -> np.ndarray:
        return self.predict(history, 1)[0]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import load_nems, split_train_test, evaluate, print_results_table, LAG

    print("=" * 70)
    print(f"  ANFIS (PyTorch, обучение на разностях) — НЭМС (LAG={LAG})")
    print("=" * 70)

    X, scaler, cols = load_nems(normalize=False)
    print(f"Выборка: {X.shape}, концепты: {cols}")

    X_train, X_test = split_train_test(X, test_ratio=0.2)
    X_train = X_train[-1500:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = ANFIS_NEMS(lag=LAG, n_concepts=6)
    model.fit(X_train)

    full = np.vstack([X_train, X_test])
    preds = np.array([
        model.predict_step(full[len(X_train) + i - model.lag - 1:len(X_train) + i])
        for i in range(len(X_test))
    ])
    result = evaluate("ANFIS", X_test, preds)
    print_results_table([result], "Результат ANFIS — НЭМС")
