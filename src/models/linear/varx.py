"""Original seven-station VARX helpers, extracted without math changes."""
import numpy as np
from sklearn.linear_model import Ridge

def design_matrix(dataset):
    x = dataset.x_scaled
    residual = dataset.residual
    rows, targets, tides = [], [], []
    for t in dataset.indices:
        t = int(t)
        rows.append(np.concatenate([
            x[t - dataset.window:t, :, 0].reshape(-1),
            x[t - 1].reshape(-1),
            x[t - dataset.window:t].mean(axis=0).reshape(-1),
        ]))
        targets.append(residual[t:t + dataset.horizon].T.reshape(-1))
        tides.append(dataset.tide[t:t + dataset.horizon].T)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), np.asarray(tides, dtype=np.float64)

def fit_varx(data, alpha_grid=(0.1, 1.0, 10.0, 100.0)):
    tx, ty, _ = design_matrix(data["single_train"])
    vx, vy, _ = design_matrix(data["single_val"])
    best_alpha, best_loss = None, np.inf
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, solver="lsqr").fit(tx, ty)
        loss = np.mean((model.predict(vx) - vy) ** 2)
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(tx, ty)
    outputs = {}
    for name in ("single_train", "single_val", "single_test"):
        x, y, tide = design_matrix(data[name])
        outputs[name] = {
            "pred": model.predict(x).reshape(-1, 7, data[name].horizon).astype(np.float32),
            "true": y.reshape(-1, 7, data[name].horizon).astype(np.float32),
            "tide": tide.astype(np.float32),
        }
    return outputs, best_alpha
