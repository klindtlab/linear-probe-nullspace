"""Ridge regression with a cheap regularization path.

The diagnostics need the same ridge fit at nine alphas inside five CV folds, for
several hundred conditions. Refitting from scratch would dominate the run, so
each fold eigendecomposes the centered Gram matrix once and then every alpha is
a diagonal rescale. Results match `sklearn.linear_model.Ridge` with
`fit_intercept=True` to numerical precision, which `tests/test_fast_ridge.py`
asserts.
"""

from __future__ import annotations

import numpy as np

import rebuttal_config as C


class RidgePath:
    """Fit once, then read out coefficients for any alpha.

    Centers X and Y, eigendecomposes X_c^T X_c, and solves
        W(alpha) = V diag(1 / (lambda + alpha)) V^T X_c^T Y_c
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.y_mean = Y.mean(axis=0, keepdims=True)
        Xc = X - self.x_mean
        Yc = Y - self.y_mean
        G = Xc.T @ Xc
        # Symmetrize against round-off before eigh.
        G = 0.5 * (G + G.T)
        self.evals, self.evecs = np.linalg.eigh(G)
        self.evals = np.maximum(self.evals, 0.0)
        self._vt_xty = self.evecs.T @ (Xc.T @ Yc)

    def coef(self, alpha: float) -> np.ndarray:
        return self.evecs @ (self._vt_xty / (self.evals[:, None] + alpha))

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.x_mean
        return Xc @ self.coef(alpha) + self.y_mean


def aggregate_r2(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """Total variance explained, summed over output columns."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    sse = ((Y - Yhat) ** 2).sum()
    sst = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - sse / max(sst, 1e-30))


def kfold(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return [(np.setdiff1d(idx, f), f) for f in folds]


def select_alpha(X: np.ndarray, Y: np.ndarray, alphas=C.RIDGE_ALPHAS,
                 k: int = C.RIDGE_CV_FOLDS, seed: int = 0) -> tuple[float, dict]:
    """Choose alpha by k-fold CV inside the supplied (training) data only.

    The selection criterion is the same aggregate R^2 the diagnostic reports, so
    the regularization is tuned for the quantity being measured.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    scores = {a: [] for a in alphas}
    for tr, va in kfold(X.shape[0], k, seed=seed):
        path = RidgePath(X[tr], Y[tr])
        for a in alphas:
            scores[a].append(aggregate_r2(Y[va], path.predict(X[va], a)))
    mean_scores = {float(a): float(np.mean(v)) for a, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return float(best), mean_scores


def fit_predict(X_tr, Y_tr, X_te, alpha: float | None = None,
                cv_seed: int = 0):
    """Convenience: select alpha inside train if not given, fit, predict test."""
    cv = None
    if alpha is None:
        alpha, cv = select_alpha(X_tr, Y_tr, seed=cv_seed)
    path = RidgePath(X_tr, Y_tr)
    return path.predict(X_te, alpha), float(alpha), cv
