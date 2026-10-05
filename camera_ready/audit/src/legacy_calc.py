"""The submitted recipe, transcribed from the submission archive.

Source: `svg_world/4_reverse_predictivity.py` (and its loader, shared
with `3_linear_probe.py`). Reproduced here unchanged in substance so the audit
can run the exact recipe that produced the reported 0.18 / 0.02, and then move
one factor at a time toward the revised recipe.

Four differences from the revised estimator, each isolated in the attribution
ladder:

  1. features are cast to float32, not float64;
  2. the split is `sklearn.model_selection.train_test_split(test_size=0.5,
     random_state=42)`, a single row-level split, not five unit-level splits;
  3. ridge alpha is hardcoded to 1.0, not chosen by CV inside the training half;
  4. the statistic uses per-column VARIANCES of the residual and of the
     features,
         1 - sum_d Var(x_d - xhat_d) / sum_d Var(x_d),
     which centers the residual and therefore discards any mean-offset error,
     where the revised estimator uses uncentered sums of squared error over the
     same denominator's deviations,
         1 - sum SSE / sum SST.
  5. the target block is all 32 stored latent columns, 12 of which are constant.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def legacy_split(n: int, seed: int = 42, test_size: float = 0.5):
    """The submitted split, expressed as index arrays."""
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return tr, te


def legacy_reverse_r2(Z_tr, X_tr, Z_te, X_te, alpha: float = 1.0) -> float:
    """`reverse_r2` from 4_reverse_predictivity.py, verbatim in substance."""
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(np.asarray(Z_tr, np.float32), np.asarray(X_tr, np.float32))
    X_pred = probe.predict(np.asarray(Z_te, np.float32))
    X_te = np.asarray(X_te, np.float32)
    total = X_te.var(axis=0).sum() + 1e-12
    resid = (X_te - X_pred).var(axis=0).sum()
    return float(1.0 - resid / total)


def legacy_standardize(X_tr, X_te):
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


def legacy_pca_whiten(X_tr, X_te, n_components=None):
    from sklearn.decomposition import PCA

    if n_components is None:
        n_components = min(X_tr.shape) - 1
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    return pca.fit_transform(X_tr), pca.transform(X_te)


def legacy_full(Z, X, alpha: float = 1.0, seed: int = 42,
                whiten_k=None) -> dict:
    """The three normalizations the submitted script prints, on one split."""
    tr, te = legacy_split(len(X), seed=seed)
    X = np.asarray(X, np.float32)
    Z = np.asarray(Z, np.float32)
    X_tr, X_te, Z_tr, Z_te = X[tr], X[te], Z[tr], Z[te]
    raw = legacy_reverse_r2(Z_tr, X_tr, Z_te, X_te, alpha)
    Xs_tr, Xs_te = legacy_standardize(X_tr, X_te)
    std = legacy_reverse_r2(Z_tr, Xs_tr, Z_te, Xs_te, alpha)
    k = whiten_k or (min(X_tr.shape) - 1)
    Xw_tr, Xw_te = legacy_pca_whiten(X_tr, X_te, n_components=k)
    pw = legacy_reverse_r2(Z_tr, Xw_tr, Z_te, Xw_te, alpha)
    return dict(raw=raw, standardized=std, pca_whitened=pw, whiten_k=int(k),
                alpha=float(alpha), seed=int(seed),
                n_train=int(len(tr)), n_test=int(len(te)))
