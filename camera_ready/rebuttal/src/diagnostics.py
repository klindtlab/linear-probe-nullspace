"""D1, D2 and D3, with one definition each, shared by every lane and figure.

D1, reverse predictivity.
    Fit ridge from targets to features on the training rows; report the share of
    held-out total feature variance it explains,
        D1 = 1 - sum_d SSE_d / sum_d SST_d.
    Also report the top-r eigenvalue share of the held-out feature covariance,
        ceiling(r) = sum_{i<=r} lambda_i / sum_i lambda_i,
    a Ky Fan bound on what ANY rank-r linear structure could explain of the total
    variance, and the ratio D1 / ceiling(r). The ratio normalizes against a loose
    bound, not an attainable maximum, so raw D1 is always reported beside it.
    r is the effective rank of the target block on the training rows, so constant
    columns (absorbed by the intercept) cannot inflate it.

D2, spectral concentration.
    S(k) = reverse R^2 after PCA and whitening FITTED ON THE TRAINING ROWS ONLY,
    retaining the top k principal directions. Whitening removes variance
    concentration inside the retained subspace, so S(k) rising early means the
    target-aligned directions are the leading principal directions rather than
    merely present. The scalar is S(r_eff). This is the Section 3.3 definition,
    and it is now what both the simulation panel and the dataset panels compute.

D3, out-of-distribution generalization of the forward probe.
    SVG-World: fit on one style's training rows, evaluate on the other style's
    test rows, both directions, features centered per style so direction transfer
    is tested rather than mean transfer.
    Added datasets: fit on the training half of a checkerboard over the two
    highest-cardinality varying factors, evaluate on held-out combinations, so
    every marginal factor value was seen but half their joint combinations were
    not.

Uncertainty.
    Point estimates average five repeated 50/50 splits over split units (scene
    ids, or source factor-combination hashes). Intervals are paired bootstrap
    over held-out split units (10,000 resamples) with the fitted probe held
    fixed, so a pretrained-minus-random difference is evaluated on identical rows.
"""

from __future__ import annotations

import numpy as np

import rebuttal_config as C
from fast_ridge import RidgePath, select_alpha


# ---------------------------------------------------------------------------
# Splits and their audit
# ---------------------------------------------------------------------------
def unit_split(n: int, seed: int, test_fraction: float = C.TEST_FRACTION):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def audit_split(train_idx, test_idx, unit_hash: np.ndarray) -> dict:
    """Leakage audit at the index level AND at the content level.

    `unit_hash` is a deterministic hash of the row's generative content (the
    latent vector, or the source factor combination), so two different rows that
    encode the same underlying scene cannot straddle the split unnoticed.
    """
    return dict(
        n_train=int(len(train_idx)), n_test=int(len(test_idx)),
        index_overlap=int(len(np.intersect1d(train_idx, test_idx))),
        duplicate_units_in_source=int(len(unit_hash) - len(np.unique(unit_hash))),
        content_overlap=int(len(np.intersect1d(unit_hash[train_idx],
                                               unit_hash[test_idx]))),
    )


def effective_rank(Y: np.ndarray, tol: float = 1e-8) -> int:
    Yc = Y - Y.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Yc, compute_uv=False)
    if s.size == 0 or s.max() <= 0:
        return 0
    return int((s > tol * s.max()).sum())


# ---------------------------------------------------------------------------
# Per-row terms, so the bootstrap is exact and cheap
# ---------------------------------------------------------------------------
def _agg_terms(Y, Yhat):
    """Per-row squared error and squared deviation, summed over columns."""
    Y = np.asarray(Y, np.float64)
    resid = ((Y - Yhat) ** 2).sum(axis=1)
    dev = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=1)
    return resid, dev


def _r2(resid, dev, idx=None) -> float:
    if idx is not None:
        resid, dev = resid[idx], dev[idx]
    return float(1.0 - resid.sum() / max(dev.sum(), 1e-30))


def bootstrap_ci(terms_a, terms_b=None, n_boot=C.N_BOOTSTRAP,
                 seed=C.BOOTSTRAP_SEED, level=C.CI_LEVEL) -> dict:
    """Paired bootstrap over held-out rows. With two term sets the statistic is
    the difference of their R^2 on the SAME resampled rows."""
    resid_a, dev_a = terms_a
    n = len(resid_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    sa = 1.0 - resid_a[idx].sum(axis=1) / np.maximum(dev_a[idx].sum(axis=1), 1e-30)
    stats = sa
    if terms_b is not None:
        rb, db = terms_b
        sb = 1.0 - rb[idx].sum(axis=1) / np.maximum(db[idx].sum(axis=1), 1e-30)
        stats = sa - sb
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    point = _r2(*terms_a) - (_r2(*terms_b) if terms_b is not None else 0.0)
    return dict(point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0), boot_std=float(stats.std()),
                n_boot=int(n_boot), n_units=int(n))


def bootstrap_ci_chunked(terms_a, terms_b=None, n_boot=C.N_BOOTSTRAP,
                         seed=C.BOOTSTRAP_SEED, level=C.CI_LEVEL,
                         chunk=500) -> dict:
    """Memory-bounded variant for large held-out sets."""
    resid_a, dev_a = terms_a
    n = len(resid_a)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        sa = 1.0 - resid_a[idx].sum(axis=1) / np.maximum(dev_a[idx].sum(axis=1), 1e-30)
        if terms_b is not None:
            rb, db = terms_b
            sb = 1.0 - rb[idx].sum(axis=1) / np.maximum(db[idx].sum(axis=1), 1e-30)
            sa = sa - sb
        stats[done:done + m] = sa
        done += m
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    point = _r2(*terms_a) - (_r2(*terms_b) if terms_b is not None else 0.0)
    return dict(point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0), boot_std=float(stats.std()),
                n_boot=int(n_boot), n_units=int(n))


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------
def _gram_eigenvalues(Xc: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of Xc^T Xc, via the smaller Gram matrix.

    Equivalent to the squared singular values of Xc but far cheaper: for a
    5000 x 768 block this is a 768 x 768 symmetric eigendecomposition instead of
    a full 5000 x 768 SVD. That difference was the dominant CPU cost in the
    smoke, where diagnostics took longer than all GPU inference combined.
    """
    G = Xc.T @ Xc
    G = 0.5 * (G + G.T)
    ev = np.linalg.eigvalsh(G)[::-1]
    return np.maximum(ev, 0.0)


def variance_ceiling(X_te: np.ndarray, r: int):
    Xc = np.asarray(X_te, np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    ev = _gram_eigenvalues(Xc)
    share = ev / ev.sum()
    r = max(min(r, len(ev)), 1)
    return float(share[:r].sum()), share


def d1_reverse(Y_tr, X_tr, Y_te, X_te, alpha=None, cv_seed=0) -> dict:
    """Raw D1, the Ky Fan ceiling at the target's effective rank, and the ratio."""
    if alpha is None:
        alpha, _ = select_alpha(Y_tr, X_tr, seed=cv_seed)
    path = RidgePath(Y_tr, X_tr)
    X_pred = path.predict(Y_te, alpha)
    resid, dev = _agg_terms(X_te, X_pred)
    r = effective_rank(Y_tr)
    ceil, share = variance_ceiling(X_te, r)
    raw = _r2(resid, dev)
    return dict(alpha=float(alpha), target_rank=int(r), d1_raw=raw,
                ceiling=float(ceil),
                d1_normalized=float(raw / ceil) if ceil > 0 else float("nan"),
                pc1_share=float(share[0]),
                top12_share=float(share[:12].sum()),
                top64_share=float(share[:min(64, len(share))].sum()),
                terms=(resid, dev))


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------
def pca_whiten_fit(X_tr: np.ndarray):
    """PCA + whitening fitted on training rows only."""
    X = np.asarray(X_tr, np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    G = Xc.T @ Xc
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0.0), V[:, order]
    eig = w / max(Xc.shape[0] - 1, 1)
    keep = int(np.sum(eig > eig[0] * 1e-10)) if eig[0] > 0 else 1
    scale = 1.0 / np.sqrt(np.maximum(eig[:keep], 1e-30))

    def transform(A, k):
        k = min(k, keep)
        return ((np.asarray(A, np.float64) - mu) @ V[:, :k]) * scale[:k]

    return transform, eig, keep


def d2_curve(Y_tr, X_tr, Y_te, X_te, ks=C.D2_KS, alpha=None, cv_seed=0):
    """S(k) over a k grid, plus the training feature eigenvalues.

    alpha is selected once at k = r_eff and reused across the curve, so a single
    regularization choice cannot make the shape of the curve an artifact of
    per-point tuning.
    """
    transform, eig, keep = pca_whiten_fit(X_tr)
    r_eff = max(effective_rank(Y_tr), 1)
    if alpha is None:
        A_ref = transform(X_tr, min(r_eff, keep))
        alpha, _ = select_alpha(Y_tr, A_ref, seed=cv_seed)
    curve = {}
    for k in sorted(set(list(ks) + [r_eff])):
        if k > keep or k < 1:
            continue
        A_tr, A_te = transform(X_tr, k), transform(X_te, k)
        pred = RidgePath(Y_tr, A_tr).predict(Y_te, alpha)
        resid, dev = _agg_terms(A_te, pred)
        curve[int(k)] = dict(S=_r2(resid, dev), terms=(resid, dev))
    return dict(curve=curve, alpha=float(alpha), r_eff=int(r_eff),
                n_components_kept=int(keep),
                eigenvalues_top20=[float(v) for v in eig[:20]],
                eigenvalue_total=float(eig.sum()),
                S_at_r_eff=float(curve[min(r_eff, keep)]["S"]))


# ---------------------------------------------------------------------------
# Forward probe and D3
# ---------------------------------------------------------------------------
def forward_probe(X_tr, Y_tr, X_te, Y_te, alpha=None, cv_seed=0,
                  labels: list[str] | None = None) -> dict:
    """Per-target forward R^2 and Pearson rho, with per-row terms for CIs."""
    if alpha is None:
        alpha, _ = select_alpha(X_tr, Y_tr, seed=cv_seed)
    pred = RidgePath(X_tr, Y_tr).predict(X_te, alpha)
    Y_te = np.asarray(Y_te, np.float64)
    sq_res = (Y_te - pred) ** 2
    sq_dev = (Y_te - Y_te.mean(axis=0, keepdims=True)) ** 2
    r2_per = 1.0 - sq_res.sum(axis=0) / np.maximum(sq_dev.sum(axis=0), 1e-30)
    yt = Y_te - Y_te.mean(axis=0, keepdims=True)
    yp = pred - pred.mean(axis=0, keepdims=True)
    rho = ((yt * yp).sum(axis=0) /
           (np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12))
    return dict(alpha=float(alpha),
                r2_per_target=[float(v) for v in r2_per],
                rho_per_target=[float(v) for v in rho],
                labels=labels,
                r2_mean=float(r2_per.mean()),
                r2_aggregate=_r2(sq_res.sum(axis=1), sq_dev.sum(axis=1)),
                terms=(sq_res.sum(axis=1), sq_dev.sum(axis=1)),
                terms_percol=(sq_res, sq_dev))


def d3_cross_style(X_src_tr, Y_tr, X_tgt_te, Y_te, alpha=None, cv_seed=0,
                   labels=None) -> dict:
    """Per-style centering: what must transfer is the direction, not the mean."""
    Xs = np.asarray(X_src_tr, np.float64)
    Xt = np.asarray(X_tgt_te, np.float64)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xt = Xt - Xt.mean(axis=0, keepdims=True)
    return forward_probe(Xs, Y_tr, Xt, Y_te, alpha=alpha, cv_seed=cv_seed,
                         labels=labels)
