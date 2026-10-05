"""Independent implementation of D1, D2, D3 and their controls.

Written from the equations printed in the submitted manuscript, without
importing the prior rebuttal run's `diagnostics.py`. The prior module is used
only as a second opinion in the identical-array parity check
(`run_audit.py --stage parity`); nothing here calls it.

Sources, verbatim from the submission:

  D1, Eq. (3):  R2_rev = 1 - E||f(x) - W_rev z(x) - mu||^2 / E||f(x) - E f||^2,
                (W_rev, mu) = argmin E||f - W z - mu||^2.
  Prop. 2, Eq. (4): for f = A z + mu + eps with E[eps z^T] = 0,
                R2_rev = tr(A Cov(z) A^T) / (tr(A Cov(z) A^T) + tr Cov(eps)).
  D2, Sec 3.3: the curve R2_rev(k), reverse R2 after projecting f to its top k
                principal components with each component rescaled to unit
                variance (PCA-whitening). Here S(k).
  D3, Sec 3.3: train the forward probe on one style, evaluate on the paired
                style with the same latent layout.

Estimators are held-out throughout: the map is fit on training rows only and
every reported quantity is computed on held-out rows.

Two quantities defined here are NOT in the submission and are labelled as
audit-side proposals:

  C(k)  the fraction of the held-out latent-predictable feature variance that
        lies inside the training top-k principal subspace. Monotone in k, in
        [0, 1], equal to 1 at k = d, and equal to 1 at k = r exactly when the
        latent-aligned subspace is contained in the top-r principal subspace,
        which is the property D2 is described as measuring.
  ceiling(r)  the top-r eigenvalue share of the held-out feature covariance, a
        Ky Fan upper bound on what any rank-r linear structure can explain of
        the total variance.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Ridge in the "many outputs, few inputs or few outputs" regime.
# ---------------------------------------------------------------------------


def ridge_fit(A: np.ndarray, B: np.ndarray, alpha: float):
    """Least squares B ~ A W + b with an L2 penalty alpha on W (not on b).

    Implemented from the normal equations on centered data, so the intercept is
    exact and the penalty never touches it.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    a_mu = A.mean(axis=0, keepdims=True)
    b_mu = B.mean(axis=0, keepdims=True)
    Ac, Bc = A - a_mu, B - b_mu
    p = Ac.shape[1]
    G = Ac.T @ Ac + alpha * np.eye(p)
    W = np.linalg.solve(G, Ac.T @ Bc)
    return W, b_mu - a_mu @ W


def ridge_predict(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(A, np.float64) @ W + b


def ridge_cv_alpha(A: np.ndarray, B: np.ndarray, alphas, folds: int = 5,
                   seed: int = 0) -> tuple[float, dict]:
    """K-fold CV inside the given rows only. Score is the aggregate R^2 over all
    output columns, i.e. 1 - sum SSE / sum SST, matching how D1 aggregates.

    The whole alpha grid is scored from ONE eigendecomposition per fold. Solving
    the normal equations separately for every alpha re-forms the p x p Gram
    matrix 45 times per call, which at p = 2048 dominated the run: the first
    attempt at the dataset lanes spent two hours in this function. With the
    Gram eigendecomposed once, each additional alpha costs a diagonal rescale.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % folds
    alphas = [float(a) for a in alphas]
    sse = {a: 0.0 for a in alphas}
    sst = 0.0
    for f in range(folds):
        tr, te = fold != f, fold == f
        A_tr, B_tr, A_te, B_te = A[tr], B[tr], A[te], B[te]
        a_mu = A_tr.mean(axis=0, keepdims=True)
        b_mu = B_tr.mean(axis=0, keepdims=True)
        Ac, Bc = A_tr - a_mu, B_tr - b_mu
        G = Ac.T @ Ac
        G = 0.5 * (G + G.T)
        w, V = np.linalg.eigh(G)
        w = np.maximum(w, 0.0)
        rhs = V.T @ (Ac.T @ Bc)              # p x m, rotated into eigenbasis
        A_te_rot = (A_te - a_mu) @ V         # n_te x p
        sst += float(((B_te - b_mu) ** 2).sum())
        for a in alphas:
            coef = rhs / (w[:, None] + a)
            pred = A_te_rot @ coef + b_mu
            sse[a] += float(((B_te - pred) ** 2).sum())
    scores = {a: 1.0 - sse[a] / max(sst, 1e-30) for a in alphas}
    best = max(scores, key=scores.get)
    return float(best), scores


# ---------------------------------------------------------------------------
# Aggregate R^2 with per-row terms, so a bootstrap can resample rows exactly.
# ---------------------------------------------------------------------------


def r2_terms(Y: np.ndarray, Yhat: np.ndarray, center: str = "test"
             ) -> tuple[np.ndarray, np.ndarray]:
    """Per-row squared error and per-row squared deviation from the mean.

    `center` selects the denominator's reference mean: "test" uses the held-out
    mean (the sample analogue of E f on the evaluation distribution), "train"
    requires Yhat to carry the training mean and is provided for the ablation
    that shows the choice is immaterial here.
    """
    Y = np.asarray(Y, np.float64)
    resid = ((Y - Yhat) ** 2).sum(axis=1)
    if center == "test":
        ref = Y.mean(axis=0, keepdims=True)
    elif center == "zero":
        ref = np.zeros((1, Y.shape[1]))
    else:
        raise ValueError(center)
    dev = ((Y - ref) ** 2).sum(axis=1)
    return resid, dev


def r2_from_terms(resid: np.ndarray, dev: np.ndarray, idx=None) -> float:
    if idx is not None:
        resid, dev = resid[idx], dev[idx]
    return float(1.0 - resid.sum() / max(dev.sum(), 1e-30))


def effective_rank(Y: np.ndarray, tol: float = 1e-8) -> int:
    Yc = np.asarray(Y, np.float64)
    Yc = Yc - Yc.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Yc, compute_uv=False)
    if s.size == 0 or s.max() <= 0:
        return 0
    return int((s > tol * s.max()).sum())


def _eig_desc(Xc: np.ndarray):
    """Descending eigenvalues/vectors of the centered second-moment matrix."""
    G = Xc.T @ Xc
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    return np.maximum(w[order], 0.0), V[:, order]


def variance_ceiling(X: np.ndarray, r: int) -> tuple[float, np.ndarray]:
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    ev, _ = _eig_desc(Xc)
    share = ev / max(ev.sum(), 1e-300)
    r = int(max(min(r, len(ev)), 1))
    return float(share[:r].sum()), share


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------


def d1(Z_tr, X_tr, Z_te, X_te, alpha: float) -> dict:
    """Eq. (3), held out. Returns the raw statistic, the Ky Fan ceiling at the
    target's effective rank, the ratio, and per-row terms for the bootstrap."""
    W, b = ridge_fit(Z_tr, X_tr, alpha)
    pred = ridge_predict(Z_te, W, b)
    resid, dev = r2_terms(X_te, pred)
    r = effective_rank(Z_tr)
    ceil, share = variance_ceiling(X_te, r)
    raw = r2_from_terms(resid, dev)
    return dict(
        d1_raw=raw, alpha=float(alpha), target_rank=int(r), ceiling=float(ceil),
        d1_normalized=float(raw / ceil) if ceil > 0 else float("nan"),
        pc1_share=float(share[0]),
        participation_ratio=float(share.sum() ** 2 / (share ** 2).sum()),
        terms=(resid, dev), W=W, b=b,
    )


def prop2_closed_form(A: np.ndarray, cov_z: np.ndarray,
                      cov_eps: np.ndarray) -> float:
    """Eq. (4). `cov_z` may be singular; no inverse is required here."""
    sig = float(np.trace(A @ cov_z @ A.T))
    return sig / (sig + float(np.trace(cov_eps)))


def prop2_from_data(Z: np.ndarray, X: np.ndarray) -> float:
    """Eq. (4) evaluated on a sample by taking A as the population-optimal map,
    with a pseudoinverse so a singular Cov(z) is handled rather than crashing.
    This is the corrected form of Prop. 2 the audit recommends printing."""
    Z = np.asarray(Z, np.float64)
    X = np.asarray(X, np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Zc.shape[0]
    cov_z = Zc.T @ Zc / n
    cov_xz = Xc.T @ Zc / n
    A = cov_xz @ np.linalg.pinv(cov_z)
    resid = Xc - Zc @ A.T
    cov_eps = resid.T @ resid / n
    return prop2_closed_form(A, cov_z, cov_eps)


# ---------------------------------------------------------------------------
# D2: S(k) as printed in Sec 3.3, and the monotone alternative C(k)
# ---------------------------------------------------------------------------


def pca_whitener(X_tr: np.ndarray, var_floor: float = 1e-10):
    """PCA basis and whitening scales fitted on training rows only."""
    X = np.asarray(X_tr, np.float64)
    mu = X.mean(axis=0, keepdims=True)
    ev, V = _eig_desc(X - mu)
    var = ev / max(X.shape[0] - 1, 1)
    keep = int((var > var[0] * var_floor).sum()) if var[0] > 0 else 1
    scale = 1.0 / np.sqrt(np.maximum(var[:keep], 1e-300))

    def transform(B: np.ndarray, k: int) -> np.ndarray:
        k = int(min(k, keep))
        return ((np.asarray(B, np.float64) - mu) @ V[:, :k]) * scale[:k]

    return dict(transform=transform, var=var, keep=keep, mu=mu, V=V)


def d2_S_curve(Z_tr, X_tr, Z_te, X_te, ks, alpha: float) -> dict:
    """S(k): D1 computed on the top-k PCA-whitened features (train-fitted)."""
    wh = pca_whitener(X_tr)
    out = {}
    for k in sorted({int(k) for k in ks}):
        if k < 1 or k > wh["keep"]:
            continue
        A_tr, A_te = wh["transform"](X_tr, k), wh["transform"](X_te, k)
        W, b = ridge_fit(Z_tr, A_tr, alpha)
        resid, dev = r2_terms(A_te, ridge_predict(Z_te, W, b))
        out[k] = dict(S=r2_from_terms(resid, dev), terms=(resid, dev))
    return dict(curve=out, keep=int(wh["keep"]),
                var_top20=[float(v) for v in wh["var"][:20]])


def d2_C_curve(Z_tr, X_tr, Z_te, X_te, ks, alpha: float) -> dict:
    """C(k): share of the held-out latent-predictable feature variance that lies
    in the training top-k principal subspace.

    g(x) = W z(x) + b is fit on the training rows in the ORIGINAL feature
    coordinates (no whitening). On held-out rows,
        C(k) = sum_i ||P_k (g_i - g_bar)||^2 / sum_i ||g_i - g_bar||^2,
    with P_k the projector onto the top-k training principal directions. C is
    monotone nondecreasing, lands in [0, 1], and reaches 1 at k = d.
    """
    wh = pca_whitener(X_tr)
    V = wh["V"]
    W, b = ridge_fit(Z_tr, X_tr, alpha)
    g = ridge_predict(Z_te, W, b)
    gc = g - g.mean(axis=0, keepdims=True)
    coords = gc @ V                          # rotate into the PC basis
    energy = (coords ** 2).sum(axis=0)       # per-direction energy of g
    total = float(energy.sum())
    cum = np.cumsum(energy) / max(total, 1e-300)
    curve = {int(k): float(cum[min(int(k), len(cum)) - 1])
             for k in sorted({int(k) for k in ks}) if 1 <= int(k) <= len(cum)}

    def first_at(level: float) -> int:
        hit = np.nonzero(cum >= level)[0]
        return int(hit[0] + 1) if hit.size else int(len(cum))

    return dict(curve=curve, k50=first_at(0.5), k90=first_at(0.9),
                k99=first_at(0.99), d=int(len(cum)),
                cum=[float(v) for v in cum])


# ---------------------------------------------------------------------------
# Forward probe and D3
# ---------------------------------------------------------------------------


def forward_probe(X_tr, Y_tr, X_te, Y_te, alpha: float,
                  labels=None) -> dict:
    W, b = ridge_fit(X_tr, Y_tr, alpha)
    Y_te = np.asarray(Y_te, np.float64)
    pred = ridge_predict(X_te, W, b)
    sq_res = (Y_te - pred) ** 2
    sq_dev = (Y_te - Y_te.mean(axis=0, keepdims=True)) ** 2
    r2_per = 1.0 - sq_res.sum(axis=0) / np.maximum(sq_dev.sum(axis=0), 1e-30)
    yt = Y_te - Y_te.mean(axis=0, keepdims=True)
    yp = pred - pred.mean(axis=0, keepdims=True)
    rho = ((yt * yp).sum(axis=0) /
           (np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12))
    return dict(alpha=float(alpha), labels=labels,
                r2_per_target=[float(v) for v in r2_per],
                rho_per_target=[float(v) for v in rho],
                r2_mean=float(r2_per.mean()), rho_mean=float(rho.mean()),
                r2_aggregate=r2_from_terms(sq_res.sum(axis=1),
                                           sq_dev.sum(axis=1)),
                terms=(sq_res.sum(axis=1), sq_dev.sum(axis=1)),
                W=W, b=b)


def d3_transfer(X_src_tr, Y_src_tr, X_tgt, Y_tgt, alpha: float,
                calib_frac: float = 0.5, seed: int = 0, labels=None) -> dict:
    """Three separate quantities, never merged:

    zero_shot     probe fitted on the source and applied to the target with the
                  SOURCE feature mean subtracted, so nothing about the target is
                  used. This is the only variant that is genuinely OOD.
    direction     Pearson rho between prediction and truth on the target, which
                  is invariant to any affine recalibration and therefore
                  measures directional association only.
    calibrated    the source probe's predictions rescaled by a per-target affine
                  fit on a held-in half of the target, scored on the other half.
    """
    Xs = np.asarray(X_src_tr, np.float64)
    Xt = np.asarray(X_tgt, np.float64)
    W, b = ridge_fit(Xs, Y_src_tr, alpha)

    Y_tgt = np.asarray(Y_tgt, np.float64)
    # Source-only pipeline: the fitted intercept already carries the source
    # feature mean, so the probe is applied to the target features untouched.
    pred_zs = ridge_predict(Xt, W, b)
    sq_res = (Y_tgt - pred_zs) ** 2
    sq_dev = (Y_tgt - Y_tgt.mean(axis=0, keepdims=True)) ** 2
    r2_zs = 1.0 - sq_res.sum(axis=0) / np.maximum(sq_dev.sum(axis=0), 1e-30)

    yt = Y_tgt - Y_tgt.mean(axis=0, keepdims=True)
    yp = pred_zs - pred_zs.mean(axis=0, keepdims=True)
    rho = ((yt * yp).sum(axis=0) /
           (np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12))

    # target-calibrated: affine per column, fit on half the target rows
    n = Xt.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = int(round(calib_frac * n))
    cal, ev = perm[:n_cal], perm[n_cal:]
    r2_cal = []
    for j in range(Y_tgt.shape[1]):
        p = pred_zs[cal, j]
        t = Y_tgt[cal, j]
        pm, tm = p.mean(), t.mean()
        denom = float(((p - pm) ** 2).sum())
        slope = float(((p - pm) * (t - tm)).sum() / denom) if denom > 0 else 0.0
        inter = tm - slope * pm
        fit = slope * pred_zs[ev, j] + inter
        sse = float(((Y_tgt[ev, j] - fit) ** 2).sum())
        sst = float(((Y_tgt[ev, j] - Y_tgt[ev, j].mean()) ** 2).sum())
        r2_cal.append(1.0 - sse / max(sst, 1e-30))

    return dict(alpha=float(alpha), labels=labels,
                zero_shot_r2_per_target=[float(v) for v in r2_zs],
                zero_shot_r2_mean=float(np.mean(r2_zs)),
                zero_shot_r2_aggregate=r2_from_terms(sq_res.sum(axis=1),
                                                     sq_dev.sum(axis=1)),
                rho_per_target=[float(v) for v in rho],
                rho_mean=float(np.mean(rho)),
                calibrated_r2_per_target=[float(v) for v in r2_cal],
                calibrated_r2_mean=float(np.mean(r2_cal)),
                n_calib=int(n_cal), n_eval=int(n - n_cal))


# ---------------------------------------------------------------------------
# Splits and uncertainty
# ---------------------------------------------------------------------------


def unit_split(n: int, seed: int, test_fraction: float = 0.5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def split_audit(train_idx, test_idx, unit_hash: np.ndarray) -> dict:
    return dict(
        n_train=int(len(train_idx)), n_test=int(len(test_idx)),
        index_overlap=int(len(np.intersect1d(train_idx, test_idx))),
        duplicate_units=int(len(unit_hash) - len(np.unique(unit_hash))),
        content_overlap=int(len(np.intersect1d(unit_hash[train_idx],
                                               unit_hash[test_idx]))),
    )


def paired_bootstrap(terms_a, terms_b=None, n_boot: int = 10_000,
                     seed: int = 7, level: float = 0.95,
                     chunk: int = 500) -> dict:
    """Resample held-out rows; recompute the aggregate R^2 inside each resample.
    With two term sets the statistic is their difference on the same rows."""
    ra, da = terms_a
    n = len(ra)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        s = 1.0 - ra[idx].sum(axis=1) / np.maximum(da[idx].sum(axis=1), 1e-30)
        if terms_b is not None:
            rb, db = terms_b
            s = s - (1.0 - rb[idx].sum(axis=1) /
                     np.maximum(db[idx].sum(axis=1), 1e-30))
        stats[done:done + m] = s
        done += m
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    point = r2_from_terms(ra, da)
    if terms_b is not None:
        point -= r2_from_terms(*terms_b)
    return dict(point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0),
                boot_std=float(stats.std()), n_boot=int(n_boot),
                n_units=int(n))


def hierarchical_bootstrap(per_seed_terms: dict, baseline_terms,
                           n_boot: int = 10_000, seed: int = 7,
                           level: float = 0.95, chunk: int = 250) -> dict:
    """Two-level bootstrap for a pretrained-minus-random gap.

    `per_seed_terms` maps a random-initialization seed to that instance's
    (resid, dev) on the SAME held-out rows as `baseline_terms`. Each resample
    draws random-init seeds with replacement AND held-out rows with replacement,
    so the interval carries both the finite evaluation sample and the finite
    number of control initializations. Reduces to the paired row bootstrap when
    only one seed is supplied.
    """
    rb, db = baseline_terms
    n = len(rb)
    seeds = sorted(per_seed_terms)
    S = len(seeds)
    ra = np.stack([per_seed_terms[s][0] for s in seeds])
    da = np.stack([per_seed_terms[s][1] for s in seeds])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        row = rng.integers(0, n, size=(m, n))
        pick = rng.integers(0, S, size=(m, S))
        base = 1.0 - rb[row].sum(axis=1) / np.maximum(db[row].sum(axis=1), 1e-30)
        ctrl = np.empty(m)
        for i in range(m):
            rr = ra[pick[i]][:, row[i]].sum(axis=1)
            dd = da[pick[i]][:, row[i]].sum(axis=1)
            ctrl[i] = np.mean(1.0 - rr / np.maximum(dd, 1e-30))
        stats[done:done + m] = base - ctrl
        done += m
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    point = r2_from_terms(rb, db) - float(np.mean(
        [r2_from_terms(*per_seed_terms[s]) for s in seeds]))
    return dict(point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0),
                boot_std=float(stats.std()), n_boot=int(n_boot),
                n_units=int(n), n_control_seeds=int(S))
