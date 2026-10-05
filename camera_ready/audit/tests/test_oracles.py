"""Oracle tests for the independent calculator, and validity tests for the
definitions the manuscript prints.

Two kinds of test live here and they are labelled:

  ORACLE   the estimator must recover a quantity whose exact value is known by
           construction. A failure is a bug in `audit_calc`.
  VALIDITY the definition must satisfy an invariance or implication the
           manuscript claims for it. A failure is a problem with the definition,
           not with the code, and is recorded as such.
"""

from __future__ import annotations

import numpy as np
import pytest

import audit_calc as A


def make_affine(n=4000, k=3, d=32, noise=1.0, seed=0, cov_z=None):
    rng = np.random.default_rng(seed)
    if cov_z is None:
        Z = rng.normal(size=(n, k))
    else:
        Z = rng.multivariate_normal(np.zeros(k), cov_z, size=n)
    W = rng.normal(size=(k, d))
    E = rng.normal(size=(n, d)) * noise
    X = Z @ W + rng.normal(size=(1, d)) * 5.0 + E
    return Z, X, W, E


# --------------------------------------------------------------------------
# ORACLE: D1 recovers the Prop. 2 closed form
# --------------------------------------------------------------------------
def test_oracle_d1_matches_prop2_closed_form():
    n, k, d, noise = 40000, 3, 24, 0.7
    Z, X, W, E = make_affine(n=n, k=k, d=d, noise=noise, seed=1)
    tr, te = A.unit_split(n, seed=0)
    out = A.d1(Z[tr], X[tr], Z[te], X[te], alpha=1e-6)
    cov_z = np.cov(Z.T, bias=True)
    sig = float(np.trace(W.T @ cov_z @ W))
    closed = sig / (sig + d * noise ** 2)
    assert abs(out["d1_raw"] - closed) < 0.01, (out["d1_raw"], closed)
    # and the sample-plug-in form of Eq. (4) agrees with the held-out estimator
    assert abs(A.prop2_from_data(Z, X) - out["d1_raw"]) < 0.01


# --------------------------------------------------------------------------
# ORACLE: singular / constant target columns
# --------------------------------------------------------------------------
def test_oracle_constant_target_columns_are_inert():
    n = 4000
    Z, X, _, _ = make_affine(n=n, k=3, d=16, noise=0.5, seed=2)
    Zpad = np.hstack([Z, np.ones((n, 5)) * 3.0])   # 5 constant columns
    tr, te = A.unit_split(n, seed=0)
    a = A.d1(Z[tr], X[tr], Z[te], X[te], alpha=1e-6)
    b = A.d1(Zpad[tr], X[tr], Zpad[te], X[te], alpha=1e-6)
    assert abs(a["d1_raw"] - b["d1_raw"]) < 1e-6
    assert a["target_rank"] == 3 and b["target_rank"] == 3


def test_oracle_rank_deficient_cov_z_needs_pseudoinverse():
    n, k = 4000, 4
    rng = np.random.default_rng(3)
    Zbase = rng.normal(size=(n, 2))
    Z = np.hstack([Zbase, Zbase @ rng.normal(size=(2, k - 2))])  # rank 2
    W = rng.normal(size=(k, 12))
    X = Z @ W + rng.normal(size=(n, 12)) * 0.4
    assert A.effective_rank(Z) == 2
    # Eq. (4) with a plain inverse would raise; the pseudoinverse form returns
    # a finite value that matches the held-out estimator.
    tr, te = A.unit_split(n, seed=0)
    est = A.d1(Z[tr], X[tr], Z[te], X[te], alpha=1e-6)["d1_raw"]
    assert abs(A.prop2_from_data(Z, X) - est) < 0.01
    # Cov(z) is numerically singular, so the plain-inverse form of Eq. (4) is
    # not usable: its condition number is astronomically large.
    Zc = Z - Z.mean(0, keepdims=True)
    assert np.linalg.cond(Zc.T @ Zc / n) > 1e12


# --------------------------------------------------------------------------
# ORACLE / VALIDITY: transformations of the feature coordinates
# --------------------------------------------------------------------------
def test_oracle_d1_invariant_to_orthogonal_feature_transform():
    n = 4000
    Z, X, _, _ = make_affine(n=n, k=3, d=16, noise=0.6, seed=4)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    tr, te = A.unit_split(n, seed=0)
    a = A.d1(Z[tr], X[tr], Z[te], X[te], alpha=1e-6)["d1_raw"]
    b = A.d1(Z[tr], (X @ Q)[tr], Z[te], (X @ Q)[te], alpha=1e-6)["d1_raw"]
    assert abs(a - b) < 1e-8


def test_validity_d1_is_not_invariant_to_coordinate_rescaling():
    """The Def. 1 residual bound sigma^2 ||W||_F^2 and D1 itself both move under
    a rescaling of individual feature coordinates, which changes no content of
    the representation. This is the unit-dependence the audit reports."""
    n, d = 4000, 16
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(n, 3))
    X = np.zeros((n, d))
    X[:, :8] = Z @ rng.normal(size=(3, 8))          # signal-carrying block
    X += rng.normal(size=(n, d)) * 0.6              # residual everywhere
    scale = np.ones(d)
    scale[8:] = 20.0                       # inflate the residual-only block
    tr, te = A.unit_split(n, seed=0)
    a = A.d1(Z[tr], X[tr], Z[te], X[te], alpha=1e-6)["d1_raw"]
    b = A.d1(Z[tr], (X * scale)[tr], Z[te], (X * scale)[te], alpha=1e-6)["d1_raw"]
    fa = A.forward_probe(X[tr], Z[tr], X[te], Z[te], alpha=1e-6)["r2_aggregate"]
    fb = A.forward_probe((X * scale)[tr], Z[tr], (X * scale)[te], Z[te],
                         alpha=1e-6)["r2_aggregate"]
    assert abs(a - b) > 0.05, (a, b)       # D1 moves materially
    assert abs(fa - fb) < 0.02             # forward decoding does not


# --------------------------------------------------------------------------
# VALIDITY: S(k) vs C(k)
# --------------------------------------------------------------------------
def _strict_linear_world_model(n=6000, k=3, d=64, noise=0.05, seed=6):
    """f = W z + mu + eps with a small isotropic residual: a linear world model
    with residual sigma^2 in the sense of Def. 1."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, k))
    Wm, _ = np.linalg.qr(rng.normal(size=(d, k)))
    X = Z @ (Wm * 4.0).T + rng.normal(size=(n, d)) * noise
    return Z, X


def test_validity_S_curve_decays_even_for_a_strict_linear_world_model():
    """Sec 3.3 states that a strict linear world model has S(k) approaching 1
    from k = k_lat onward. It does not: whitening rescales the residual
    directions to unit variance, so S(k) falls as k grows however small the
    residual is."""
    Z, X = _strict_linear_world_model()
    n = len(Z)
    tr, te = A.unit_split(n, seed=0)
    ks = [3, 8, 16, 32, 64]
    S = A.d2_S_curve(Z[tr], X[tr], Z[te], X[te], ks, alpha=1e-6)["curve"]
    assert S[3]["S"] > 0.95
    assert S[64]["S"] < 0.30                      # collapses, not approaching 1
    C = A.d2_C_curve(Z[tr], X[tr], Z[te], X[te], ks, alpha=1e-6)
    assert C["curve"][3] > 0.99                   # C(r) is at 1 as claimed
    assert C["k90"] <= 3


def test_validity_C_curve_is_monotone_and_bounded():
    Z, X = _strict_linear_world_model(noise=1.0, seed=7)
    n = len(Z)
    tr, te = A.unit_split(n, seed=0)
    ks = list(range(1, 65))
    C = A.d2_C_curve(Z[tr], X[tr], Z[te], X[te], ks, alpha=1e-6)
    v = np.array([C["curve"][k] for k in ks])
    assert np.all(np.diff(v) >= -1e-12)
    assert v.min() >= -1e-12 and v.max() <= 1 + 1e-12
    assert abs(v[-1] - 1.0) < 1e-9


def test_validity_C_detects_a_latent_subspace_hidden_below_the_top_pcs():
    """C(r) is low exactly when the latent-aligned subspace is not leading, which
    is the property D2 is described as measuring."""
    rng = np.random.default_rng(8)
    n, d = 6000, 40
    Z = rng.normal(size=(n, 2))
    B = np.zeros((n, d))
    B[:, 30:32] = Z * 1.0                      # latents in weak directions
    B[:, :4] += rng.normal(size=(n, 4)) * 10.0  # unrelated dominant variance
    B += rng.normal(size=(n, d)) * 0.05
    tr, te = A.unit_split(n, seed=0)
    C = A.d2_C_curve(Z[tr], B[tr], Z[te], B[te], [2, 4, 40], alpha=1e-6)
    assert C["curve"][2] < 0.1
    assert C["curve"][40] > 0.99


# --------------------------------------------------------------------------
# VALIDITY: the Ky Fan ceiling is a bound, not an attainable maximum
# --------------------------------------------------------------------------
def test_validity_ky_fan_ceiling_bounds_d1_but_normalisation_is_loose():
    Z, X = _strict_linear_world_model(noise=1.0, seed=9)
    n = len(Z)
    tr, te = A.unit_split(n, seed=0)
    out = A.d1(Z[tr], X[tr], Z[te], X[te], alpha=1e-6)
    assert out["d1_raw"] <= out["ceiling"] + 1e-9
    # A random-direction control with the same spectrum reaches a high ratio
    # without carrying any latent structure in its leading directions.
    rng = np.random.default_rng(10)
    Xrand = rng.normal(size=X.shape) * X.std(axis=0, keepdims=True)
    o2 = A.d1(Z[tr], Xrand[tr], Z[te], Xrand[te], alpha=1e-6)
    assert o2["d1_raw"] < 0.02


# --------------------------------------------------------------------------
# ORACLE: D3 under distribution shift
# --------------------------------------------------------------------------
def test_oracle_d3_separates_zero_shot_direction_and_calibrated():
    """A target domain whose features are the source's, affinely rescaled, has a
    perfect direction score and a perfect calibrated score, while zero-shot R^2
    is arbitrarily negative. Reporting one number for all three hides this."""
    rng = np.random.default_rng(11)
    n, d = 4000, 20
    Z = rng.normal(size=(n, 2))
    Wm = rng.normal(size=(2, d))
    Xs = Z @ Wm + rng.normal(size=(n, d)) * 0.1
    Xt = Xs * 3.0 + 7.0                       # same directions, new scale/offset
    tr, te = A.unit_split(n, seed=0)
    out = A.d3_transfer(Xs[tr], Z[tr], Xt[te], Z[te], alpha=1e-6)
    assert out["zero_shot_r2_mean"] < -1.0
    assert min(out["rho_per_target"]) > 0.99
    assert out["calibrated_r2_mean"] > 0.98


# --------------------------------------------------------------------------
# ORACLE: CV, bootstrap, split audit
# --------------------------------------------------------------------------
def test_oracle_cv_alpha_prefers_weak_regularisation_when_signal_is_clean():
    Z, X = _strict_linear_world_model(noise=0.05, seed=12)
    tr, _ = A.unit_split(len(Z), seed=0)
    best, scores = A.ridge_cv_alpha(Z[tr], X[tr], [1e-3, 1e0, 1e3, 1e5])
    assert best <= 1.0
    assert scores[1e-3] > scores[100000.0]


def test_oracle_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(13)
    Aa = rng.normal(size=(300, 5))
    Bb = rng.normal(size=(300, 7))
    W, b = A.ridge_fit(Aa, Bb, 2.5)
    sk = Ridge(alpha=2.5, fit_intercept=True).fit(Aa, Bb)
    assert np.allclose(W.T, sk.coef_, atol=1e-9)
    assert np.allclose(b.ravel(), sk.intercept_, atol=1e-9)


def test_oracle_hierarchical_bootstrap_widens_with_seed_spread():
    rng = np.random.default_rng(14)
    n = 800
    dev = np.abs(rng.normal(size=n)) + 1.0
    base = (dev * 0.2, dev)
    tight = {s: (dev * (0.8 + 0.001 * s), dev) for s in range(5)}
    wide = {s: (dev * (0.8 + 0.05 * s), dev) for s in range(5)}
    a = A.hierarchical_bootstrap(tight, base, n_boot=2000)
    b = A.hierarchical_bootstrap(wide, base, n_boot=2000)
    assert b["boot_std"] > a["boot_std"]


def test_oracle_split_audit_catches_duplicate_content():
    h = np.array([1, 1, 2, 3, 4, 5, 6, 7])
    tr, te = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])
    au = A.split_audit(tr, te, h)
    assert au["index_overlap"] == 0
    assert au["duplicate_units"] == 1
    assert au["content_overlap"] == 1
