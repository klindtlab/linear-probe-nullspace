"""Unit tests for the ridge path and the three diagnostics, on toy fixtures.

The point of these is that a diagnostic must separate the two regimes it claims
to distinguish BEFORE it is pointed at a real encoder: planted rank-r linear
structure should score near the ceiling, and an unstructured feature map near
zero. Run with `python -m pytest tests -q` from the experiment directory.
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

import diagnostics as D
import factors as FA
import fast_ridge as FR
import rebuttal_config as C


def _planted(n=1200, r=4, d_signal=50, d_noise=50, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.uniform(0, 1, (n, r))
    X = np.concatenate([Y @ rng.normal(size=(r, d_signal)),
                        noise * rng.normal(size=(n, d_noise))], axis=1)
    return Y, X, rng.normal(size=X.shape)


@pytest.mark.parametrize("alpha", [1e-3, 1.0, 100.0])
def test_ridge_path_matches_sklearn(alpha):
    Y, X, _ = _planted()
    ref = Ridge(alpha=alpha, fit_intercept=True).fit(X, Y).predict(X[:64])
    got = FR.RidgePath(X, Y).predict(X[:64], alpha)
    assert np.abs(ref - got).max() < 1e-9


def test_select_alpha_stays_inside_grid():
    Y, X, _ = _planted()
    a, scores = FR.select_alpha(X, Y)
    assert a in C.RIDGE_ALPHAS and len(scores) == len(C.RIDGE_ALPHAS)


def test_d1_separates_planted_from_random():
    Y, X, Xr = _planted()
    tr, te = D.unit_split(len(Y), seed=0)
    good = D.d1_reverse(Y[tr], X[tr], Y[te], X[te])
    bad = D.d1_reverse(Y[tr], Xr[tr], Y[te], Xr[te])
    assert good["d1_raw"] > 0.7 and abs(bad["d1_raw"]) < 0.05
    # D1 can never exceed its Ky Fan ceiling by more than float noise.
    assert good["d1_raw"] <= good["ceiling"] + 1e-6
    assert 0.9 < good["d1_normalized"] <= 1.0 + 1e-6
    assert good["target_rank"] == Y.shape[1]


def test_d2_curve_signature():
    """A rank-r linear image gives S(k)=1 up to k=r then decays under whitening;
    an unstructured map is flat near zero. This is the Section 3.3 prediction."""
    Y, X, Xr = _planted()
    tr, te = D.unit_split(len(Y), seed=0)
    good = D.d2_curve(Y[tr], X[tr], Y[te], X[te], ks=(1, 2, 4, 8, 16, 32))
    bad = D.d2_curve(Y[tr], Xr[tr], Y[te], Xr[te], ks=(1, 2, 4, 8, 16, 32))
    assert good["r_eff"] == Y.shape[1]
    assert good["S_at_r_eff"] > 0.95
    assert good["curve"][32]["S"] < good["curve"][4]["S"]
    assert max(abs(v["S"]) for v in bad["curve"].values()) < 0.05


def test_paired_bootstrap_excludes_zero_only_when_it_should():
    Y, X, Xr = _planted()
    tr, te = D.unit_split(len(Y), seed=0)
    a = D.d1_reverse(Y[tr], X[tr], Y[te], X[te])
    b = D.d1_reverse(Y[tr], Xr[tr], Y[te], Xr[te])
    real = D.bootstrap_ci_chunked(a["terms"], b["terms"], n_boot=2000)
    assert real["excludes_zero"] and real["ci_lo"] > 0
    null = D.bootstrap_ci_chunked(b["terms"], b["terms"], n_boot=2000)
    assert not null["excludes_zero"]


def test_split_audit_detects_content_overlap():
    h = np.arange(100)
    tr, te = D.unit_split(100, seed=0)
    assert D.audit_split(tr, te, h)["content_overlap"] == 0
    leaky = np.zeros(100, dtype=int)
    assert D.audit_split(tr, te, leaky)["content_overlap"] > 0


def test_checkerboard_keeps_every_marginal_and_holds_out_combinations():
    spec = C.DATASETS["dsprites"]["factors"]
    nv = [f["n_values"] for f in spec]
    idx = FA.sample_factor_balanced(nv, 3000)
    F = FA.index_to_factors(idx, nv)
    assert np.array_equal(FA.factors_to_index(F, nv), idx)
    tr, te, info = FA.checkerboard_split(F, spec)
    assert info["every_marginal_in_train"]
    assert info["combination_overlap"] == 0
    assert info["n_train"] > 0 and info["n_test"] > 0


def test_cyclic_targets_are_encoded_as_sin_cos():
    spec = C.DATASETS["dsprites"]["factors"]
    nv = [f["n_values"] for f in spec]
    F = FA.index_to_factors(FA.sample_factor_balanced(nv, 500), nv)
    Y, labels, Ycat, cat_labels = FA.encode_targets(F, spec)
    assert "orientation_sin" in labels and "orientation_cos" in labels
    assert "shape" not in labels                    # categorical -> appendix
    assert Ycat.shape[1] == 3 and len(cat_labels) == 3
    j = labels.index("orientation_sin")
    assert np.abs(Y[:, j] ** 2 + Y[:, j + 1] ** 2 - 1).max() < 1e-9


def test_simulation_lift_is_only_weakly_linear_in_the_latents():
    """The frozen Fourier lift is the fact that answers the affine-degeneration
    objection: the MLP's input is only ~0.245-linear in z, so a network that were
    affine in its input could not be affine in z."""
    import simulation as S

    d = S.build_data("cpu")
    z_tr, z_te = d["z_train"].numpy(), d["z_test"].numpy()
    x_tr, x_te = d["x_train"].numpy(), d["x_test"].numpy()
    rev = S.reverse_r2(z_tr, x_tr, z_te, x_te)
    assert abs(rev - 0.2452) < 1e-3, rev
    fwd, _ = S.forward_r2(x_tr, z_tr, x_te, z_te)
    # Proposition 1 in miniature: z is perfectly linearly readable FROM the lift.
    assert fwd.min() > 0.99


def test_simulation_audit_flags_an_affine_network():
    """A network with no active ReLU gating must look affine to the audit."""
    import torch

    import simulation as S

    d = S.build_data("cpu")
    torch.manual_seed(0)
    m = S.TwoLayerMLP(C.SIM["n_ambient"], 32)
    with torch.no_grad():   # force every pre-activation positive -> ReLU = identity
        m.fc1.bias.fill_(1e4)
    aud = S.audit_network(m, d["x_test"])
    assert aud["n_regions"] == 1
    assert aud["jacobian_constant"] and aud["relu_active_fraction"] == 1.0


def test_dsprites_orientation_period_matches_the_stored_angle_grid():
    """The cyclic period must come from the dataset, not from our own formula.

    dSprites stores orientation as linspace(0, 2*pi, 40) INCLUSIVE, so the spacing
    is 2*pi/39 and class 39 is exactly 2*pi, the same angle as class 0. An earlier
    version declared period 40, which placed the last bin at 351 degrees instead of
    360. This test pins the declared period to what the grid implies and pins the
    exclusion of the duplicated bin.
    """
    spec = [f for f in C.DATASETS["dsprites"]["factors"]
            if f["name"] == "orientation"][0]
    n_stored = spec["n_values"]
    # A closed grid of n points spanning a full turn has spacing 2*pi/(n-1).
    assert spec["period"] == n_stored - 1 == 39
    assert spec["exclude_values"] == [39]
    ex = FA.exclusions_from_spec(C.DATASETS["dsprites"]["factors"])
    nv = [f["n_values"] for f in C.DATASETS["dsprites"]["factors"]]
    F = FA.index_to_factors(FA.sample_factor_balanced(nv, 2000, exclude=ex), nv)
    assert 39 not in set(F[:, 2].tolist())
    Y, labels, _, _ = FA.encode_targets(F, C.DATASETS["dsprites"]["factors"])
    j = labels.index("orientation_sin")
    ang = np.arctan2(Y[:, j], Y[:, j + 1]) % (2 * np.pi)
    # every encoded angle must land on a multiple of 2*pi/39
    resid = np.abs(((ang / (2 * np.pi / 39)) + 0.5) % 1.0 - 0.5)
    assert resid.max() < 1e-9


def test_mpi3d_axes_are_ordinal_not_cyclic():
    """Both MPI3D axes are robot degrees of freedom, so neither wraps around."""
    kinds = {f["name"]: f["kind"]
             for f in C.DATASETS["mpi3d_realistic"]["factors"]}
    assert kinds["horizontal_axis"] == "linear"
    assert kinds["vertical_axis"] == "linear"
    _, labels, _, _ = FA.encode_targets(
        FA.index_to_factors(np.arange(64), [6, 6, 2, 3, 3, 40, 40]),
        C.DATASETS["mpi3d_realistic"]["factors"])
    assert "horizontal_axis" in labels
    assert "horizontal_axis_sin" not in labels
