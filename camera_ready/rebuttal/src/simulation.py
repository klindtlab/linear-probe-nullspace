"""Section 4 controlled simulation: sweep, integrity audit, and eigenspectra.

Reproduces `0_simulation.ipynb` verbatim in setup and training, then adds the
three things the rebuttal needs:

1. A nondegeneracy audit (distinct ReLU regions on the data, gating fraction,
   dead units, active fraction, Jacobian variation) so the claim that the
   trained network is not an affine map is measured rather than asserted.
2. D2 computed with ONE definition, the paper's Section 3.3 statistic:
   reverse R^2 after PCA-whitening the features on the training half. The
   notebook's `spectral_concentration` is retained under its own name so the
   conclusion can be shown to survive the alternative choice.
3. Eigenvalues of the feature covariance on a linear scale, which shows a
   rank-2 spectrum rather than a bug.

The pipeline is: z ~ Unif([-1,1]^2) -> FIXED frozen Fourier lift
x = cos(z W_lift + b_lift) in R^256 -> TwoLayerMLP -> features f in R^h, with a
linear readout U trained jointly. The lift is the detail whose omission from
Section 4 produced the affine-degeneration objection: the MLP never sees the
latent square.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rebuttal_config as C
from fast_ridge import RidgePath

SIM = C.SIM


# ---------------------------------------------------------------------------
# Data: latents and the frozen Fourier lift
# ---------------------------------------------------------------------------
def build_data(device: str):
    """Latents and the frozen lift.

    Generated on CPU and then moved to `device`: CUDA and CPU RNG streams differ,
    and the lift is the fixed dataset of this simulation, not a nuisance seed. On
    CPU-generated data the lift's linearity is 0.2452, the value the recovered
    audit reports, on any device.
    """
    n_lat, n_amb = SIM["n_latent"], SIM["n_ambient"]

    def sample_z(n, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.empty(n, n_lat).uniform_(-1.0, 1.0, generator=g).to(device)

    gen = torch.Generator(device="cpu").manual_seed(SIM["lift_seed"])
    W_lift = (torch.randn(n_lat, n_amb, generator=gen) * SIM["lift_scale"]).to(device)
    b_lift = (torch.rand(n_amb, generator=gen) * 2 * np.pi).to(device)

    def lift(z):
        return torch.cos(z @ W_lift + b_lift)

    z_train = sample_z(SIM["n_train"], SIM["z_train_seed"])
    z_test = sample_z(SIM["n_test"], SIM["z_test_seed"])
    return dict(
        z_train=z_train,
        z_test=z_test,
        x_train=lift(z_train).detach(),
        x_test=lift(z_test).detach(),
        W_lift=W_lift,
        b_lift=b_lift,
    )


class TwoLayerMLP(nn.Module):
    """f(x) = W2 ReLU(W1 x + b1) + b2. Output is signed, dim = n_hidden."""

    def __init__(self, n_in: int, n_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def preact(self, x):
        return self.fc1(x)


def train_supervised(model, x_tr, z_tr, weight_decay: float, n_steps: int | None = None):
    n_steps = SIM["n_steps"] if n_steps is None else n_steps
    device = x_tr.device
    with torch.no_grad():
        d = model(x_tr[:1]).shape[-1]
    readout = nn.Linear(d, SIM["n_latent"]).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(readout.parameters()),
        lr=SIM["lr"],
        weight_decay=weight_decay,
    )
    n = x_tr.shape[0]
    for _ in range(n_steps):
        idx = torch.randint(0, n, (SIM["batch_size"],), device=device)
        loss = F.mse_loss(readout(model(x_tr[idx])), z_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, readout, float(loss.item())


def features(model, x) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics. One definition each, shared with the SVG-World side.
# ---------------------------------------------------------------------------
def forward_r2(f_tr, z_tr, f_te, z_te, alpha=None):
    """Per-latent forward R^2: predict the latents FROM the features."""
    alpha = SIM["ridge_alpha"] if alpha is None else alpha
    path = RidgePath(f_tr, z_tr)
    z_hat = path.predict(f_te, alpha)
    num = ((z_te - z_hat) ** 2).sum(axis=0)
    den = ((z_te - z_te.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - num / den, path.coef(alpha).T


def reverse_r2(z_tr, f_tr, z_te, f_te, alpha=None) -> float:
    """D1. Share of total held-out feature variance linearly explained by z.

    Note the argument order: the REGRESSOR is z and the TARGET is f, so the
    prediction is made from z_te.
    """
    alpha = SIM["ridge_alpha"] if alpha is None else alpha
    f_hat = RidgePath(z_tr, f_tr).predict(z_te, alpha)
    sse = ((f_te - f_hat) ** 2).sum()
    sst = ((f_te - f_te.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1 - sse / sst)


def pca_whiten_fit(f_tr: np.ndarray, k_max: int | None = None):
    """PCA + whitening fitted on the training half only.

    Returns a callable mapping features to whitened top-k coordinates and the
    eigenvalues of the training feature covariance.
    """
    mu = f_tr.mean(axis=0, keepdims=True)
    fc = f_tr - mu
    G = fc.T @ fc
    w, V = np.linalg.eigh(0.5 * (G + G.T))
    order = np.argsort(w)[::-1]
    w, Vt = np.maximum(w[order], 0.0), V[:, order].T
    n = fc.shape[0]
    eig = w / max(n - 1, 1)
    keep = int(np.sum(eig > eig[0] * 1e-12)) if eig[0] > 0 else 1
    if k_max is not None:
        keep = min(keep, k_max)
    keep = max(keep, 1)
    scale = 1.0 / np.sqrt(np.maximum(eig[:keep], 1e-12))

    def transform(f: np.ndarray, k: int) -> np.ndarray:
        k = min(k, keep)
        return ((f - mu) @ Vt[:k].T) * scale[:k]

    return transform, eig, keep


def d2_whitened_curve(z_tr, f_tr, z_te, f_te, ks, alpha=None):
    """D2 as printed in Section 3.3: reverse R^2 after PCA-whitening to top-k.

    Whitening removes all variance-concentration structure inside the retained
    subspace, so S(k) rising early means the latent-aligned directions ARE the
    leading principal directions.
    """
    transform, eig, keep = pca_whiten_fit(f_tr)
    out = {}
    for k in ks:
        if k > keep:
            continue
        out[int(k)] = reverse_r2(z_tr, transform(f_tr, k), z_te, transform(f_te, k),
                                 alpha=alpha)
    return out, eig, keep


def spectral_concentration_legacy(f_tr, z_tr, f_te, z_te, k_latent, alpha=None):
    """The notebook's `spectral_concentration`, kept verbatim for comparison.

    Fraction of TOTAL held-out feature variance that is both z-explainable and
    inside the top-k_latent principal subspace. Not whitened, and bounded above
    by the top-k variance share, which is why it can approach 1 only when the
    spectrum itself collapses to rank k_latent.
    """
    alpha = SIM["ridge_alpha"] if alpha is None else alpha
    mu = f_tr.mean(axis=0, keepdims=True)
    fc_tr, fc_te = f_tr - mu, f_te - mu
    _, _, Vt = np.linalg.svd(fc_tr, full_matrices=False)
    P = Vt[:k_latent].T
    f_hat = RidgePath(z_tr, fc_tr @ P).predict(z_te, alpha)
    return float(((f_hat - f_hat.mean(0)) ** 2).sum() / (fc_te ** 2).sum())


def variance_ceiling(f_te: np.ndarray, r: int) -> float:
    """Top-r eigenvalue share of the held-out feature covariance.

    A loose upper bound (Ky Fan) on what ANY rank-r linear structure could
    explain of the total feature variance, used to normalize D1.
    """
    fc = f_te - f_te.mean(axis=0, keepdims=True)
    s = np.linalg.svd(fc, compute_uv=False)
    ev = s ** 2
    r = min(r, len(ev))
    return float(ev[:r].sum() / ev.sum())


# ---------------------------------------------------------------------------
# Integrity audit: is the trained network affine on the data?
# ---------------------------------------------------------------------------
def audit_network(model, x_test, n_jac=200, jac_seed=0) -> dict:
    model.eval()
    with torch.no_grad():
        pre = model.preact(x_test).float().cpu().numpy()
    pat = pre > 0
    rate = pat.mean(axis=0)

    W1 = model.fc1.weight.detach().float().cpu().numpy()
    W2 = model.fc2.weight.detach().float().cpu().numpy()
    rng = np.random.default_rng(jac_seed)
    v = rng.normal(size=(W1.shape[1],))
    v /= np.linalg.norm(v)
    W1v = W1 @ v
    n_jac = min(n_jac, pat.shape[0])
    Jv = (pat[:n_jac] * W1v) @ W2.T
    jac_cv = float(Jv.std(axis=0).mean() / (np.abs(Jv).mean() + 1e-12))

    return dict(
        n_test=int(pat.shape[0]),
        n_regions=int(len(np.unique(pat, axis=0))),
        frac_gating=float(((rate > 0.05) & (rate < 0.95)).mean()),
        frac_dead=float((rate < 0.01).mean()),
        relu_active_fraction=float(pat.mean()),
        jacobian_cv=jac_cv,
        jacobian_constant=bool(jac_cv < 1e-6),
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(device: str, widths, weight_decays, n_seeds, n_steps=None,
              out_dir=None, verbose=True):
    data = build_data(device)
    x_tr, x_te = data["x_train"], data["x_test"]
    z_tr, z_te = data["z_train"], data["z_test"]
    z_tr_np, z_te_np = z_tr.float().cpu().numpy(), z_te.float().cpu().numpy()

    # OOD (D3 in the simulation): radial split, train inside ||z|| < 0.8.
    thr = SIM["ood_norm_threshold"]
    m_tr_in = (z_tr.norm(dim=1) < thr).cpu().numpy()
    m_te_out = (z_te.norm(dim=1) >= thr).cpu().numpy()

    # How linear the FROZEN lift itself is in the latents. This is the number
    # that answers the affine-degeneration objection directly: the MLP's input is
    # only weakly linear in z, so a network that is affine in x could not be
    # affine in z, yet the trained network reaches reverse R^2 near 1.
    x_tr_np = data["x_train"].float().cpu().numpy()
    x_te_np = data["x_test"].float().cpu().numpy()
    lift_linearity = reverse_r2(z_tr_np, x_tr_np, z_te_np, x_te_np)
    lift_forward, _ = forward_r2(x_tr_np, z_tr_np, x_te_np, z_te_np)

    k_lat = SIM["n_latent"]
    rows, audits, spectra, d2_curves = [], [], [], []

    def evaluate(f_tr, f_te):
        r2_fwd, _ = forward_r2(f_tr, z_tr_np, f_te, z_te_np)
        rev = reverse_r2(z_tr_np, f_tr, z_te_np, f_te)
        d2_pt, eig, keep = d2_whitened_curve(z_tr_np, f_tr, z_te_np, f_te, [k_lat])
        legacy = spectral_concentration_legacy(f_tr, z_tr_np, f_te, z_te_np, k_lat)
        ceil = variance_ceiling(f_te, k_lat)
        r2_ood, _ = forward_r2(f_tr[m_tr_in], z_tr_np[m_tr_in],
                               f_te[m_te_out], z_te_np[m_te_out])
        return dict(
            fwd_r2=float(r2_fwd.mean()),
            rev_r2=rev,
            d2_whitened_at_klat=float(d2_pt.get(k_lat, np.nan)),
            spectral_concentration_legacy=legacy,
            d1_ceiling=ceil,
            d1_normalized=float(rev / ceil) if ceil > 0 else float("nan"),
            ood_r2=float(r2_ood.mean()),
        ), eig

    conditions = [("random", None)] + [("trained", wd) for wd in weight_decays]
    t_start = time.time()
    total = n_seeds * len(conditions) * len(widths)
    done = 0
    for seed in range(n_seeds):
        for cond, wd in conditions:
            for width in widths:
                torch.manual_seed(seed)
                model = TwoLayerMLP(SIM["n_ambient"], width).to(device)
                final_loss = float("nan")
                if cond == "trained":
                    model, _, final_loss = train_supervised(
                        model, x_tr, z_tr, weight_decay=wd, n_steps=n_steps)
                f_tr, f_te = features(model, x_tr), features(model, x_te)
                metrics, eig = evaluate(f_tr, f_te)
                aud = audit_network(model, x_te)
                rows.append(dict(seed=seed, cond=cond, wd=wd, width=width,
                                 final_train_loss=final_loss, **metrics, **aud))
                audits.append(dict(seed=seed, cond=cond, wd=wd, width=width, **aud))
                n_rep = SIM["n_eigenvalues_reported"]
                spectra.append(dict(
                    seed=seed, cond=cond, wd=wd, width=width,
                    eigenvalues=[float(v) for v in eig[:n_rep]],
                    eig_total=float(eig.sum()),
                ))
                if width == SIM["audit_width"]:
                    curve, _, _ = d2_whitened_curve(
                        z_tr_np, f_tr, z_te_np, f_te,
                        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024])
                    d2_curves.append(dict(seed=seed, cond=cond, wd=wd, width=width,
                                          curve={str(k): v for k, v in curve.items()}))
                done += 1
                if verbose:
                    el = time.time() - t_start
                    print(f"[sim {done}/{total}] seed={seed} {cond} wd={wd} h={width} "
                          f"fwd={metrics['fwd_r2']:+.4f} rev={metrics['rev_r2']:+.4f} "
                          f"regions={aud['n_regions']}/{aud['n_test']} "
                          f"active={aud['relu_active_fraction']:.3f} "
                          f"jac_cv={aud['jacobian_cv']:.3f} "
                          f"({el:.0f}s elapsed)", flush=True)

    out = dict(
        config={k: (list(v) if isinstance(v, tuple) else v) for k, v in SIM.items()},
        widths=list(widths), weight_decays=list(weight_decays), n_seeds=n_seeds,
        n_steps=SIM["n_steps"] if n_steps is None else n_steps,
        lift_linearity_reverse_r2=lift_linearity,
        lift_forward_r2_per_latent=[float(v) for v in lift_forward],
        rows=rows, spectra=spectra, d2_curves=d2_curves,
        wall_seconds=round(time.time() - t_start, 1),
        device=device,
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "simulation_sweep.json"), "w") as f:
            json.dump(out, f, indent=1)
        _write_csv(os.path.join(out_dir, "simulation_sweep.csv"), rows)
        print(f"wrote {out_dir}/simulation_sweep.{{json,csv}}", flush=True)
    return out


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")


def summarize_audit(out: dict, out_dir: str | None = None) -> dict:
    """The nondegeneracy table the rebuttal quotes, aggregated over seeds.

    A network that is affine on the data has exactly one linear region, a
    constant Jacobian, and no gating units. Reporting mean and range over seeds
    (rather than one seed's numbers) is what makes the claim a measurement.
    """
    rows = out["rows"]
    w_audit = SIM["audit_width"]

    def agg(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None]
        if not v:
            return None
        return dict(mean=float(np.mean(v)), min=float(np.min(v)),
                    max=float(np.max(v)), n=len(v))

    keys = ["n_regions", "frac_gating", "frac_dead", "relu_active_fraction",
            "jacobian_cv", "rev_r2", "fwd_r2", "d2_whitened_at_klat",
            "d1_ceiling", "ood_r2", "spectral_concentration_legacy"]
    conditions = {}
    for label, pred in (
        ("trained_wd1", lambda r: r["cond"] == "trained" and r["wd"] == 1.0),
        ("trained_wd0", lambda r: r["cond"] == "trained" and r["wd"] == 0.0),
        ("random_init", lambda r: r["cond"] == "random"),
    ):
        sel = [r for r in rows if pred(r) and r["width"] == w_audit]
        conditions[label] = {k: agg(sel, k) for k in keys}
        conditions[label]["n_test_points"] = (sel[0]["n_test"] if sel else None)
        conditions[label]["any_single_region"] = bool(
            any(r["n_regions"] <= 1 for r in sel))
        conditions[label]["any_constant_jacobian"] = bool(
            any(r["jacobian_constant"] for r in sel))

    by_width = []
    for w in sorted({r["width"] for r in rows}):
        for label, pred in (
            ("trained_wd1", lambda r: r["cond"] == "trained" and r["wd"] == 1.0),
            ("trained_wd0", lambda r: r["cond"] == "trained" and r["wd"] == 0.0),
            ("random_init", lambda r: r["cond"] == "random"),
        ):
            sel = [r for r in rows if pred(r) and r["width"] == w]
            if not sel:
                continue
            row = dict(width=w, condition=label, n_seeds=len(sel))
            for k in keys:
                a = agg(sel, k)
                if a:
                    row[f"{k}_mean"] = round(a["mean"], 6)
                    row[f"{k}_min"] = round(a["min"], 6)
                    row[f"{k}_max"] = round(a["max"], 6)
            by_width.append(row)

    summary = dict(
        audit_width=w_audit,
        lift_linearity_reverse_r2=out["lift_linearity_reverse_r2"],
        conditions=conditions,
        nondegenerate=bool(
            not conditions["trained_wd1"]["any_single_region"]
            and not conditions["trained_wd1"]["any_constant_jacobian"]),
        n_seeds=out["n_seeds"], widths=out["widths"],
    )
    if out_dir:
        _write_csv(os.path.join(out_dir, "simulation_audit_by_width.csv"), by_width)
        with open(os.path.join(out_dir, "simulation_audit_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        spec_rows = []
        for sp in out["spectra"]:
            if sp["width"] != w_audit:
                continue
            for i, v in enumerate(sp["eigenvalues"], start=1):
                spec_rows.append(dict(seed=sp["seed"], cond=sp["cond"], wd=sp["wd"],
                                      width=sp["width"], pc_index=i, eigenvalue=v,
                                      eigenvalue_share=v / max(sp["eig_total"], 1e-30)))
        _write_csv(os.path.join(out_dir, "simulation_eigenvalues.csv"), spec_rows)
        curve_rows = []
        for c in out["d2_curves"]:
            for k, v in c["curve"].items():
                curve_rows.append(dict(seed=c["seed"], cond=c["cond"], wd=c["wd"],
                                       width=c["width"], k=int(k), S=v))
        _write_csv(os.path.join(out_dir, "simulation_d2_curves.csv"), curve_rows)
    return summary
