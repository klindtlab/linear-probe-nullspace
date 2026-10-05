"""Two controls for the Section 4 nondegeneracy audit.

Both turn a claim about the simulation into a measurement.

1. `forced_positive` calibrates the audit. Shifting every pre-activation positive
   makes the network exactly affine on its data, so a working audit must report one
   linear region, a constant Jacobian, and an active fraction of 1. Without this the
   audit's null result ("no seed produced a single region") is not falsifiable.

2. `affine_selector` is a constructive counterexample to an inference that is easy to
   make: that reaching reverse R^2 = 0.9956 from a lift only 0.2452-linear in
   the latents is itself proof the network cannot be affine in x. It is not. An affine
   map of a rich random feature vector can *select* the latent-decodable subspace and
   discard the rest, which is the paper's own Proposition 1. Here we build that map
   explicitly (project x onto the top-k_lat directions of the ridge-fitted x-hat,
   which is affine in x by construction) and measure its reverse R^2.

Small enough to run in-process (a 256->1024->2 MLP, 8000 training points).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from rebuttal_config import SIM
from simulation import (
    TwoLayerMLP,
    audit_network,
    build_data,
    features,
    reverse_r2,
    train_supervised,
)


def forced_positive_audit(model, x_test) -> dict:
    """Make every pre-activation positive, so the network is affine on this data."""
    with torch.no_grad():
        pre = model.preact(x_test)
        shift = float(pre.min().item())
        saved = model.fc1.bias.detach().clone()
        # +1.0 keeps it strictly positive rather than exactly at the kink
        model.fc1.bias.add_(-shift + 1.0)
        assert model.preact(x_test).min().item() > 0.0, "forcing failed"
        aud = audit_network(model, x_test)
        model.fc1.bias.copy_(saved)
    return aud


def affine_selector_reverse_r2(z_tr, x_tr, z_te, x_te, k: int) -> dict:
    """Reverse R^2 of an affine map of x that selects the z-decodable subspace.

    Ridge-fit x from z, take the top-k principal directions of the fitted values, and
    project the *raw* x onto them. The resulting feature map is affine in x and needs
    no nonlinearity, yet it can be almost perfectly linearly decoded from z.
    """
    zt = np.asarray(z_tr, dtype=np.float64)
    xt = np.asarray(x_tr, dtype=np.float64)
    ze = np.asarray(z_te, dtype=np.float64)
    xe = np.asarray(x_te, dtype=np.float64)

    zc = np.c_[zt, np.ones(len(zt))]
    beta = np.linalg.lstsq(zc, xt, rcond=None)[0]
    x_hat = zc @ beta                                  # the z-linear part of x
    # top-k directions of the z-explainable component
    u = np.linalg.svd(x_hat - x_hat.mean(0), full_matrices=False)[2][:k].T
    f_tr, f_te = xt @ u, xe @ u                        # affine in x by construction
    return dict(
        affine_selector_rev_r2=reverse_r2(zt, f_tr, ze, f_te,
                                          alpha=SIM["ridge_alpha"]),
        k_selected=k,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/simulation")
    ap.add_argument("--width", type=int, default=SIM["audit_width"])
    args = ap.parse_args()

    torch.set_num_threads(2)
    device = "cpu"
    data = build_data(device)
    k_lat = SIM["n_latent"]

    rows = []
    for seed in range(SIM["n_seeds"]):
        torch.manual_seed(seed)
        model = TwoLayerMLP(SIM["n_ambient"], args.width).to(device)
        train_supervised(model, data["x_train"], data["z_train"], weight_decay=1.0)

        trained = audit_network(model, data["x_test"])
        forced = forced_positive_audit(model, data["x_test"])
        f_tr = features(model, data["x_train"])
        f_te = features(model, data["x_test"])
        rev = reverse_r2(np.asarray(data["z_train"].cpu()), f_tr,
                         np.asarray(data["z_test"].cpu()), f_te,
                         alpha=SIM["ridge_alpha"])
        rows.append(dict(
            seed=seed, width=args.width,
            trained_n_regions=trained["n_regions"],
            trained_jacobian_cv=trained["jacobian_cv"],
            trained_relu_active_fraction=trained["relu_active_fraction"],
            trained_rev_r2=rev,
            forced_positive_n_regions=forced["n_regions"],
            forced_positive_jacobian_cv=forced["jacobian_cv"],
            forced_positive_jacobian_constant=forced["jacobian_constant"],
            forced_positive_relu_active_fraction=forced["relu_active_fraction"],
            **affine_selector_reverse_r2(
                data["z_train"].cpu(), data["x_train"].cpu(),
                data["z_test"].cpu(), data["x_test"].cpu(), k_lat),
        ))
        print(f"[control] seed={seed} trained_regions={trained['n_regions']} "
              f"forced_regions={forced['n_regions']} "
              f"forced_jac_cv={forced['jacobian_cv']:.3e} "
              f"affine_selector_rev_r2={rows[-1]['affine_selector_rev_r2']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    import csv
    path = os.path.join(args.out_dir, "simulation_controls.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(key):
        v = [float(r[key]) for r in rows]
        return dict(mean=float(np.mean(v)), min=float(np.min(v)),
                    max=float(np.max(v)))

    summary = {
        "n_seeds": SIM["n_seeds"], "width": args.width,
        "lift_linearity_reverse_r2": None,
        "audit_calibration": {
            "forced_positive_n_regions": agg("forced_positive_n_regions"),
            "forced_positive_jacobian_cv": agg("forced_positive_jacobian_cv"),
            "forced_positive_relu_active_fraction":
                agg("forced_positive_relu_active_fraction"),
            "all_seeds_single_region": all(
                r["forced_positive_n_regions"] == 1 for r in rows),
            "all_seeds_constant_jacobian": all(
                bool(r["forced_positive_jacobian_constant"]) for r in rows),
        },
        "affine_selector": {
            "k_selected": k_lat,
            "reverse_r2": agg("affine_selector_rev_r2"),
            "trained_reverse_r2": agg("trained_rev_r2"),
            "interpretation": (
                "An affine map of x reaches this reverse R^2, so a high reverse R^2 "
                "from a weakly latent-linear lift does not by itself imply the "
                "network is nonaffine. Nondegeneracy rests on the activation "
                "statistics instead."
            ),
        },
    }
    with open(os.path.join(args.out_dir, "simulation_controls.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
