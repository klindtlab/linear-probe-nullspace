"""Does the corrected D2 definition rescue the simulation panels that S(k) breaks?

The prior run found that under one consistent definition of the spectral
statistic, S(k) = reverse R^2 after train-fitted PCA whitening, the random
initialization reaches S(k_lat) = 0.71 in the Section 4 simulation, so the
"spectral concentration" panel no longer separates trained from random. The
oracle tests in `tests/test_oracles.py` show why: S(k) decays for a strict linear
world model too, and rises for a residual-dominated map once whitening equalizes
low-variance directions.

This script recomputes the simulation's D1 and D2 with both statistics on the
same features, across seeds, for the three conditions the paper compares. It is
CPU-only and small: the simulation is a two-layer MLP on 256-dimensional lifted
inputs.

    python src/sim_definitions.py --seeds 5 --width 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "prior"))

import audit_calc as AC                                    # noqa: E402
import rebuttal_config as C                                # noqa: E402
import simulation as SIM                                   # noqa: E402

KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dat = SIM.build_data("cpu")
    x_tr, x_te = dat["x_train"], dat["x_test"]
    z_tr = dat["z_train"].numpy().astype(np.float64)
    z_te = dat["z_test"].numpy().astype(np.float64)
    k_lat = z_tr.shape[1]
    rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        for cond, wd in (("random", None), ("trained_wd1", 1.0),
                         ("trained_wd0", 0.0)):
            torch.manual_seed(seed)
            model = SIM.TwoLayerMLP(C.SIM["n_ambient"], args.width)
            if wd is not None:
                model, _readout, _loss = SIM.train_supervised(
                    model, x_tr, dat["z_train"], weight_decay=wd)
            f_tr = SIM.features(model, x_tr).astype(np.float64)
            f_te = SIM.features(model, x_te).astype(np.float64)
            alpha = C.SIM["ridge_alpha"]
            d1 = AC.d1(z_tr, f_tr, z_te, f_te, alpha=alpha)
            S = AC.d2_S_curve(z_tr, f_tr, z_te, f_te, KS, alpha=alpha)
            Cc = AC.d2_C_curve(z_tr, f_tr, z_te, f_te, KS, alpha=alpha)
            fwd = AC.forward_probe(f_tr, z_tr, f_te, z_te, alpha=alpha)
            row = dict(seed=seed, condition=cond, width=args.width,
                       forward_r2=fwd["r2_aggregate"], d1_raw=d1["d1_raw"],
                       d1_ceiling=d1["ceiling"], d1_normalized=d1["d1_normalized"],
                       S_at_k_lat=S["curve"][k_lat]["S"],
                       S_at_full=S["curve"][max(S["curve"])]["S"],
                       C_at_k_lat=Cc["curve"][k_lat], C_k50=Cc["k50"],
                       C_k90=Cc["k90"], C_k99=Cc["k99"],
                       participation_ratio=d1["participation_ratio"])
            for k in KS:
                if k in S["curve"]:
                    row[f"S_k{k}"] = S["curve"][k]["S"]
                if k in Cc["curve"]:
                    row[f"C_k{k}"] = Cc["curve"][k]
            rows.append(row)
            print(f"[sim] seed={seed} {cond:<12} fwd={row['forward_r2']:.4f} "
                  f"D1={row['d1_raw']:.4f} S(k_lat)={row['S_at_k_lat']:.4f} "
                  f"C(k_lat)={row['C_at_k_lat']:.4f} k90={row['C_k90']}",
                  flush=True)

    out = args.out or os.path.join(os.environ.get(
        "EXPERIMENT_DIR", "."), "results", "sim_definitions.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv

    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in rows[0], k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {}
    for cond in ("random", "trained_wd1", "trained_wd0"):
        sel = [r for r in rows if r["condition"] == cond]
        summary[cond] = {k: dict(mean=float(np.mean([r[k] for r in sel])),
                                 std=float(np.std([r[k] for r in sel])),
                                 min=float(np.min([r[k] for r in sel])),
                                 max=float(np.max([r[k] for r in sel])))
                         for k in ("forward_r2", "d1_raw", "d1_normalized",
                                   "S_at_k_lat", "C_at_k_lat", "C_k90")}
    js = out.replace(".csv", ".json")
    with open(js, "w") as f:
        json.dump(dict(width=args.width, n_seeds=args.seeds, k_lat=k_lat,
                       seconds=round(time.time() - t0, 1), summary=summary), f,
                  indent=1)
    print(f"[write] {out}\n[write] {js}")
    for cond, s in summary.items():
        print(f"  {cond:<12} D1={s['d1_raw']['mean']:.4f} "
              f"S(k_lat)={s['S_at_k_lat']['mean']:.4f} "
              f"C(k_lat)={s['C_at_k_lat']['mean']:.4f}")


if __name__ == "__main__":
    main()
