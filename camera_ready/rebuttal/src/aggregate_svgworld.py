"""Aggregate the SVG-World panel, with the correct statistic for cross-style D3.

Why D3 is reported as Pearson rho rather than R^2 here. The cross-style probe is
fitted on one style and applied to the other, and the two styles' feature
distributions differ in scale and offset even after per-style centering. R^2
punishes that miscalibration without bound: measured values run to -25, which says
the prediction is badly scaled, not that the direction failed to transfer. Rho is
invariant to a per-target affine rescaling, so it measures what the diagnostic is
asking, whether the latent-aligned DIRECTION survives the style change. Equivalently,
rho^2 is the R^2 attainable after an optimal per-target affine recalibration on the
target style, and it is reported alongside.

The submitted pipeline reported rho for cross-domain transfer for exactly this
reason (its own code comment says so). This script keeps that choice, documents it,
and reports the raw R^2 too so the miscalibration is visible rather than hidden.

    python src/aggregate_svgworld.py --results results
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import rebuttal_config as C

LANE = "svgworld_pil"


def _mean_of_list(s: str) -> float:
    return float(np.mean([float(x) for x in str(s).split(";")]))


def _per_target(s: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(s).split(";")])


def d3_summary(results: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results, LANE, "metrics_d3.csv"))
    d["model"] = d["source"].str.split("|").str[0]
    d["variant"] = d["source"].str.split("|").str[1]
    d["direction"] = (d["source"].str.split("|").str[2] + "->"
                      + d["target"].str.split("|").str[2])
    d["rho_mean"] = d["rho_per_target"].map(_mean_of_list)
    g = d.groupby(["model", "variant", "block"], as_index=False).agg(
        d3_rho=("rho_mean", "mean"), d3_rho_sd=("rho_mean", "std"),
        d3_r2_uncalibrated=("r2_mean", "mean"),
        n_measurements=("rho_mean", "size"))
    g["d3_rho_squared"] = g["d3_rho"] ** 2
    return g


def d1_summary(results: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results, LANE, "metrics_d1.csv"))
    parts = d["condition"].str.split("|", expand=True)
    d["model"], d["variant"], d["style"] = parts[0], parts[1], parts[2]
    return d.groupby(["model", "variant", "block"], as_index=False).agg(
        d1_raw=("d1_raw", "mean"), d1_raw_sd=("d1_raw", "std"),
        ceiling=("ceiling", "mean"),
        d1_normalized=("d1_normalized", "mean"),
        target_rank=("target_rank", "first"), width=("width", "first"),
        n_measurements=("d1_raw", "size"))


def interval_summary(results: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results, LANE, "intervals.csv"))
    parts = d["statistic"].str.split("|", expand=True)
    d["diagnostic"], d["model"] = parts[0], parts[1]
    d["block"] = parts[3].fillna(parts[2])
    return d


def build(results: str) -> pd.DataFrame:
    d1 = d1_summary(results)
    d3 = d3_summary(results)
    ci = interval_summary(results)

    fwd = ci[ci["diagnostic"] == "forward"].groupby(
        ["model", "block"], as_index=False).agg(
        forward_gap=("diff", "mean"), forward_gap_lo=("diff_ci_lo", "mean"),
        forward_gap_hi=("diff_ci_hi", "mean"),
        forward_pre=("pretrained", "mean"), forward_rand=("random", "mean"))
    d1ci = ci[ci["diagnostic"] == "D1_raw"].groupby(
        ["model", "block"], as_index=False).agg(
        d1_gap=("diff", "mean"), d1_gap_lo=("diff_ci_lo", "mean"),
        d1_gap_hi=("diff_ci_hi", "mean"),
        d1_excludes_zero=("diff_excludes_zero", "all"),
        n_units=("n_units", "first"))

    wide = {}
    for (m, b), sub in d1.groupby(["model", "block"]):
        row = dict(model=m, block=b, width=int(sub["width"].iloc[0]),
                   target_rank=int(sub["target_rank"].iloc[0]))
        for v in ("pre", "rand"):
            s = sub[sub["variant"] == v]
            if len(s):
                row[f"d1_{v}"] = float(s["d1_raw"].iloc[0])
                row[f"ceiling_{v}"] = float(s["ceiling"].iloc[0])
                row[f"d1_norm_{v}"] = float(s["d1_normalized"].iloc[0])
        wide[(m, b)] = row
    for (m, b), sub in d3.groupby(["model", "block"]):
        row = wide.setdefault((m, b), dict(model=m, block=b))
        for v in ("pre", "rand"):
            s = sub[sub["variant"] == v]
            if len(s):
                row[f"d3_rho_{v}"] = float(s["d3_rho"].iloc[0])
                row[f"d3_rho_sd_{v}"] = float(s["d3_rho_sd"].iloc[0])
                row[f"d3_rho2_{v}"] = float(s["d3_rho_squared"].iloc[0])
                row[f"d3_r2_uncal_{v}"] = float(s["d3_r2_uncalibrated"].iloc[0])
    out = pd.DataFrame(list(wide.values()))
    out = out.merge(fwd, on=["model", "block"], how="left")
    out = out.merge(d1ci, on=["model", "block"], how="left")
    out["d3_rho_gap"] = out["d3_rho_pre"] - out["d3_rho_rand"]
    out["d1_norm_gap"] = out["d1_norm_pre"] - out["d1_norm_rand"]
    out["model_label"] = out["model"].map(lambda m: C.MODELS[m]["label"])
    out["paradigm"] = out["model"].map(lambda m: C.MODELS[m]["paradigm"])
    order = list(C.SVGWORLD_MODELS)
    out["order"] = out["model"].map(order.index)
    return out.sort_values(["order", "block"]).drop(columns="order")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = build(args.results)
    p = os.path.join(args.results, "svgworld_summary.csv")
    out.round(6).to_csv(p, index=False)
    print(f"[write] {p} ({len(out)} rows)")
    cols = ["model_label", "block", "width", "d1_pre", "d1_rand", "d1_gap",
            "ceiling_pre", "d1_norm_pre", "d1_norm_rand",
            "forward_pre", "forward_rand", "d3_rho_pre", "d3_rho_rand",
            "d3_rho_gap", "d3_r2_uncal_pre"]
    pd.set_option("display.width", 240)
    print(out[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
