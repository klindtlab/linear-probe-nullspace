"""Write each figure bundle's committed plot-ready data from the lane tables.

Run on CPU once the lanes' CSVs are in `results/`. Keeps every
bundle re-renderable from its own small `data.json` without rerunning anything.

    python figures/build_figure_data.py --results results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

HERE = os.path.dirname(os.path.abspath(__file__))
COND = {"random": "random_init"}


def _cond(row) -> str:
    if row["cond"] == "random":
        return "random_init"
    return "trained_wd1" if float(row["wd"]) == 1.0 else "trained_wd0"


def write(bundle: str, payload) -> None:
    d = os.path.join(HERE, bundle)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "data.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[data] {p} ({os.path.getsize(p) / 1e3:.1f} KB)")


def simulation_bundles(results: str) -> None:
    src = os.path.join(results, "simulation")

    by_width = os.path.join(src, "simulation_audit_by_width.csv")
    if os.path.exists(by_width):
        df = pd.read_csv(by_width)
        keep = ["width", "condition", "n_seeds"]
        for base in ["n_regions", "jacobian_cv", "relu_active_fraction",
                     "frac_gating", "frac_dead", "rev_r2", "fwd_r2",
                     "d2_whitened_at_klat", "ood_r2", "d1_ceiling",
                     "spectral_concentration_legacy"]:
            keep += [c for c in (f"{base}_mean", f"{base}_min", f"{base}_max")
                     if c in df.columns]
        sub = df[keep].round(6)
        write("simulation_integrity", sub.to_dict(orient="records"))

        # The four-panel diagnostics figure needs a spread, not a min/max band.
        diag = sub.copy()
        for base in ["fwd_r2", "rev_r2", "d2_whitened_at_klat", "ood_r2"]:
            m, lo, hi = f"{base}_mean", f"{base}_min", f"{base}_max"
            if m in diag:
                diag[f"{base}_sd"] = ((diag[hi] - diag[lo]) / 2).round(6)
        cols = ["width", "condition", "n_seeds"] + [
            c for base in ["fwd_r2", "rev_r2", "d2_whitened_at_klat", "ood_r2"]
            for c in (f"{base}_mean", f"{base}_sd") if c in diag]
        write("simulation_diagnostics", diag[cols].to_dict(orient="records"))

    eig = os.path.join(src, "simulation_eigenvalues.csv")
    if os.path.exists(eig):
        df = pd.read_csv(eig)
        df["condition"] = df.apply(_cond, axis=1)
        g = (df.groupby(["condition", "pc_index"])["eigenvalue_share"]
             .agg(eigenvalue_share_mean="mean", eigenvalue_share_sd="std")
             .reset_index().round(8))
        write("simulation_eigenspectrum", g.to_dict(orient="records"))


DATASET_LABEL = {"svgworld": "SVG-World", "shapes3d": "3D Shapes",
                 "dsprites": "dSprites", "mpi3d_realistic": "MPI3D-realistic"}
# Production lanes carry the `_pil` tag (the reference resize backend). The
# untagged runs are the retained device-backend sensitivity analysis.
CANONICAL_SUFFIX = "_pil"
# Tag suffixes stripped when resolving a lane back to its dataset. `_pil2` is the
# MPI3D rerun with the corrected ordinal encoding of horizontal_axis.
TAG_SUFFIXES = ("_pil2", "_pil")


def canonical_lanes(results: str) -> list[str]:
    import glob
    lanes = [os.path.basename(os.path.dirname(f))
             for f in glob.glob(os.path.join(results, "*", "intervals.csv"))]
    tagged = {base_lane(n) for n in lanes if n != base_lane(n)}
    keep = [n for n in lanes if n != base_lane(n) or base_lane(n) not in tagged]
    return sorted(keep)


def base_lane(lane: str) -> str:
    for suf in TAG_SUFFIXES:
        if lane.endswith(suf):
            return lane[: -len(suf)]
    return lane


def cross_dataset_bundle(results: str) -> None:
    """Pretrained-minus-random gaps for the ADDED datasets only.

    SVG-World is excluded deliberately: its D3 is cross-style transfer, whose R^2 is
    uninterpretable (per-style scale mismatch drives it to -25), so plotting its
    transfer R^2 gap beside the datasets' checkerboard R^2 gaps would put an
    uninterpretable quantity on a shared axis. SVG-World appears in
    cross_model_gaps, where D3 is reported as Pearson rho.
    """
    import rebuttal_config as C
    rows = []
    for lane in canonical_lanes(results):
        if base_lane(lane) == "svgworld":
            continue
        f = os.path.join(results, lane, "intervals.csv")
        df = pd.read_csv(f)
        parts = df["statistic"].astype(str).str.split("|", expand=True)
        df["diagnostic"] = parts[0].str.replace("D3_checkerboard", "D3", regex=False)
        df["diagnostic"] = df["diagnostic"].str.replace(
            "D3_island_to_western", "D3", regex=False)
        df["model"] = parts[1]
        df = df[df["diagnostic"].isin(["forward", "D1_raw", "D3"])]
        # SVG-World carries a style dimension; average the two directions.
        keep = ["lane", "diagnostic", "model", "diff", "diff_ci_lo", "diff_ci_hi",
                "pretrained", "random", "n_units"]
        g = df.groupby(["lane", "diagnostic", "model"], as_index=False)[
            ["diff", "diff_ci_lo", "diff_ci_hi", "pretrained", "random",
             "n_units"]].mean()
        g["dataset_label"] = DATASET_LABEL.get(base_lane(lane), base_lane(lane))
        g["model_label"] = g["model"].map(
            lambda m: C.MODELS[m]["label"] if m in C.MODELS else m)
        rows.append(g[keep[:3] + ["diff", "diff_ci_lo", "diff_ci_hi", "pretrained",
                                  "random", "n_units", "dataset_label",
                                  "model_label"]])
    if rows:
        out = pd.concat(rows, ignore_index=True).round(6)
        write("cross_dataset_gaps", out.to_dict(orient="records"))


def cross_model_bundle(results: str) -> None:
    """The SVG-World panel, from the aggregated summary.

    D3 is the mean Pearson rho of the cross-style forward probe, not R^2: the
    probe is fitted on one style and applied to the other, and the per-style scale
    and offset mismatch drives R^2 arbitrarily negative (measured down to -25)
    without telling us whether the latent-aligned direction transferred. See
    src/aggregate_svgworld.py.
    """
    p = os.path.join(results, "svgworld_summary.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df = df[df["block"] == "pooled"]
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(
            model=r["model"], model_label=r["model_label"],
            paradigm=r["paradigm"], width=int(r["width"]),
            diagnostic="forward", diff=r["forward_gap"],
            diff_ci_lo=r["forward_gap_lo"], diff_ci_hi=r["forward_gap_hi"],
            pretrained=r["forward_pre"], random=r["forward_rand"]))
        rows.append(dict(
            model=r["model"], model_label=r["model_label"],
            paradigm=r["paradigm"], width=int(r["width"]),
            diagnostic="D1_raw", diff=r["d1_gap"],
            diff_ci_lo=r["d1_gap_lo"], diff_ci_hi=r["d1_gap_hi"],
            pretrained=r["d1_pre"], random=r["d1_rand"]))
        # rho has no bootstrap on the cached terms, so the band is the spread
        # across the 5 splits x 2 transfer directions.
        sd = (r.get("d3_rho_sd_pre", 0) or 0) + (r.get("d3_rho_sd_rand", 0) or 0)
        rows.append(dict(
            model=r["model"], model_label=r["model_label"],
            paradigm=r["paradigm"], width=int(r["width"]),
            diagnostic="D3_rho", diff=r["d3_rho_gap"],
            diff_ci_lo=r["d3_rho_gap"] - sd, diff_ci_hi=r["d3_rho_gap"] + sd,
            pretrained=r["d3_rho_pre"], random=r["d3_rho_rand"]))
    out = pd.DataFrame(rows).round(6)
    write("cross_model_gaps", out.to_dict(orient="records"))


def d2_curve_bundle(results: str) -> None:
    """S(k) for every encoder on SVG-World, pretrained and random."""
    import rebuttal_config as C  # noqa: F811

    lanes = [n for n in canonical_lanes(results) if base_lane(n) == "svgworld"]
    if not lanes:
        return
    p = os.path.join(results, lanes[0], "metrics_d2_curve.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    parts = df["condition"].astype(str).str.split("|", expand=True)
    df["model"], df["variant"], df["style"] = parts[0], parts[1], parts[2]
    df = df[df["block"] == "pooled"]
    g = (df.groupby(["model", "variant", "k"], as_index=False)["S"]
         .mean().rename(columns={"S": "S_mean"}))
    series = []
    for (m, v), sub in g.groupby(["model", "variant"]):
        sub = sub.sort_values("k")
        series.append(dict(model=m, variant=v,
                           model_label=C.MODELS[m]["label"],
                           k=[int(x) for x in sub["k"]],
                           S=[round(float(x), 6) for x in sub["S_mean"]]))
    d2_path = os.path.join(results, lanes[0], "metrics_d2.csv")
    if os.path.exists(d2_path):
        d2 = pd.read_csv(d2_path)
        r_eff = int(d2["r_eff"].iloc[0]) if "r_eff" in d2.columns else None
    else:
        # r_eff is the effective rank of the headline target block, which for the
        # 12 position latents is 12 (all vary, none constant).
        r_eff = len(C.LATENT_SUBSETS[C.HEADLINE_SUBSET])
    write("d2_curves", dict(meta=dict(r_eff=r_eff, lane=lanes[0]), series=series))


def pixel_control_bundle(results: str) -> None:
    """Pixel-space D1 per dataset beside the mean encoder D1 gap on that dataset."""
    p = os.path.join(results, "pixel_vs_gap.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df["dataset_label"] = df["ds"].map(
        {"svgworld": "SVG-World", "shapes3d": "3D Shapes", "dsprites": "dSprites",
         "mpi3d_realistic": "MPI3D-realistic"}).fillna(df["ds"])
    df["n_targets"] = df["ds"].map(
        {"svgworld": 12, "shapes3d": 8, "dsprites": 5, "mpi3d_realistic": 4})
    keep = ["ds", "dataset_label", "pixel_d1_raw", "pixel_d1_ceiling",
            "pixel_forward_r2", "mean_encoder_d1_gap", "n_targets"]
    write("pixel_control", df[keep].round(6).to_dict(orient="records"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    simulation_bundles(args.results)
    cross_dataset_bundle(args.results)
    cross_model_bundle(args.results)
    d2_curve_bundle(args.results)
    pixel_control_bundle(args.results)


if __name__ == "__main__":
    main()
