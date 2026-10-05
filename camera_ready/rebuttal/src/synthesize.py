"""Synthesis: join the lanes' small CSVs into the verdict-bearing tables.

Runs on CPU outside any GPU job. Input is whatever lane directories exist under
`results/`; output is one aggregated table per
diagnostic plus the plot-ready JSON the figures read.

Point estimates are means over the five repeated splits. Intervals come from the
lanes' paired bootstrap on the shared split, which is what the `intervals.csv`
files carry, so nothing is recomputed here from summary statistics.

    python src/synthesize.py --results results --out results
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

import rebuttal_config as C

DIAG_FILES = {
    "d1": "metrics_d1.csv",
    "d2": "metrics_d2.csv",
    "d2_curve": "metrics_d2_curve.csv",
    "forward": "metrics_forward.csv",
    "d3": "metrics_d3.csv",
    "intervals": "intervals.csv",
    "splits": "split_audits.csv",
    "sensitivity": "sensitivity_latent_subsets.csv",
}


def load_lanes(root: str) -> dict[str, dict[str, pd.DataFrame]]:
    lanes: dict[str, dict[str, pd.DataFrame]] = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        lane = os.path.basename(d)
        got = {}
        for key, fname in DIAG_FILES.items():
            p = os.path.join(d, fname)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                got[key] = pd.read_csv(p)
        rp = os.path.join(d, "lane_report.json")
        if os.path.exists(rp):
            got["report"] = json.load(open(rp))
        if got:
            lanes[lane] = got
            print(f"[load] {lane}: {sorted(k for k in got)}")
    return lanes


def _split_condition(df: pd.DataFrame) -> pd.DataFrame:
    """`condition` is 'model|variant' or 'model|variant|style'."""
    df = df.copy()
    parts = df["condition"].astype(str).str.split("|", expand=True)
    df["model"] = parts[0]
    df["variant"] = parts[1] if parts.shape[1] > 1 else None
    df["style"] = parts[2] if parts.shape[1] > 2 else None
    return df


def model_label(m: str) -> str:
    return C.MODELS[m]["label"] if m in C.MODELS else m


def paradigm(m: str) -> str:
    return C.MODELS[m]["paradigm"] if m in C.MODELS else ""


def aggregate_d1(lanes: dict) -> pd.DataFrame:
    rows = []
    for lane, got in lanes.items():
        if "d1" not in got:
            continue
        df = _split_condition(got["d1"])
        keys = ["lane", "model", "variant", "block"] + \
               (["style"] if df["style"].notna().any() else [])
        g = df.groupby(keys, dropna=False).agg(
            d1_raw_mean=("d1_raw", "mean"), d1_raw_std=("d1_raw", "std"),
            ceiling_mean=("ceiling", "mean"),
            d1_norm_mean=("d1_normalized", "mean"),
            d1_norm_std=("d1_normalized", "std"),
            pc1_share_mean=("pc1_share", "mean"),
            width=("width", "first"), target_rank=("target_rank", "first"),
            alpha_median=("alpha", "median"), n_splits=("split_seed", "nunique"),
        ).reset_index()
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["model_label"] = out["model"].map(model_label)
        out["paradigm"] = out["model"].map(paradigm)
    return out


def aggregate_forward(lanes: dict, kind="within") -> pd.DataFrame:
    rows = []
    for lane, got in lanes.items():
        if "forward" not in got:
            continue
        df = got["forward"]
        df = df[df["kind"] == kind]
        if not len(df):
            continue
        df = _split_condition(df)
        keys = ["lane", "model", "variant", "block"] + \
               (["style"] if df["style"].notna().any() else [])
        g = df.groupby(keys, dropna=False).agg(
            r2_mean=("r2_mean", "mean"), r2_std=("r2_mean", "std"),
            r2_aggregate=("r2_aggregate", "mean"),
            n_splits=("split_seed", "nunique")).reset_index()
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["model_label"] = out["model"].map(model_label)
    return out


def aggregate_d3(lanes: dict) -> pd.DataFrame:
    rows = []
    for lane, got in lanes.items():
        if "d3" not in got:
            continue
        df = got["d3"].copy()
        if "source" in df.columns and df["source"].notna().any():
            src = df["source"].astype(str).str.split("|", expand=True)
            df["model"] = src[0]
            df["variant"] = src[1]
            df["direction"] = (df["source"].astype(str).str.split("|").str[-1] + "->"
                               + df["target"].astype(str).str.split("|").str[-1])
            keys = ["lane", "model", "variant", "block"]
        else:
            df = _split_condition(df)
            df["direction"] = "checkerboard"
            keys = ["lane", "model", "variant", "block"]
        agg = dict(r2_mean=("r2_mean", "mean"), r2_std=("r2_mean", "std"),
                   n_rows=("r2_mean", "size"))
        if "iid_reference_r2_mean" in df.columns:
            agg["iid_reference_r2_mean"] = ("iid_reference_r2_mean", "mean")
        g = df.groupby(keys, dropna=False).agg(**agg).reset_index()
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["model_label"] = out["model"].map(model_label)
    return out


def aggregate_intervals(lanes: dict) -> pd.DataFrame:
    rows = [got["intervals"] for got in lanes.values() if "intervals" in got]
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    parts = out["statistic"].astype(str).str.split("|", expand=True)
    out["diagnostic"] = parts[0]
    out["model"] = parts[1]
    out["model_label"] = out["model"].map(model_label)
    out["paradigm"] = out["model"].map(paradigm)
    out["rest"] = parts[2] if parts.shape[1] > 2 else None
    out["block"] = parts[3] if parts.shape[1] > 3 else None
    return out


def d2_curves(lanes: dict) -> pd.DataFrame:
    rows = []
    for lane, got in lanes.items():
        if "d2_curve" not in got:
            continue
        df = _split_condition(got["d2_curve"])
        keys = ["lane", "model", "variant", "block", "k"] + \
               (["style"] if df["style"].notna().any() else [])
        g = df.groupby(keys, dropna=False).agg(
            S_mean=("S", "mean"), S_std=("S", "std")).reset_index()
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["model_label"] = out["model"].map(model_label)
    return out


def leakage_summary(lanes: dict) -> pd.DataFrame:
    rows = []
    for lane, got in lanes.items():
        if "splits" in got:
            df = got["splits"].copy()
            df["lane"] = lane
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_plot_data(d1, fwd, d3, ci, curves) -> dict:
    """Small plot-ready payload: no figure reads a raw metrics CSV."""
    def recs(df, cols):
        if not len(df):
            return []
        keep = [c for c in cols if c in df.columns]
        return df[keep].round(6).to_dict(orient="records")

    return dict(
        d1=recs(d1, ["lane", "model", "model_label", "paradigm", "variant", "style",
                     "block", "width", "target_rank", "d1_raw_mean", "d1_raw_std",
                     "ceiling_mean", "d1_norm_mean", "d1_norm_std",
                     "pc1_share_mean", "n_splits"]),
        forward=recs(fwd, ["lane", "model", "model_label", "variant", "style",
                           "block", "r2_mean", "r2_std", "n_splits"]),
        d3=recs(d3, ["lane", "model", "model_label", "variant", "block",
                     "direction", "r2_mean", "r2_std", "iid_reference_r2_mean"]),
        intervals=recs(ci, ["lane", "statistic", "diagnostic", "model",
                            "model_label", "paradigm", "rest", "block", "diff",
                            "diff_ci_lo", "diff_ci_hi", "diff_excludes_zero",
                            "pretrained", "pretrained_ci_lo", "pretrained_ci_hi",
                            "random", "random_ci_lo", "random_ci_hi", "n_units"]),
        d2_curve=recs(curves, ["lane", "model", "model_label", "variant", "style",
                               "block", "k", "S_mean", "S_std"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    lanes = load_lanes(args.results)
    if not lanes:
        raise SystemExit(f"no lane directories with metrics under {args.results}")

    d1 = aggregate_d1(lanes)
    fwd = aggregate_forward(lanes, "within")
    fwd_cat = aggregate_forward(lanes, "within_categorical")
    d3 = aggregate_d3(lanes)
    ci = aggregate_intervals(lanes)
    curves = d2_curves(lanes)
    leak = leakage_summary(lanes)

    os.makedirs(args.out, exist_ok=True)
    for name, df in [("agg_d1", d1), ("agg_forward", fwd),
                     ("agg_forward_categorical", fwd_cat), ("agg_d3", d3),
                     ("agg_intervals", ci), ("agg_d2_curve", curves),
                     ("agg_split_audits", leak)]:
        if len(df):
            p = os.path.join(args.out, f"{name}.csv")
            df.to_csv(p, index=False)
            print(f"[write] {p} ({len(df)} rows)")

    plot = build_plot_data(d1, fwd, d3, ci, curves)
    p = os.path.join(args.out, "plot_data.json")
    with open(p, "w") as f:
        json.dump(plot, f, indent=1)
    print(f"[write] {p} ({os.path.getsize(p) / 1e3:.0f} KB)")

    # ---- integrity summary the report and the claims manifest are checked against
    integrity = dict(
        lanes=sorted(lanes),
        n_conditions_d1=int(len(d1)),
        leakage_clean=bool(len(leak) == 0 or (
            (leak.get("index_overlap", pd.Series([0])).max() == 0)
            and (leak.get("content_overlap", pd.Series([0])).max() == 0))),
        d1_never_exceeds_ceiling=bool(
            len(d1) == 0 or (d1["d1_raw_mean"] <= d1["ceiling_mean"] + 1e-6).all()),
        n_intervals=int(len(ci)),
        n_intervals_excluding_zero=int(ci["diff_excludes_zero"].sum()) if len(ci) else 0,
    )
    if len(ci):
        pos = ci[(ci["diagnostic"] == "D1_raw") & (ci["diff_ci_lo"] > 0)]
        integrity["d1_positive_gaps"] = int(len(pos))
        integrity["d1_models_with_positive_gap"] = sorted(pos["model"].unique().tolist())
        neg = ci[(ci["diagnostic"] == "D1_raw") & (ci["diff_ci_hi"] < 0)]
        integrity["d1_negative_gaps"] = int(len(neg))
        integrity["d1_ties"] = int(len(ci[(ci["diagnostic"] == "D1_raw")
                                         & ~ci["diff_excludes_zero"]]))
    p = os.path.join(args.out, "integrity_summary.json")
    with open(p, "w") as f:
        json.dump(integrity, f, indent=1)
    print(f"[write] {p}")
    print(json.dumps(integrity, indent=1))


if __name__ == "__main__":
    main()
