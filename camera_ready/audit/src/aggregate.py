"""Read the lane CSVs and write the audit tables.

Runs on CPU on the small CSVs the jobs produce. Every output table is small and
plot-ready.

    PYTHONPATH=src python src/aggregate.py --lanes results/prod
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

RAND_PREFIX = "rand"


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
def parity_table(lanes: str, out: str) -> pd.DataFrame | None:
    df = _read(os.path.join(lanes, "svgworld", "implementation_parity.csv"))
    if df is None:
        return None
    cols = ["d1_abs_diff", "ceiling_abs_diff", "d2_max_abs_diff",
            "forward_abs_diff"]
    summary = pd.DataFrame([{
        "n_conditions": len(df),
        **{f"max_{c}": float(df[c].max()) for c in cols},
        **{f"median_{c}": float(df[c].median()) for c in cols},
    }])
    df.to_csv(os.path.join(out, "implementation_parity.csv"), index=False)
    summary.to_csv(os.path.join(out, "implementation_parity_summary.csv"),
                   index=False)
    return summary


def recipe_table(lanes: str, out: str) -> pd.DataFrame | None:
    df = _read(os.path.join(lanes, "svgworld", "recipe_ladder.csv"))
    if df is None:
        return None
    df.to_csv(os.path.join(out, "recipe_ladder.csv"), index=False)
    base = df[df.step == "baseline"].set_index("condition")["d1"]
    rev = df[df.label == "revised recipe"].set_index("condition")["d1"]
    single = (df[df.step == "single"]
              .assign(factor=lambda d: d.label.str.replace("one factor: ", "",
                                                           regex=False))
              .pivot(index="condition", columns="factor",
                     values="delta_vs_submitted"))
    tab = single.copy()
    tab.insert(0, "submitted_d1", base)
    tab["revised_d1"] = rev
    tab["total_change"] = tab["revised_d1"] - tab["submitted_d1"]
    tab = tab.reset_index()
    tab.to_csv(os.path.join(out, "recipe_attribution.csv"), index=False)
    # which single factor dominates, per condition
    fcols = [c for c in single.columns]
    dom = []
    for _, r in tab.iterrows():
        mags = {c: abs(r[c]) for c in fcols if pd.notna(r[c])}
        top = max(mags, key=mags.get) if mags else None
        dom.append(dict(condition=r["condition"], dominant_factor=top,
                        dominant_delta=float(r[top]) if top else np.nan,
                        second_largest=float(sorted(mags.values())[-2])
                        if len(mags) > 1 else np.nan,
                        total_change=float(r["total_change"])))
    pd.DataFrame(dom).to_csv(os.path.join(out, "recipe_dominant_factor.csv"),
                             index=False)
    return tab


def _instance_kind(name: str) -> str:
    if name == "pre":
        return "pretrained"
    if name.endswith("_bncal"):
        return "random_bncal"
    return "random"


def inversion_table(lanes: str, out: str, datasets) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        df = _read(os.path.join(lanes, ds, "metrics.csv"))
        if df is None:
            continue
        df["kind"] = df.instance.map(_instance_kind)
        keys = ["dataset", "sample_seed", "model", "encoding"]
        for key, g in df.groupby(keys):
            pre = g[g.kind == "pretrained"]
            rnd = g[g.kind == "random"]
            bn = g[g.kind == "random_bncal"]
            if pre.empty or rnd.empty:
                continue
            d1_pre = float(pre.d1_raw.iloc[0])
            rec = dict(zip(keys, key))
            rec.update(
                n=int(pre.n.iloc[0]),
                d1_pretrained=d1_pre,
                d1_random_mean=float(rnd.d1_raw.mean()),
                d1_random_min=float(rnd.d1_raw.min()),
                d1_random_max=float(rnd.d1_raw.max()),
                n_random_seeds=int(len(rnd)),
                gap=d1_pre - float(rnd.d1_raw.mean()),
                gap_worst_case=d1_pre - float(rnd.d1_raw.max()),
                gap_best_case=d1_pre - float(rnd.d1_raw.min()),
                sign_consistent_over_seeds=bool(
                    np.all(np.sign(d1_pre - rnd.d1_raw.values) ==
                           np.sign(d1_pre - rnd.d1_raw.mean()))),
                forward_pretrained=float(pre.forward_r2.iloc[0]),
                forward_random_mean=float(rnd.forward_r2.mean()),
                d3_rho_pretrained=float(pre.get("d3_rho", pd.Series([np.nan])).iloc[0]),
                d3_rho_random_mean=float(rnd.get("d3_rho", pd.Series([np.nan])).mean()),
                pr_pretrained=float(pre.fs_participation_ratio.iloc[0]),
                pr_random_mean=float(rnd.fs_participation_ratio.mean()),
                d1_norm_pretrained=float(pre.d1_normalized.iloc[0]),
                d1_norm_random_mean=float(rnd.d1_normalized.mean()),
            )
            if not bn.empty:
                rec.update(d1_random_bncal_mean=float(bn.d1_raw.mean()),
                           gap_bncal=d1_pre - float(bn.d1_raw.mean()),
                           pr_random_bncal=float(bn.fs_participation_ratio.mean()))
            rows.append(rec)
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab.to_csv(os.path.join(out, "inversion_robustness.csv"), index=False)
        sign = []
        for (ds, model), g in tab.groupby(["dataset", "model"]):
            neg = int((g.gap < 0).sum())
            sign.append(dict(
                dataset=ds, model=model, n_cells=int(len(g)),
                n_cells_gap_negative=neg,
                sign_stable=bool(neg == 0 or neg == len(g)),
                gap_min=float(g.gap.min()), gap_max=float(g.gap.max()),
                gap_median=float(g.gap.median()),
                canonical_gap=float(g[g.encoding == "canonical"].gap.mean()),
                worst_case_gap=float(g.gap_worst_case.min()),
            ))
        pd.DataFrame(sign).to_csv(os.path.join(out, "inversion_sign_summary.csv"),
                                  index=False)
    return tab


def intervals_table(lanes: str, out: str, datasets) -> pd.DataFrame:
    frames = []
    for ds in list(datasets) + ["svgworld"]:
        df = _read(os.path.join(lanes, ds, "intervals.csv"))
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(os.path.join(out, "intervals_hierarchical.csv"), index=False)
    return tab


def svgworld_panel(lanes: str, out: str) -> pd.DataFrame | None:
    df = _read(os.path.join(lanes, "svgworld", "panel_d1_d2_d3.csv"))
    if df is None:
        return None
    df["kind"] = df.instance.map(_instance_kind)
    df.to_csv(os.path.join(out, "svgworld_panel.csv"), index=False)
    rows = []
    for (pool, style), g in df.groupby(["pooling", "style"]):
        pre = g[g.kind == "pretrained"]
        rnd = g[g.kind == "random"]
        rows.append(dict(pooling=pool, style=style,
                         d1_pretrained=float(pre.d1_raw.iloc[0]),
                         d1_random_mean=float(rnd.d1_raw.mean()),
                         d1_random_min=float(rnd.d1_raw.min()),
                         d1_random_max=float(rnd.d1_raw.max()),
                         n_random_seeds=int(len(rnd)),
                         gap=float(pre.d1_raw.iloc[0] - rnd.d1_raw.mean()),
                         gap_worst_case=float(pre.d1_raw.iloc[0] -
                                              rnd.d1_raw.max()),
                         S_at_r_pre=float(pre.S_at_r.iloc[0]),
                         S_at_r_rand=float(rnd.S_at_r.mean()),
                         C_at_r_pre=float(pre.C_at_r.iloc[0]),
                         C_at_r_rand=float(rnd.C_at_r.mean()),
                         forward_pre=float(pre.forward_r2.iloc[0]),
                         forward_rand=float(rnd.forward_r2.mean())))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "svgworld_summary.csv"), index=False)
    return tab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lanes", default="results/prod")
    p.add_argument("--out", default="results")
    p.add_argument("--datasets", default="shapes3d,dsprites,mpi3d_realistic")
    args = p.parse_args()
    datasets = [d for d in args.datasets.split(",") if d]
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    par = parity_table(args.lanes, args.out)
    if par is not None:
        summary["implementation_parity"] = par.to_dict("records")[0]
    rec = recipe_table(args.lanes, args.out)
    if rec is not None:
        summary["recipe_attribution_conditions"] = int(len(rec))
    inv = inversion_table(args.lanes, args.out, datasets)
    if not inv.empty:
        summary["inversion_cells"] = int(len(inv))
        summary["inversion_negative_cells"] = int((inv.gap < 0).sum())
    iv = intervals_table(args.lanes, args.out, datasets)
    if not iv.empty:
        summary["intervals_rows"] = int(len(iv))
    sw = svgworld_panel(args.lanes, args.out)
    if sw is not None:
        summary["svgworld_rows"] = int(len(sw))
    with open(os.path.join(args.out, "aggregate_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
