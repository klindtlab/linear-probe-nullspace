"""Recompute the cross-dataset aggregates and the pixel-vs-gap table.

Run after any dataset lane rerun. It reads whichever tagged lanes are canonical,
recomputes the per-dataset mean encoder D1 gap, and rewrites results/pixel_vs_gap.csv
so the pixel-space control is always compared against current encoder numbers.

    python src/refresh_dataset_numbers.py
"""
from __future__ import annotations

import glob
import os

import pandas as pd
from scipy.stats import spearmanr

TAGS = ("_pil3", "_pil2", "_pil")
LABEL = {"svgworld": "SVG-World", "shapes3d": "3D Shapes", "dsprites": "dSprites",
         "mpi3d_realistic": "MPI3D-realistic"}


def base(lane: str) -> str:
    for t in TAGS:
        if lane.endswith(t):
            return lane[: -len(t)]
    return lane


def canonical(results: str) -> dict[str, str]:
    """Highest-numbered tag wins for each dataset."""
    best: dict[str, str] = {}
    rank = {t: i for i, t in enumerate(TAGS)}
    for f in glob.glob(os.path.join(results, "*", "intervals.csv")):
        lane = os.path.basename(os.path.dirname(f))
        b = base(lane)
        if lane == b:
            continue
        tag = lane[len(b):]
        if b not in best or rank[tag] < rank[best[b][len(b):]]:
            best[b] = lane
    return best


def main():
    results = "results"
    lanes = canonical(results)
    print("canonical lanes:", lanes)
    gaps = {}
    for b, lane in lanes.items():
        d = pd.read_csv(os.path.join(results, lane, "intervals.csv"))
        p = d["statistic"].str.split("|", expand=True)
        # dataset lanes have 3 fields (diag|model|block), SVG-World has 4
        # (diag|model|style|block), so resolve the block column by width.
        block = p[3].fillna(p[2]) if p.shape[1] > 3 else p[2]
        d = d.assign(diagnostic=p[0], block=block)
        d1 = d[(d.diagnostic == "D1_raw") & (d.block == "pooled")]
        gaps[b] = float(d1["diff"].mean())
        print(f"  {b:18s} mean D1 gap {gaps[b]:+.4f}  (n={len(d1)}, lane={lane})")

    px = pd.read_csv(os.path.join(results, "pixel_control", "pixel_control.csv"))
    px["ds"] = px.dataset.str.replace("svgworld_island", "svgworld").str.replace(
        "svgworld_western", "svgworld")
    t = px.groupby("ds")[["pixel_d1_raw", "pixel_forward_r2",
                          "pixel_d1_ceiling"]].mean()
    t["mean_encoder_d1_gap"] = [gaps[i] for i in t.index]
    t = t.sort_values("pixel_d1_raw").reset_index()
    t.round(6).to_csv(os.path.join(results, "pixel_vs_gap.csv"), index=False)
    print("\n" + t.round(4).to_string(index=False))
    r = spearmanr(t.pixel_d1_raw, t.mean_encoder_d1_gap)
    print(f"\nSpearman(pixel D1, encoder D1 gap) = {r.statistic:.4f} "
          f"(n = {len(t)}, p = {r.pvalue:.4f})")
    print("wrote results/pixel_vs_gap.csv")


if __name__ == "__main__":
    main()
