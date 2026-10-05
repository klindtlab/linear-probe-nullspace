"""Check a CelebA run and print everything the paper needs from it.

  python celeba_report.py results_celeba.csv [--ref reference/results_celeba.csv]
                          [--log run.log] [--out report]

Writes <out>.txt (checks + numbers for Sec. 9) and <out>_table.tex (rows of
tab:celeba). Exit code 1 if a check fails.
"""
import argparse
import re
import sys
import numpy as np
import pandas as pd

CELLS = [("dinov3_b16", "cls", "DINOv3-B/16", "CLS"),
         ("dinov3_b16", "patches_mean", "DINOv3-B/16", "patches"),
         ("clip_b16", "cls", "CLIP-B/16", "image")]
NORMS = ("raw", "standardized", "pca_whitened")


def f3(x):
    return f"{x:.3f}"


def cell(df, enc, feat):
    g = df[(df.encoder == enc) & (df.features == feat)]
    out = {}
    for norm in NORMS:
        h = g[g.normalization == norm]
        out[norm] = (h[h.weights == "pre"].iloc[0], h[h.weights == "rand"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ref", default=None, help="reference CSV (our released results) to compare against")
    ap.add_argument("--log", default=None, help="run log to scan for warnings/checks")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    lines, rows, ok = [], [], True
    ratio_pre, ratio_rnd, sc_pre, sc_rnd, pr_pre, pr_rnd = [], [], [], [], [], []

    lines.append("== tab:celeba rows ==")
    for enc, feat, ename, fname in CELLS:
        c = cell(df, enc, feat)
        pre, rnd = c["raw"]
        sc_p = c["pca_whitened"][0].R2_rev
        sc_r = c["pca_whitened"][1].R2_rev.mean()
        rows.append(f"{ename} & {fname} & pre & {f3(pre.forward_r2)} & "
                    f"{f3(pre.R2_rev)} [{f3(pre.rev_lo)}, {f3(pre.rev_hi)}] & "
                    f"{f3(pre.ceiling)} & {f3(pre.rev_over_ceiling)} & {f3(sc_p)} & "
                    f"{pre.part_ratio:.1f} \\\\")
        rows.append(f" & & rand & {f3(rnd.forward_r2.mean())} & "
                    f"{f3(rnd.R2_rev.mean())} $\\pm$ {f3(rnd.R2_rev.std(ddof=1))} & "
                    f"{f3(rnd.ceiling.mean())} & {f3(rnd.rev_over_ceiling.mean())} & "
                    f"{f3(sc_r)} & {rnd.part_ratio.mean():.1f} \\\\")
        ratio_pre.append(pre.rev_over_ceiling); ratio_rnd.append(rnd.rev_over_ceiling.mean())
        sc_pre.append(sc_p); sc_rnd.append(sc_r)
        pr_pre.append(pre.part_ratio); pr_rnd.append(rnd.part_ratio.mean())

        # ordering under every feature normalization
        for norm in NORMS:
            p, r = c[norm]
            gap = p.R2_rev - r.R2_rev.mean()
            flag = "ok" if gap > 0 else "FAIL"
            ok &= gap > 0
            lines.append(f"  gap {ename} {fname:8s} {norm:13s} {gap:+.4f}  {flag}")
    lines += rows

    raw = df[df.normalization == "raw"]
    clip = raw[(raw.encoder == "clip_b16") & (raw.weights == "pre")].R2_rev.iloc[0]
    dino = raw[(raw.encoder == "dinov3_b16") & (raw.features == "cls")
               & (raw.weights == "pre")].R2_rev.iloc[0]
    lines += [
        "",
        "== numbers for Sec. 9 (Real Photographs) ==",
        f"  share of ceiling, pretrained: {min(ratio_pre):.2f}--{max(ratio_pre):.2f}"
        f"   (DINOv3 CLS {ratio_pre[0]:.2f}, patches {ratio_pre[1]:.2f}, CLIP {ratio_pre[2]:.2f})",
        f"  share of ceiling, random:     {min(ratio_rnd):.2f}--{max(ratio_rnd):.2f}"
        f"   (DINOv3 CLS {ratio_rnd[0]:.2f}, patches {ratio_rnd[1]:.2f}, CLIP {ratio_rnd[2]:.2f})",
        f"  SC pretrained {min(sc_pre):.2f}--{max(sc_pre):.2f}, random {min(sc_rnd):.2f}--{max(sc_rnd):.2f}",
        f"  PR pretrained {min(pr_pre):.1f}--{max(pr_pre):.1f}, random <= {max(pr_rnd):.1f}",
        f"  raw R2_rev CLIP {clip:.3f} vs DINOv3 CLS {dino:.3f}  "
        f"({'CLIP higher' if clip > dino else 'DINOv3 higher: revise the CLIP sentence'})",
    ]

    if a.ref:
        ref = pd.read_csv(a.ref)
        key = ["encoder", "weights", "seed", "features", "normalization"]
        m = df.merge(ref, on=key, suffixes=("", "_ref"))
        lines += ["", "== this run vs reference (max relative |diff|) =="]
        for enc in ("dinov3_b16", "clip_b16"):
            sub = m[m.encoder == enc]
            cols = ["forward_r2", "R2_rev", "ceiling", "part_ratio"]
            d = max(np.nanmax(np.abs(sub[c] - sub[c + "_ref"])
                              / np.maximum(np.abs(sub[c + "_ref"]), 1e-12)) for c in cols)
            good = d < 1e-2   # fresh extraction on other hardware: float-level noise
            ok &= good
            lines.append(f"  {enc}: {d:.1e}  "
                         f"{'ok (matches reference)' if good else 'FAIL: differs from reference'}")

    if a.log:
        txt = open(a.log, errors="ignore").read()
        n_warn = txt.count("QuickGELU mismatch")
        rec = [float(x) for x in re.findall(r"recoding check .*?= ([0-9.e+-]+)", txt)]
        st = re.findall(r"by at most ([0-9.e+-]+)", txt)
        ok &= n_warn == 0
        lines += ["", "== log checks ==",
                  f"  QuickGELU warnings: {n_warn}  {'ok' if n_warn == 0 else 'FAIL'}",
                  f"  synthetic recoding self-test max diff: {st[0] if st else 'n/a'}",
                  f"  real-data recoding check max diff: "
                  f"{max(rec):.1e}" if rec else "  real-data recoding check: n/a"]

    lines += ["", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"]
    report = "\n".join(lines)
    print(report)
    open(a.out + ".txt", "w").write(report + "\n")
    open(a.out + "_table.tex", "w").write("\n".join(rows) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
