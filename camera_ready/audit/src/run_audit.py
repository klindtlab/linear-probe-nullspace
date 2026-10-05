"""One entrypoint for the audit.

    python src/run_audit.py --stage smoke                      # tiny, everything
    python src/run_audit.py --stage svgworld --n 10000 --tag prod
    python src/run_audit.py --stage dataset --dataset shapes3d --tag prod

Stages
------
smoke      every stage at a tiny size, in one job, through this same code path.
svgworld   recipe attribution (submitted recipe -> revised recipe, one factor at
           a time), implementation parity on identical arrays, rasterizer
           comparison, and the five-random-seed D1/D2/D3 panel on DINOv3 B/16.
dataset    the cross-dataset inversion: three encoders, one pretrained and five
           random initializations each (ResNet-50 also BatchNorm-calibrated),
           three dataset sample seeds, three target encodings, with hierarchical
           intervals over held-out units and control initializations.

Everything is written to $ARTIFACTS_DIR/<tag>/<stage>/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

import audit_calc as AC
import audit_encode as AE
import legacy_calc as LC
import rebuttal_config as C

ALPHAS = C.RIDGE_ALPHAS
D2_KS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 767)
RAND_SEEDS = (42, 43, 44, 45, 46)
DATASET_SAMPLE_SEEDS = (1234, 7, 2024)


def out_dir(tag: str, stage: str) -> str:
    d = os.path.join(C.artifacts_dir(), tag, stage)
    os.makedirs(d, exist_ok=True)
    return d


def jdump(obj, path: str):
    def enc(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=enc)
    os.replace(tmp, path)
    print(f"[write] {path}", flush=True)


def rows_to_csv(rows: list[dict], path: str):
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    os.replace(tmp, path)
    print(f"[write] {path} ({len(rows)} rows)", flush=True)


# ===========================================================================
# Stage: svgworld
# ===========================================================================
def alt_rasterizer_check(Z: np.ndarray, n_scenes: int, size: int) -> dict:
    """Rasterize the same SVG strings with a second, independent engine.

    The submitted pipeline used cairosvg, which the job image cannot load. The
    prior run substituted resvg and validated its geometry against the scene
    generators. This check goes further: it rasterizes identical SVG strings with
    a different engine and reports pixel agreement, so any renderer contribution
    to D1 is measured rather than argued.
    """
    from svgworld import raster, svg_island
    from svgworld_data import square_pad_svg

    report = {"primary_backend": raster.backend_info(), "alt_backend": None}
    alt = None
    try:
        import skia  # type: ignore

        def alt(svg: str, size: int):
            dom = skia.SVGDOM.MakeFromString(skia.String(svg))
            surf = skia.Surface(size, size)
            with surf as canvas:
                canvas.clear(skia.ColorWHITE)
                dom.setContainerSize(skia.Size(size, size))
                dom.render(canvas)
            img = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)
            return np.asarray(img)[:, :, :3].copy()

        report["alt_backend"] = {"backend": "skia-python",
                                 "version": getattr(skia, "__version__", "unknown")}
    except Exception as exc:                       # pragma: no cover
        report["alt_backend_error"] = f"skia unavailable: {exc}"

    if alt is None:
        try:
            from io import BytesIO

            from reportlab.graphics import renderPM  # type: ignore
            from svglib.svglib import svg2rlg  # type: ignore

            def alt(svg: str, size: int):
                from PIL import Image

                drawing = svg2rlg(BytesIO(svg.encode("utf-8")))
                sc = size / max(drawing.width, drawing.height)
                drawing.width, drawing.height = size, size
                drawing.scale(sc, sc)
                png = renderPM.drawToString(drawing, fmt="PNG", bg=0xFFFFFF)
                return np.asarray(Image.open(BytesIO(png)).convert("RGB"))

            report["alt_backend"] = {"backend": "svglib+reportlab"}
            report.pop("alt_backend_error", None)
        except Exception as exc:                   # pragma: no cover
            report["alt_backend_error2"] = f"svglib unavailable: {exc}"

    if alt is None:
        # Always-available fallback: rasterize at 4x and downsample. This does
        # not test a second engine, it tests whether the statistic is sensitive
        # to the sub-pixel coverage decisions where two engines differ
        # (anti-aliasing, edge sampling), which is the mechanism by which a
        # rasterizer swap could move D1 at all.
        def alt(svg: str, size: int):
            from PIL import Image

            big = raster.svg_to_rgb_array(svg, size=size * 4)
            return np.asarray(Image.fromarray(big).resize(
                (size, size), Image.LANCZOS))

        report["alt_backend"] = {"backend": "resvg_py 4x supersampled",
                                 "version": "0.3.3",
                                 "note": "same engine, different sub-pixel "
                                         "coverage; not a cairosvg reproduction"}

    prim, secd = [], []
    for i in range(n_scenes):
        svg = square_pad_svg(svg_island.generate_scene_svg(Z[i]))
        prim.append(raster.svg_to_rgb_array(svg, size=size))
        try:
            a = alt(svg, size)
        except Exception as exc:                   # pragma: no cover
            report["alt_render_error"] = str(exc)
            return report
        if a.shape != prim[-1].shape:
            report["alt_shape_mismatch"] = [list(a.shape), list(prim[-1].shape)]
            return report
        secd.append(a)
    P = np.stack(prim).astype(np.int16)
    S = np.stack(secd).astype(np.int16)
    diff = np.abs(P - S)
    report.update(n_scenes=int(n_scenes),
                  mean_abs_pixel_diff=float(diff.mean()),
                  p99_abs_pixel_diff=float(np.percentile(diff, 99)),
                  frac_pixels_differing_gt2=float((diff > 2).mean()),
                  primary_pixel_mean=float(P.mean()),
                  alt_pixel_mean=float(S.mean()))
    report["_images"] = (np.stack(prim).astype(np.uint8),
                         np.stack(secd).astype(np.uint8))
    return report


def recipe_ladder(Z_all: np.ndarray, X: np.ndarray, cond: str) -> list[dict]:
    """From the submitted recipe to the revised one, one factor at a time.

    Factors, each defined against the submitted baseline:
      formula   variance-of-residual (submitted) -> sum of squared error
      dtype     float32 (submitted) -> float64
      targets   all 32 stored columns (submitted) -> the 12 position latents
      alpha     1.0 (submitted) -> 5-fold CV inside the training half
      split     sklearn row split, random_state 42 (submitted) -> unit split,
                seed 0, shared across the paired styles
    Reported twice: each factor alone against the baseline, and cumulatively in
    the order above.
    """
    pos = C.POSITION_DIMS
    rows: list[dict] = []

    def evaluate(formula: str, dtype: str, targets: str, alpha_mode: str,
                 split: str, label: str, step: str) -> dict:
        Zt = Z_all[:, pos] if targets == "position12" else Z_all
        Xc = np.asarray(X, np.float32 if dtype == "f32" else np.float64)
        Zc = np.asarray(Zt, np.float32 if dtype == "f32" else np.float64)
        if split == "legacy42":
            tr, te = LC.legacy_split(len(Xc), seed=42)
        else:
            tr, te = AC.unit_split(len(Xc), seed=0)
        if alpha_mode == "fixed1":
            alpha = 1.0
        else:
            alpha, _ = AC.ridge_cv_alpha(Zc[tr].astype(np.float64),
                                         Xc[tr].astype(np.float64), ALPHAS,
                                         folds=C.RIDGE_CV_FOLDS, seed=0)
        if formula == "variance":
            val = LC.legacy_reverse_r2(Zc[tr], Xc[tr], Zc[te], Xc[te], alpha=alpha)
        else:
            val = AC.d1(Zc[tr], Xc[tr], Zc[te], Xc[te], alpha=alpha)["d1_raw"]
        return dict(condition=cond, label=label, step=step, formula=formula,
                    dtype=dtype, targets=targets, alpha_mode=alpha_mode,
                    split=split, alpha=float(alpha), d1=float(val),
                    n_train=int(len(tr)), n_test=int(len(te)))

    base = dict(formula="variance", dtype="f32", targets="all32",
                alpha_mode="fixed1", split="legacy42")
    rows.append(evaluate(**base, label="submitted recipe", step="baseline"))
    b = rows[0]["d1"]

    single = [("formula", "sse"), ("dtype", "f64"), ("targets", "position12"),
              ("alpha_mode", "cv"), ("split", "unit0")]
    for key, val in single:
        cfg = dict(base)
        cfg[key] = val
        r = evaluate(**cfg, label=f"one factor: {key}={val}", step="single")
        r["delta_vs_submitted"] = r["d1"] - b
        rows.append(r)

    cfg = dict(base)
    for key, val in single:
        cfg[key] = val
        r = evaluate(**cfg, label=f"cumulative through {key}={val}",
                     step="cumulative")
        r["delta_vs_submitted"] = r["d1"] - b
        rows.append(r)
    rows[-1]["label"] = "revised recipe"
    return rows


def stage_svgworld(args) -> dict:
    import svgworld_data as SW
    from svgworld import raster

    od = out_dir(args.tag, "svgworld")
    t0 = time.time()
    Z = SW.sample_latents(args.n, seed=C.DATA_SEED)
    images = {s: SW.render_style(Z, s, size=C.IMAGE_SIZE) for s in C.STYLES}
    report = dict(stage="svgworld", n_scenes=int(args.n),
                  raster=raster.backend_info(),
                  render_report=SW.render_report(images))

    # --- rasterizer comparison on identical SVG strings --------------------
    alt = alt_rasterizer_check(Z, min(args.n_alt, args.n), C.IMAGE_SIZE)
    alt_imgs = alt.pop("_images", None)
    report["rasterizer_comparison"] = alt

    device = "cuda" if args.device == "auto" and _cuda() else args.device
    if device == "auto":
        device = "cpu"
    report["device"] = device

    # --- features: pretrained + five random initializations ----------------
    instances = ([dict(name="pre", variant="pre", seed=0)] +
                 [dict(name=f"rand{s}", variant="rand", seed=s)
                  for s in RAND_SEEDS[:args.n_rand_seeds]])
    feats: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    infos = {}
    for style in C.STYLES:
        got = AE.extract_instances(images[style], "dinov3_b16", instances, device,
                                   batch_size=args.batch_size,
                                   extra_poolings=("patches_mean",))
        feats[style] = got["features"]
        infos[style] = got["info"]
    report["encoder_info"] = infos

    # --- rasterizer effect on the statistic itself ------------------------
    if alt_imgs is not None and alt.get("alt_backend"):
        n_alt = alt_imgs[0].shape[0]
        pair = AE.extract_instances(
            np.concatenate([alt_imgs[0], alt_imgs[1]]), "dinov3_b16",
            [dict(name="pre", variant="pre", seed=0),
             dict(name="rand42", variant="rand", seed=42)], device,
            batch_size=args.batch_size)["features"]
        rr = {}
        for name in ("pre", "rand42"):
            X = pair[name]["pooled"]
            for which, sl in (("primary", slice(0, n_alt)),
                              ("alt", slice(n_alt, 2 * n_alt))):
                tr, te = AC.unit_split(n_alt, seed=0)
                Xa = X[sl]
                rr[f"{name}_{which}"] = AC.d1(
                    Z[:n_alt][tr][:, C.POSITION_DIMS], Xa[tr],
                    Z[:n_alt][te][:, C.POSITION_DIMS], Xa[te], alpha=1.0)["d1_raw"]
        rr["pre_shift"] = rr["pre_alt"] - rr["pre_primary"]
        rr["rand42_shift"] = rr["rand42_alt"] - rr["rand42_primary"]
        rr["gap_primary"] = rr["pre_primary"] - rr["rand42_primary"]
        rr["gap_alt"] = rr["pre_alt"] - rr["rand42_alt"]
        rr["n_scenes"] = int(n_alt)
        report["rasterizer_d1_effect"] = rr

    # --- recipe attribution + implementation parity -----------------------
    ladder_rows, parity_rows = [], []
    for style in C.STYLES:
        for name in ("pre", "rand42"):
            for pooling, plabel in (("pooled", "cls"),
                                    ("patches_mean", "patches_mean")):
                X = feats[style][name][pooling]
                cond = f"{style}|{name}|{plabel}"
                ladder_rows += recipe_ladder(Z, X, cond)
                parity_rows.append(_parity_row(Z, X, cond))
    rows_to_csv(ladder_rows, os.path.join(od, "recipe_ladder.csv"))
    rows_to_csv(parity_rows, os.path.join(od, "implementation_parity.csv"))

    # --- the panel under the revised recipe, five random seeds ------------
    panel, ivals, d2_rows = [], [], []
    unit_hash = np.array([hash(r.tobytes()) for r in
                          np.ascontiguousarray(Z.astype(np.float32))])
    for pooling, plabel in (("pooled", "cls"), ("patches_mean", "patches_mean")):
        for style in C.STYLES:
            other = [s for s in C.STYLES if s != style][0]
            per_seed_terms = {}
            base_terms = None
            for name in feats[style]:
                X = feats[style][name][pooling]
                res = _revised_block(Z[:, C.POSITION_DIMS], X, unit_hash,
                                     split_seeds=args.split_seeds,
                                     d2_ks=D2_KS if args.full_d2 else (1, 12, 64, 256, 767),
                                     d3=_d3_cross_style(
                                         X, feats[other][name][pooling],
                                         Z[:, C.POSITION_DIMS],
                                         split_seed=args.split_seeds[0]))
                row = dict(stage="svgworld", pooling=plabel, style=style,
                           instance=name, **res["scalars"],
                           **{f"fs_{k}": v for k, v in
                              AE.feature_stats(X).items()})
                panel.append(row)
                for k, v in res["d2"].items():
                    d2_rows.append(dict(pooling=plabel, style=style,
                                        instance=name, k=k, S=v["S"], C=v["C"]))
                if name == "pre":
                    base_terms = res["terms_d1"]
                else:
                    per_seed_terms[int(name.replace("rand", ""))] = res["terms_d1"]
            if base_terms is not None and per_seed_terms:
                hb = AC.hierarchical_bootstrap(per_seed_terms, base_terms,
                                               n_boot=args.n_boot)
                one = AC.paired_bootstrap(base_terms,
                                          per_seed_terms[min(per_seed_terms)],
                                          n_boot=args.n_boot)
                ivals.append(dict(stage="svgworld", pooling=plabel, style=style,
                                  statistic="d1_gap_pre_minus_rand",
                                  hierarchical_point=hb["point"],
                                  hierarchical_lo=hb["ci_lo"],
                                  hierarchical_hi=hb["ci_hi"],
                                  hierarchical_excludes_zero=hb["excludes_zero"],
                                  n_control_seeds=hb["n_control_seeds"],
                                  single_seed_point=one["point"],
                                  single_seed_lo=one["ci_lo"],
                                  single_seed_hi=one["ci_hi"]))
    rows_to_csv(panel, os.path.join(od, "panel_d1_d2_d3.csv"))
    rows_to_csv(d2_rows, os.path.join(od, "d2_curves.csv"))
    rows_to_csv(ivals, os.path.join(od, "intervals.csv"))

    # legacy reproduction target: the submitted headline numbers
    report["legacy_reproduction"] = {
        r["condition"]: r["d1"] for r in ladder_rows if r["step"] == "baseline"}
    report["seconds"] = round(time.time() - t0, 1)
    jdump(report, os.path.join(od, "stage_report.json"))
    np.save(os.path.join(od, "Z.npy"), Z)
    return report


def _parity_row(Z: np.ndarray, X: np.ndarray, cond: str) -> dict:
    """Two independent implementations on identical arrays, at a fixed alpha.

    `audit_calc` is written from the manuscript's equations; `diagnostics` is the
    prior run's module. Agreement to numerical precision is the requirement.
    """
    import diagnostics as PRIOR

    tr, te = AC.unit_split(len(X), seed=0)
    Zt = np.asarray(Z[:, C.POSITION_DIMS], np.float64)
    Xd = np.asarray(X, np.float64)
    mine = AC.d1(Zt[tr], Xd[tr], Zt[te], Xd[te], alpha=1.0)
    theirs = PRIOR.d1_reverse(Zt[tr], Xd[tr], Zt[te], Xd[te], alpha=1.0)
    ks = (1, 12, 64, 256)
    my_S = AC.d2_S_curve(Zt[tr], Xd[tr], Zt[te], Xd[te], ks, alpha=1.0)["curve"]
    their_S = PRIOR.d2_curve(Zt[tr], Xd[tr], Zt[te], Xd[te], ks=ks,
                             alpha=1.0)["curve"]
    dS = max(abs(my_S[k]["S"] - their_S[k]["S"]) for k in ks if k in their_S)
    my_f = AC.forward_probe(Xd[tr], Zt[tr], Xd[te], Zt[te], alpha=1.0)
    their_f = PRIOR.forward_probe(Xd[tr], Zt[tr], Xd[te], Zt[te], alpha=1.0)
    return dict(condition=cond,
                d1_independent=mine["d1_raw"], d1_prior=theirs["d1_raw"],
                d1_abs_diff=abs(mine["d1_raw"] - theirs["d1_raw"]),
                ceiling_independent=mine["ceiling"], ceiling_prior=theirs["ceiling"],
                ceiling_abs_diff=abs(mine["ceiling"] - theirs["ceiling"]),
                d2_max_abs_diff=float(dS),
                forward_independent=my_f["r2_aggregate"],
                forward_prior=their_f["r2_aggregate"],
                forward_abs_diff=abs(my_f["r2_aggregate"] -
                                     their_f["r2_aggregate"]))


def _revised_block(Y: np.ndarray, X: np.ndarray, unit_hash: np.ndarray,
                   split_seeds=(0,), d2_ks=(1, 12, 64, 256, 767),
                   d3=None) -> dict:
    """D1, D2 (S and C), forward probe on a shared split, plus split audits."""
    Y = np.asarray(Y, np.float64)
    X = np.asarray(X, np.float64)
    per_split = []
    first = None
    for seed in split_seeds:
        tr, te = AC.unit_split(len(X), seed=seed)
        alpha, _ = AC.ridge_cv_alpha(Y[tr], X[tr], ALPHAS,
                                     folds=C.RIDGE_CV_FOLDS, seed=0)
        d1 = AC.d1(Y[tr], X[tr], Y[te], X[te], alpha=alpha)
        alpha_f, _ = AC.ridge_cv_alpha(X[tr], Y[tr], ALPHAS,
                                       folds=C.RIDGE_CV_FOLDS, seed=0)
        fwd = AC.forward_probe(X[tr], Y[tr], X[te], Y[te], alpha=alpha_f)
        rec = dict(d1_raw=d1["d1_raw"], d1_ceiling=d1["ceiling"],
                   d1_normalized=d1["d1_normalized"], alpha_d1=d1["alpha"],
                   forward_r2=fwd["r2_aggregate"],
                   forward_r2_mean=fwd["r2_mean"], alpha_forward=fwd["alpha"])
        per_split.append(rec)
        if first is None:
            S = AC.d2_S_curve(Y[tr], X[tr], Y[te], X[te], d2_ks, alpha=alpha)
            Ccur = AC.d2_C_curve(Y[tr], X[tr], Y[te], X[te], d2_ks, alpha=alpha)
            r_eff = AC.effective_rank(Y[tr])
            Sr = AC.d2_S_curve(Y[tr], X[tr], Y[te], X[te], [r_eff], alpha=alpha)
            Cr = AC.d2_C_curve(Y[tr], X[tr], Y[te], X[te], [r_eff], alpha=alpha)
            first = dict(
                terms_d1=d1["terms"],
                audit=AC.split_audit(tr, te, unit_hash),
                d2={k: dict(S=S["curve"][k]["S"], C=Ccur["curve"].get(k))
                    for k in S["curve"]},
                scalars=dict(rec, target_rank=int(r_eff),
                             S_at_r=Sr["curve"][min(r_eff, S["keep"])]["S"],
                             C_at_r=Cr["curve"][r_eff],
                             C_k50=Ccur["k50"], C_k90=Ccur["k90"],
                             n_train=int(len(tr)), n_test=int(len(te)),
                             content_overlap=AC.split_audit(
                                 tr, te, unit_hash)["content_overlap"]),
            )
    keys = ("d1_raw", "forward_r2")
    for k in keys:
        vals = [r[k] for r in per_split]
        first["scalars"][f"{k}_mean_over_splits"] = float(np.mean(vals))
        first["scalars"][f"{k}_std_over_splits"] = float(np.std(vals))
    if d3 is not None:
        first["scalars"].update(d3)
    return first


# ===========================================================================
# Stage: dataset
# ===========================================================================
def _target_variants(F: np.ndarray, spec: list[dict]):
    """Three defensible target encodings of the same factors.

    canonical    the revised run's encoding: cyclic factors as sin/cos over the
                 dataset's own period, ordinal factors min-max scaled,
                 categorical factors excluded.
    ordinal      every factor as its raw ordinal index, min-max scaled. No
                 trigonometric assumption anywhere.
    with_onehot  canonical plus one-hot columns for the categorical factors.
    """
    import factors as FA

    Y_can, labels_can, Y_cat, cat_labels = FA.encode_targets(F, spec)
    out = {"canonical": (Y_can, labels_can)}

    cols, labels = [], []
    for j, s in enumerate(spec):
        v = F[:, j].astype(np.float64)
        rng = v.max() - v.min()
        cols.append((v - v.min()) / (rng if rng > 0 else 1.0))
        labels.append(s["name"])
    out["ordinal"] = (np.stack(cols, axis=1), labels)

    if Y_cat.shape[1] > 0:
        out["with_onehot"] = (np.concatenate([Y_can, Y_cat], axis=1),
                              list(labels_can) + list(cat_labels))
    return out


def stage_dataset(args) -> dict:
    import datasets_extra as DX
    import factors as FA

    key = args.dataset
    od = out_dir(args.tag, key)
    device = "cuda" if args.device == "auto" and _cuda() else args.device
    if device == "auto":
        device = "cpu"
    spec = C.DATASETS[key]["factors"]
    report = dict(stage="dataset", dataset=key, device=device,
                  sample_seeds=list(args.sample_seeds), n=int(args.n),
                  rand_seeds=list(RAND_SEEDS[:args.n_rand_seeds]))
    rows, ivals, enc_rows = [], [], []
    t0 = time.time()

    for si, sample_seed in enumerate(args.sample_seeds):
        n = args.n if si == 0 else args.n_extra
        _force_sample_seed(FA, int(sample_seed))
        ds = DX.load_dataset(key, n_samples=n)
        images, F = ds["images"], ds["factors"]
        report.setdefault("manifest", {})[str(sample_seed)] = ds["manifest"]
        variants = _target_variants(F, spec)
        unit_hash = FA.combination_hash(F)
        tr_cb, te_cb, cb_info = FA.checkerboard_split(F, spec)
        cb = dict(train_idx=tr_cb, test_idx=te_cb)
        report.setdefault("checkerboard", {})[str(sample_seed)] = cb_info
        report.setdefault("n_used", {})[str(sample_seed)] = int(len(F))

        for model_key in args.models:
            insts = ([dict(name="pre", variant="pre", seed=0)] +
                     [dict(name=f"rand{s}", variant="rand", seed=s)
                      for s in RAND_SEEDS[:args.n_rand_seeds]])
            if C.MODELS[model_key]["arch"] == "resnet":
                insts += [dict(name=f"rand{s}_bncal", variant="rand", seed=s,
                               bn_calibrate=True) for s in RAND_SEEDS[:2]]
            # The D1 split for this sample is fixed by n and the split seed, so
            # the training rows are known before extraction and can be handed to
            # the BatchNorm calibration.
            calib_idx = AC.unit_split(len(images), seed=args.split_seeds[0])[0]
            got = AE.extract_instances(images, model_key, insts, device,
                                       batch_size=args.batch_size,
                                       calib_idx=calib_idx)
            for name, buf in got["features"].items():
                X = buf["pooled"]
                fs = AE.feature_stats(X)
                for vname, (Y, labels) in variants.items():
                    if vname != "canonical" and not args.all_encodings:
                        continue
                    res = _revised_block(
                        Y, X, unit_hash, split_seeds=args.split_seeds,
                        d2_ks=(1, 8, 16, 32, 64, 128, 256, 512),
                        d3=_d3_checkerboard(X, Y, cb, labels))
                    rows.append(dict(dataset=key, sample_seed=int(sample_seed),
                                     n=int(n), model=model_key, instance=name,
                                     encoding=vname, **res["scalars"],
                                     **{f"fs_{k}": v for k, v in fs.items()}))
                    # Terms are stashed per encoding: each encoding is a
                    # different target block, so its interval must be computed
                    # on its own residuals rather than relabelled.
                    _stash_terms(res, key, sample_seed, model_key, name, vname)
                    if vname == "canonical":
                        enc_rows.append(dict(
                            dataset=key, sample_seed=int(sample_seed),
                            model=model_key, instance=name,
                            **{k: v for k, v in got["info"][name].items()
                               if k in ("variant", "init_seed", "n_params",
                                        "observed_width")}))
            del got

        # hierarchical intervals per (model, encoding) for this sample seed
        for model_key in args.models:
            for vname in (["canonical"] if not args.all_encodings
                          else list(variants)):
                ivals += _intervals_for(rows, key, sample_seed, model_key, vname,
                                        args.n_boot)
    rows_to_csv(rows, os.path.join(od, "metrics.csv"))
    rows_to_csv(enc_rows, os.path.join(od, "instances.csv"))
    rows_to_csv(ivals, os.path.join(od, "intervals.csv"))
    report["seconds"] = round(time.time() - t0, 1)
    jdump(report, os.path.join(od, "stage_report.json"))
    return report


def _force_sample_seed(FA, seed: int):
    """The dataset loaders call `sample_factor_balanced` without a seed, so the
    sampling seed is a module default fixed at import. Rebind it so the same
    loader can produce three independent dataset samples."""
    orig = getattr(FA, "_sample_factor_balanced_orig", None)
    if orig is None:
        orig = FA.sample_factor_balanced
        FA._sample_factor_balanced_orig = orig

    def seeded(n_values, n_samples, seed=None, exclude=None, _s=int(seed)):
        return orig(n_values, n_samples, seed=_s, exclude=exclude)

    FA.sample_factor_balanced = seeded


_TERMS: dict[tuple, tuple] = {}


def _stash_terms(res, dataset, sample_seed, model_key, name, encoding):
    _TERMS[(dataset, int(sample_seed), model_key, name, encoding)] = \
        res["terms_d1"]


def _intervals_for(rows, dataset, sample_seed, model_key, vname, n_boot):
    base = _TERMS.get((dataset, int(sample_seed), model_key, "pre", vname))
    if base is None:
        return []
    per_seed = {s: _TERMS[(dataset, int(sample_seed), model_key, f"rand{s}",
                          vname)]
                for s in RAND_SEEDS
                if (dataset, int(sample_seed), model_key, f"rand{s}",
                    vname) in _TERMS}
    if not per_seed:
        return []
    hb = AC.hierarchical_bootstrap(per_seed, base, n_boot=n_boot)
    one = AC.paired_bootstrap(base, per_seed[min(per_seed)], n_boot=n_boot)
    out = [dict(dataset=dataset, sample_seed=int(sample_seed), model=model_key,
                encoding=vname, statistic="d1_gap_pre_minus_rand",
                hierarchical_point=hb["point"], hierarchical_lo=hb["ci_lo"],
                hierarchical_hi=hb["ci_hi"],
                hierarchical_excludes_zero=hb["excludes_zero"],
                n_control_seeds=hb["n_control_seeds"],
                single_seed_point=one["point"], single_seed_lo=one["ci_lo"],
                single_seed_hi=one["ci_hi"],
                single_seed_excludes_zero=one["excludes_zero"])]
    bn = {s: _TERMS[(dataset, int(sample_seed), model_key, f"rand{s}_bncal",
                     vname)]
          for s in RAND_SEEDS
          if (dataset, int(sample_seed), model_key, f"rand{s}_bncal",
              vname) in _TERMS}
    if bn:
        hb2 = AC.hierarchical_bootstrap(bn, base, n_boot=n_boot)
        out.append(dict(dataset=dataset, sample_seed=int(sample_seed),
                        model=model_key, encoding=vname,
                        statistic="d1_gap_pre_minus_rand_bncalibrated",
                        hierarchical_point=hb2["point"],
                        hierarchical_lo=hb2["ci_lo"], hierarchical_hi=hb2["ci_hi"],
                        hierarchical_excludes_zero=hb2["excludes_zero"],
                        n_control_seeds=hb2["n_control_seeds"]))
    return out


def _d3_cross_style(X_src, X_tgt, Y, split_seed: int = 0) -> dict:
    """D3 on SVG-World: fit the forward probe on one style's training scenes and
    evaluate on the paired style's held-out scenes. Row i is the same latent
    vector in both styles, and the split is over scene identity, so no scene
    appears on both sides in either style."""
    tr, te = AC.unit_split(len(Y), seed=split_seed)
    Xs = np.asarray(X_src, np.float64)
    Xt = np.asarray(X_tgt, np.float64)
    Y = np.asarray(Y, np.float64)
    alpha, _ = AC.ridge_cv_alpha(Xs[tr], Y[tr], ALPHAS,
                                 folds=C.RIDGE_CV_FOLDS, seed=0)
    out = AC.d3_transfer(Xs[tr], Y[tr], Xt[te], Y[te], alpha=alpha)
    ind = AC.forward_probe(Xs[tr], Y[tr], Xs[te], Y[te], alpha=alpha)
    return dict(d3_zero_shot_r2=out["zero_shot_r2_mean"],
                d3_rho=out["rho_mean"],
                d3_calibrated_r2=out["calibrated_r2_mean"],
                d3_in_support_r2=ind["r2_mean"], alpha_d3=out["alpha"])


def _d3_checkerboard(X, Y, cb, labels) -> dict:
    tr, te = cb["train_idx"], cb["test_idx"]
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    alpha, _ = AC.ridge_cv_alpha(X[tr], Y[tr], ALPHAS, folds=C.RIDGE_CV_FOLDS,
                                 seed=0)
    out = AC.d3_transfer(X[tr], Y[tr], X[te], Y[te], alpha=alpha, labels=labels)
    ind = AC.forward_probe(X[tr], Y[tr], X[tr], Y[tr], alpha=alpha)
    return dict(d3_zero_shot_r2=out["zero_shot_r2_mean"],
                d3_rho=out["rho_mean"],
                d3_calibrated_r2=out["calibrated_r2_mean"],
                d3_in_support_r2=ind["r2_mean"], alpha_d3=out["alpha"])


# ===========================================================================
# Stage: resnet_modes
# ===========================================================================
def _resnet_pass(model, images_u8, model_key, device, batch_size, mode,
                 order=None) -> np.ndarray:
    """One pooled-feature pass over the images in a given BatchNorm mode.

    mode "eval"  the module's own buffers are used (standard inference).
    mode "train" batch statistics are used. Buffers are snapshotted before and
                restored after by the caller, so nothing a train-mode pass
                computes can leak into any later condition.
    `order` optionally permutes the rows before batching, so batch COMPOSITION
    changes while the set of images does not; features are returned in the
    original row order either way.
    """
    import torch

    spec = C.MODELS[model_key]
    n = len(images_u8)
    idx = np.arange(n) if order is None else np.asarray(order)
    out = None
    bns = AE.bn_modules(model)
    model.eval()
    if mode == "train":
        for m in bns:
            m.train()
    with torch.no_grad():
        for s0 in range(0, n, batch_size):
            sel = idx[s0:s0 + batch_size]
            px = AE.preprocess_batch(images_u8[sel], spec["preproc"]).to(device)
            res, _ = AE._forward_pooled(model, px, spec, 0, ("pooled",))
            arr = res["pooled"].float().cpu().numpy()
            if out is None:
                out = np.empty((n, arr.shape[1]), np.float32)
            out[sel] = arr
    model.eval()
    return out


def _resnet_row(tag: str, condition: str, X_pre, X_rand, Y, unit_hash,
                extra: dict) -> dict:
    """D1, forward probe and spectrum descriptors for one paired condition."""
    tr, te = AC.unit_split(len(Y), seed=0)
    Y = np.asarray(Y, np.float64)
    row = dict(dataset=tag, condition=condition, n=int(len(Y)), **extra)
    stats = {}
    for name, X in (("pre", X_pre), ("rand", X_rand)):
        Xd = np.asarray(X, np.float64)
        alpha, _ = AC.ridge_cv_alpha(Y[tr], Xd[tr], ALPHAS,
                                     folds=C.RIDGE_CV_FOLDS, seed=0)
        d1 = AC.d1(Y[tr], Xd[tr], Y[te], Xd[te], alpha=alpha)
        alpha_f, _ = AC.ridge_cv_alpha(Xd[tr], Y[tr], ALPHAS,
                                       folds=C.RIDGE_CV_FOLDS, seed=0)
        fwd = AC.forward_probe(Xd[tr], Y[tr], Xd[te], Y[te], alpha=alpha_f)
        fs = AE.feature_stats(Xd)
        stats[name] = dict(d1=d1["d1_raw"], terms=d1["terms"],
                           forward=fwd["r2_aggregate"], **fs)
        for k in ("d1", "forward", "participation_ratio", "pc1_share",
                  "feature_norm_mean", "effective_rank_99"):
            row[f"{name}_{k}"] = stats[name][k]
    row["gap"] = row["pre_d1"] - row["rand_d1"]
    ci = AC.paired_bootstrap(stats["pre"]["terms"], stats["rand"]["terms"],
                             n_boot=2000)
    row.update(gap_lo=ci["ci_lo"], gap_hi=ci["ci_hi"],
               gap_excludes_zero=ci["excludes_zero"])
    return row


def stage_resnet_modes(args) -> dict:
    """Does a train-mode (batch-statistics) BatchNorm control reach the same
    conclusion as the calibrated-eval control, and does either depend on batch
    composition?

    Everything is held identical across conditions: the same pretrained and
    random weights, the same images in the same order, the same split, the same
    preprocessing and the same probe code. Only the BatchNorm mode changes.
    """
    import datasets_extra as DX
    import factors as FA

    od = out_dir(args.tag, "resnet_modes")
    device = "cuda" if args.device == "auto" and _cuda() else args.device
    if device == "auto":
        device = "cpu"
    model_key = "resnet50"
    rows, report = [], dict(stage="resnet_modes", device=device,
                           model=model_key, datasets=list(args.mode_datasets),
                           batch_sizes=[args.batch_size, args.small_batch])

    for tag in args.mode_datasets:
        if tag == "svgworld":
            import svgworld_data as SW

            Z = SW.sample_latents(args.n, seed=C.DATA_SEED)
            images = SW.render_style(Z, "island", size=C.IMAGE_SIZE)
            Y = Z[:, C.POSITION_DIMS]
            unit_hash = np.arange(len(Y))
        else:
            _force_sample_seed(FA, 1234)
            ds = DX.load_dataset(tag, n_samples=args.n)
            images = ds["images"]
            Y = _target_variants(ds["factors"],
                                 C.DATASETS[tag]["factors"])["canonical"][0]
            unit_hash = FA.combination_hash(ds["factors"])

        pre, _, _ = AE.build_instance(model_key, "pre", 0, device)
        rnd, _, _ = AE.build_instance(model_key, "rand", 42, device)
        snap_pre, snap_rnd = AE.snapshot_bn(pre), AE.snapshot_bn(rnd)
        restore_residual = {}

        # (A) both models in standard eval mode with their existing buffers
        Xp_eval = _resnet_pass(pre, images, model_key, device, args.batch_size,
                               "eval")
        Xr_eval = _resnet_pass(rnd, images, model_key, device, args.batch_size,
                               "eval")
        rows.append(_resnet_row(tag, "A_both_eval_existing_buffers", Xp_eval,
                                Xr_eval, Y, unit_hash,
                                dict(batch_size=args.batch_size,
                                     batch_order="natural", fair=True)))

        # (B) both models in train mode, batch statistics, two batch sizes and
        #     one shuffled batch composition; buffers restored after each pass
        train_feats = {}
        for bs, order_name in ((args.batch_size, "natural"),
                              (args.small_batch, "natural"),
                              (args.small_batch, "shuffled")):
            order = None
            if order_name == "shuffled":
                order = np.random.default_rng(1234).permutation(len(images))
            Xp = _resnet_pass(pre, images, model_key, device, bs, "train",
                              order)
            restore_residual[f"pre_bs{bs}_{order_name}"] = AE.restore_bn(
                pre, snap_pre)
            Xr = _resnet_pass(rnd, images, model_key, device, bs, "train",
                              order)
            restore_residual[f"rand_bs{bs}_{order_name}"] = AE.restore_bn(
                rnd, snap_rnd)
            rows.append(_resnet_row(
                tag, "B_both_train_batch_statistics", Xp, Xr, Y, unit_hash,
                dict(batch_size=bs, batch_order=order_name, fair=True)))
            train_feats[(bs, order_name)] = (Xp, Xr)

        # (C) both eval, random model's buffers calibrated on training rows only
        calib_idx = AC.unit_split(len(images), seed=0)[0]
        cal = AE.calibrate_batchnorm(rnd, images[calib_idx], model_key, device,
                                    args.batch_size)
        Xr_cal = _resnet_pass(rnd, images, model_key, device, args.batch_size,
                              "eval")
        rows.append(_resnet_row(
            tag, "C_both_eval_random_calibrated", Xp_eval, Xr_cal, Y,
            unit_hash, dict(batch_size=args.batch_size,
                            batch_order="natural", fair=True)))
        AE.restore_bn(rnd, snap_rnd)
        report.setdefault("calibration", {})[tag] = cal

        # (D) asymmetric, reported ONLY to show why it is not a fair comparison
        Xr_train = train_feats[(args.batch_size, "natural")][1]
        rows.append(_resnet_row(
            tag, "D_asymmetric_pre_eval_vs_rand_train", Xp_eval, Xr_train, Y,
            unit_hash, dict(batch_size=args.batch_size,
                            batch_order="natural", fair=False)))

        report.setdefault("bn_restore_max_abs_residual", {})[tag] = \
            restore_residual
        del pre, rnd
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()

    rows_to_csv(rows, os.path.join(od, "resnet_modes.csv"))
    jdump(report, os.path.join(od, "stage_report.json"))
    return report


# ===========================================================================
def _cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def stage_smoke(args) -> dict:
    """The whole pipeline at a tiny size, through the same functions."""
    rep = {"env": dict(python=platform.python_version(), host=platform.node())}
    import torch

    rep["env"].update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                      device_name=(torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None))
    sub = argparse.Namespace(**vars(args))
    sub.n, sub.n_alt, sub.n_rand_seeds = 192, 24, 2
    sub.split_seeds, sub.n_boot, sub.full_d2 = (0,), 500, False
    sub.tag = args.tag
    rep["svgworld"] = stage_svgworld(sub)
    sub2 = argparse.Namespace(**vars(sub))
    sub2.n, sub2.n_extra = 384, 256
    sub2.sample_seeds = (1234, 7)
    sub2.models = ("dinov3_b16", "resnet50")
    sub2.all_encodings = True
    for ds in args.smoke_datasets:
        sub2.dataset = ds
        rep[ds] = stage_dataset(sub2)
    od = out_dir(args.tag, "smoke")
    jdump(rep, os.path.join(od, "smoke_report.json"))
    print("[smoke] complete", flush=True)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["smoke", "svgworld", "dataset", "resnet_modes"])
    p.add_argument("--mode-datasets",
                   default="svgworld,shapes3d,dsprites,mpi3d_realistic")
    p.add_argument("--small-batch", type=int, default=64)
    p.add_argument("--tag", default="dev")
    p.add_argument("--dataset", default="shapes3d")
    p.add_argument("--n", type=int, default=C.N_SCENES_FULL)
    p.add_argument("--n-extra", type=int, default=10000,
                   help="sample size for the second and third dataset seeds")
    p.add_argument("--n-alt", type=int, default=256,
                   help="scenes rendered with the second rasterizer")
    p.add_argument("--n-rand-seeds", type=int, default=5)
    p.add_argument("--sample-seeds", default=",".join(
        str(s) for s in DATASET_SAMPLE_SEEDS))
    p.add_argument("--models", default="dinov3_b16,clip_b16,resnet50")
    p.add_argument("--smoke-datasets", default="shapes3d")
    p.add_argument("--split-seeds", default="0,1,2,3,4")
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--all-encodings", action="store_true")
    p.add_argument("--full-d2", action="store_true")
    args = p.parse_args()
    args.sample_seeds = tuple(int(s) for s in str(args.sample_seeds).split(","))
    args.models = tuple(m for m in str(args.models).split(",") if m)
    args.smoke_datasets = tuple(d for d in str(args.smoke_datasets).split(",") if d)
    args.split_seeds = tuple(int(s) for s in str(args.split_seeds).split(","))
    args.mode_datasets = tuple(d for d in str(args.mode_datasets).split(",") if d)
    if args.stage == "smoke":
        stage_smoke(args)
    elif args.stage == "svgworld":
        stage_svgworld(args)
    elif args.stage == "resnet_modes":
        stage_resnet_modes(args)
    else:
        stage_dataset(args)


if __name__ == "__main__":
    main()
