"""One entrypoint for every lane, and for the shared end-to-end smoke.

    python src/run_lane.py --lane svgworld        [--scale smoke|full]
    python src/run_lane.py --lane shapes3d
    python src/run_lane.py --lane dsprites
    python src/run_lane.py --lane mpi3d_realistic
    python src/run_lane.py --lane smoke           # all of the above, tiny

Compute discipline. The GPU is used for model load plus inference only: batches
are preprocessed on device, pooling is online, each model is loaded once and
reused across all batches, and the forward runs under `inference_mode`. Every
other stage is CPU work, parallelized across the sandbox's cores so it costs
seconds rather than a slice of the GPU hour: rendering and dataset preparation
run in a process pool, MPI3D's range fetches in a thread pool. Figures and the
cross-lane synthesis happen outside the job entirely, from the small CSV and
JSON outputs this script writes.

Preparation runs inside the same job as feature extraction, in parallel CPU
stages; `stage_seconds` in every lane report records the time spent per stage.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import diagnostics as D
import encoders as E
import factors as FA
import rebuttal_config as C
import svgworld_data as SW

TIMERS: dict[str, float] = {}


class stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"\n===== stage {self.name} =====", flush=True)
        return self

    def __exit__(self, *a):
        dt = time.time() - self.t0
        TIMERS[self.name] = round(TIMERS.get(self.name, 0.0) + dt, 2)
        print(f"===== stage {self.name} done in {dt:.1f}s =====", flush=True)


def out_dir(lane: str) -> str:
    d = os.path.join(C.artifacts_dir(), lane)
    os.makedirs(d, exist_ok=True)
    return d


def dump_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    print(f"[write] {path} ({os.path.getsize(path)} bytes)", flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(_csv_cell(r.get(k)) for k in keys) + "\n")
    print(f"[write] {path} ({len(rows)} rows)", flush=True)


def _csv_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return '"' + ";".join(str(x) for x in v) + '"'
    s = str(v)
    return f'"{s}"' if "," in s else s


def n_workers() -> int:
    return max(1, min(32, (os.cpu_count() or 4)))


# ---------------------------------------------------------------------------
# Parallel rendering (CPU)
# ---------------------------------------------------------------------------
_RENDER_CTX: dict = {}


def _render_init(style: str, size: int):
    _RENDER_CTX["style"] = style
    _RENDER_CTX["size"] = size


def _render_chunk(args):
    lo, Zc = args
    from svgworld import raster
    style, size = _RENDER_CTX["style"], _RENDER_CTX["size"]
    mod = SW.STYLE_MODULES[style]
    out = np.empty((len(Zc), size, size, 3), dtype=np.uint8)
    for i in range(len(Zc)):
        svg = SW.square_pad_svg(mod.generate_scene_svg(Zc[i]))
        out[i] = raster.svg_to_rgb_array(svg, size=size)
    return lo, out


def render_style_parallel(Z: np.ndarray, style: str, size: int = C.IMAGE_SIZE,
                          chunk: int = 100) -> np.ndarray:
    import multiprocessing as mp

    n = Z.shape[0]
    jobs = [(lo, Z[lo:lo + chunk]) for lo in range(0, n, chunk)]
    out = np.empty((n, size, size, 3), dtype=np.uint8)
    nw = n_workers()
    t0 = time.time()
    if nw == 1 or len(jobs) == 1:
        _render_init(style, size)
        for j in jobs:
            lo, arr = _render_chunk(j)
            out[lo:lo + len(arr)] = arr
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(nw, initializer=_render_init, initargs=(style, size)) as pool:
            done = 0
            for lo, arr in pool.imap_unordered(_render_chunk, jobs):
                out[lo:lo + len(arr)] = arr
                done += len(arr)
                if done % 2000 < chunk:
                    print(f"[render {style}] {done}/{n} "
                          f"({done / max(time.time() - t0, 1e-9):.0f} scenes/s, "
                          f"{nw} workers)", flush=True)
    dt = time.time() - t0
    print(f"[render {style}] {n} scenes in {dt:.1f}s "
          f"({n / max(dt, 1e-9):.0f} scenes/s, {nw} workers)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Batch-size tuning and autocast validation (inside the smoke only)
# ---------------------------------------------------------------------------
def tune_batch_size(images_u8, model_key: str, device: str,
                    candidates=(64, 128, 256), warmup: int = 2) -> dict:
    """Warmed throughput per candidate batch size, on the real entrypoint path."""
    import torch

    spec = C.MODELS[model_key]
    model, info = E.build_encoder(model_key, "pre", device)
    pp = spec["preproc"]
    results = {}
    for bs in candidates:
        if bs > len(images_u8):
            continue
        try:
            for _ in range(warmup):
                px = E.preprocess_batch_torch(images_u8[:bs], pp, device)
                E._forward_pooled(model, px, spec, info["n_register_tokens"], ("pooled",))
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            reps = 3
            for _ in range(reps):
                px = E.preprocess_batch_torch(images_u8[:bs], pp, device)
                E._forward_pooled(model, px, spec, info["n_register_tokens"], ("pooled",))
            if device == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / reps
            results[bs] = dict(
                seconds_per_batch=round(dt, 4),
                images_per_second=round(bs / dt, 1),
                peak_mem_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                             if device == "cuda" else None))
        except torch.cuda.OutOfMemoryError:
            results[bs] = dict(oom=True)
            torch.cuda.empty_cache()
            break
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    ok = {k: v for k, v in results.items() if not v.get("oom")}
    best = max(ok, key=lambda k: ok[k]["images_per_second"]) if ok else 64
    return dict(model=model_key, per_batch=results, chosen=int(best))


def validate_autocast(images_u8, model_key: str, device: str,
                      Y: np.ndarray | None = None, batch_size: int = 128) -> dict:
    """Compare bf16 autocast against fp32 on the SAME images, on features and on
    the D1 statistic. bf16 is used in production only if both agree closely."""
    ref = E.extract_features(images_u8, model_key, "pre", device,
                             batch_size=batch_size, dtype="fp32",
                             log_every=0)["features"]["pooled"]
    try:
        got = E.extract_features(images_u8, model_key, "pre", device,
                                 batch_size=batch_size, dtype="bf16",
                                 log_every=0)["features"]["pooled"]
    except Exception as e:
        return dict(model=model_key, bf16_failed=f"{type(e).__name__}: {e}",
                    accept_bf16=False)
    scale = float(np.abs(ref).mean()) + 1e-12
    rel = float(np.abs(ref - got).mean() / scale)
    rep = dict(model=model_key, mean_abs_feature_diff=float(np.abs(ref - got).mean()),
               relative_mean_abs_diff=rel,
               max_abs_feature_diff=float(np.abs(ref - got).max()))
    if Y is not None and len(Y) == len(ref):
        tr, te = D.unit_split(len(ref), seed=0)
        a = D.d1_reverse(Y[tr], ref[tr], Y[te], ref[te])
        b = D.d1_reverse(Y[tr], got[tr], Y[te], got[te])
        rep.update(d1_fp32=a["d1_raw"], d1_bf16=b["d1_raw"],
                   d1_abs_delta=abs(a["d1_raw"] - b["d1_raw"]))
        rep["accept_bf16"] = bool(rel < 5e-3 and rep["d1_abs_delta"] < 2e-3)
    else:
        rep["accept_bf16"] = bool(rel < 5e-3)
    return rep


# ---------------------------------------------------------------------------
# Diagnostics over a set of feature blocks
# ---------------------------------------------------------------------------
def run_diagnostics(features: dict, Y: np.ndarray, labels: list[str],
                    unit_hash: np.ndarray, lane: str,
                    split_seeds=C.SPLIT_SEEDS,
                    d3: dict | None = None,
                    Y_cat: np.ndarray | None = None,
                    cat_labels: list[str] | None = None) -> dict:
    """features: {condition_key: {block_name: X}}; Y: headline target block.

    `d3` describes the out-of-distribution test:
      {"kind": "cross_style", "pairs": [(src_key, tgt_key), ...]}
      {"kind": "checkerboard", "train": idx, "test": idx, "info": {...}}
    """
    d1_rows, d2_rows, fwd_rows, d3_rows, curve_rows = [], [], [], [], []
    split_audits = []
    boot: dict[str, dict] = {}

    splits = [D.unit_split(len(Y), seed=s) for s in split_seeds]
    for s, (tr, te) in zip(split_seeds, splits):
        split_audits.append(dict(split_seed=s, **D.audit_split(tr, te, unit_hash)))

    for cond, blocks in sorted(features.items()):
        for block, X in sorted(blocks.items()):
            for si, (tr, te) in zip(split_seeds, splits):
                d1 = D.d1_reverse(Y[tr], X[tr], Y[te], X[te])
                row = dict(lane=lane, condition=cond, block=block, split_seed=si,
                           n_train=len(tr), n_test=len(te), width=X.shape[1],
                           **{k: v for k, v in d1.items() if k != "terms"})
                d1_rows.append(row)
                if si == split_seeds[0]:
                    boot[f"d1|{cond}|{block}"] = d1["terms"]

                d2 = D.d2_curve(Y[tr], X[tr], Y[te], X[te])
                d2_rows.append(dict(
                    lane=lane, condition=cond, block=block, split_seed=si,
                    alpha=d2["alpha"], r_eff=d2["r_eff"],
                    n_components_kept=d2["n_components_kept"],
                    S_at_r_eff=d2["S_at_r_eff"],
                    eigenvalue_total=d2["eigenvalue_total"],
                    eigenvalues_top20=d2["eigenvalues_top20"]))
                for k, v in d2["curve"].items():
                    curve_rows.append(dict(lane=lane, condition=cond, block=block,
                                           split_seed=si, k=k, S=v["S"]))
                    if si == split_seeds[0]:
                        boot[f"d2|{cond}|{block}|{k}"] = v["terms"]

                fp = D.forward_probe(X[tr], Y[tr], X[te], Y[te], labels=labels)
                fwd_rows.append(dict(
                    lane=lane, condition=cond, block=block, split_seed=si,
                    kind="within", alpha=fp["alpha"], r2_mean=fp["r2_mean"],
                    r2_aggregate=fp["r2_aggregate"],
                    r2_per_target=fp["r2_per_target"],
                    rho_per_target=fp["rho_per_target"], labels=labels))
                if si == split_seeds[0]:
                    boot[f"fwd|{cond}|{block}"] = fp["terms"]

                if Y_cat is not None and Y_cat.shape[1] and si == split_seeds[0]:
                    fpc = D.forward_probe(X[tr], Y_cat[tr], X[te], Y_cat[te],
                                          labels=cat_labels)
                    fwd_rows.append(dict(
                        lane=lane, condition=cond, block=block, split_seed=si,
                        kind="within_categorical", alpha=fpc["alpha"],
                        r2_mean=fpc["r2_mean"], r2_aggregate=fpc["r2_aggregate"],
                        r2_per_target=fpc["r2_per_target"],
                        rho_per_target=fpc["rho_per_target"], labels=cat_labels))
                print(f"[diag {lane}] {cond}/{block} split{si} "
                      f"D1={d1['d1_raw']:+.4f} (ceil {d1['ceiling']:.3f}, "
                      f"norm {d1['d1_normalized']:+.3f}) "
                      f"S(r_eff)={d2['S_at_r_eff']:+.4f} "
                      f"fwd={fp['r2_mean']:+.4f}", flush=True)

    # ---- D3
    if d3 and d3["kind"] == "cross_style":
        for src, tgt in d3["pairs"]:
            for block in sorted(set(features[src]) & set(features[tgt])):
                for si, (tr, te) in zip(split_seeds, splits):
                    r = D.d3_cross_style(features[src][block][tr], Y[tr],
                                         features[tgt][block][te], Y[te],
                                         labels=labels)
                    d3_rows.append(dict(
                        lane=lane, kind="cross_style", source=src, target=tgt,
                        block=block, split_seed=si, alpha=r["alpha"],
                        r2_mean=r["r2_mean"], r2_aggregate=r["r2_aggregate"],
                        r2_per_target=r["r2_per_target"],
                        rho_per_target=r["rho_per_target"], labels=labels))
                    if si == split_seeds[0]:
                        boot[f"d3|{src}->{tgt}|{block}"] = r["terms"]
                    print(f"[diag {lane}] D3 {src}->{tgt}/{block} split{si} "
                          f"fwd={r['r2_mean']:+.4f}", flush=True)
    elif d3 and d3["kind"] == "checkerboard":
        tr, te = d3["train"], d3["test"]
        for cond, blocks in sorted(features.items()):
            for block, X in sorted(blocks.items()):
                r = D.forward_probe(X[tr], Y[tr], X[te], Y[te], labels=labels)
                base = D.forward_probe(*_iid_reference(X, Y, len(tr)))
                d3_rows.append(dict(
                    lane=lane, kind="checkerboard", condition=cond, block=block,
                    split_seed=0, alpha=r["alpha"], r2_mean=r["r2_mean"],
                    r2_aggregate=r["r2_aggregate"],
                    r2_per_target=r["r2_per_target"],
                    rho_per_target=r["rho_per_target"], labels=labels,
                    iid_reference_r2_mean=base["r2_mean"],
                    **{f"cb_{k}": v for k, v in d3["info"].items()}))
                boot[f"d3|{cond}|{block}"] = r["terms"]
                boot[f"d3iid|{cond}|{block}"] = base["terms"]
                print(f"[diag {lane}] D3 checkerboard {cond}/{block} "
                      f"held-out={r['r2_mean']:+.4f} "
                      f"iid-ref={base['r2_mean']:+.4f}", flush=True)

    return dict(d1=d1_rows, d2=d2_rows, d2_curve=curve_rows, forward=fwd_rows,
                d3=d3_rows, split_audits=split_audits, boot_terms=boot)


def _iid_reference(X, Y, n_train, seed=0):
    """A random split of the same size as the checkerboard training half, so the
    held-out-combination number has an in-distribution reference at equal n."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    tr, te = perm[:n_train], perm[n_train:]
    return X[tr], Y[tr], X[te], Y[te]


def paired_intervals(boot_terms: dict, lane: str,
                     pairs: list[tuple[str, str, str]]) -> list[dict]:
    """95% paired bootstrap intervals for pretrained-minus-random differences."""
    rows = []
    for label, key_pre, key_rand in pairs:
        if key_pre not in boot_terms or key_rand not in boot_terms:
            continue
        ci = D.bootstrap_ci_chunked(boot_terms[key_pre], boot_terms[key_rand])
        abs_pre = D.bootstrap_ci_chunked(boot_terms[key_pre])
        abs_rand = D.bootstrap_ci_chunked(boot_terms[key_rand])
        rows.append(dict(lane=lane, statistic=label,
                         diff=ci["point"], diff_ci_lo=ci["ci_lo"],
                         diff_ci_hi=ci["ci_hi"],
                         diff_excludes_zero=ci["excludes_zero"],
                         pretrained=abs_pre["point"],
                         pretrained_ci_lo=abs_pre["ci_lo"],
                         pretrained_ci_hi=abs_pre["ci_hi"],
                         random=abs_rand["point"],
                         random_ci_lo=abs_rand["ci_lo"],
                         random_ci_hi=abs_rand["ci_hi"],
                         n_units=ci["n_units"], n_boot=ci["n_boot"]))
        print(f"[ci {lane}] {label}: diff {ci['point']:+.4f} "
              f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] "
              f"pre {abs_pre['point']:+.4f} rand {abs_rand['point']:+.4f}",
              flush=True)
    return rows


def save_features(feats: dict, d: str, tag: str):
    """float16 archive for reuse; diagnostics always run on the float32 originals.

    Written under `features/` so a truncated artifact upload cannot displace the
    metrics tables, which are the actual evidence.
    """
    sub = os.path.join(d, "features")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"features_{tag}.npz")
    np.savez(path, **{k: v.astype(np.float16) for k, v in feats.items()})
    print(f"[write] {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Lane: SVG-World model panel
# ---------------------------------------------------------------------------
def lane_svgworld(n_scenes: int, models: tuple[str, ...], device: str,
                  batch_size: int, dtype: str, lane: str = "svgworld",
                  with_simulation: bool = True, sim_kw: dict | None = None,
                  split_seeds=C.SPLIT_SEEDS, n_png_keep: int = 24,
                  tag: str = "", keep_features: bool = False) -> dict:
    lane = f"{lane}_{tag}" if tag else lane
    d = out_dir(lane)
    report: dict = dict(lane=lane, n_scenes=int(n_scenes), models=list(models),
                        device=device, batch_size=batch_size, dtype=dtype,
                        preproc_backend=C.PREPROC_BACKEND)

    if with_simulation:
        import simulation as S
        with stage("simulation"):
            kw = dict(widths=C.SIM["widths"], weight_decays=C.SIM["weight_decays"],
                      n_seeds=C.SIM["n_seeds"])
            kw.update(sim_kw or {})
            sim = S.run_sweep(device=device, out_dir=d, **kw)
            report["simulation"] = dict(
                lift_linearity_reverse_r2=sim["lift_linearity_reverse_r2"],
                n_rows=len(sim["rows"]), wall_seconds=sim["wall_seconds"],
                widths=sim["widths"], n_seeds=sim["n_seeds"])
            report["simulation_reload_ok"] = _check_json_reload(
                os.path.join(d, "simulation_sweep.json"), "rows")

    with stage("render_checks"):
        import render_checks as RC
        report["render_checks"] = RC.run_all()
        print(json.dumps(report["render_checks"], indent=1, default=_json_default),
              flush=True)

    with stage("render"):
        Z = SW.sample_latents(n_scenes)
        np.save(os.path.join(d, "Z.npy"), Z)
        images = {}
        for style in C.STYLES:
            images[style] = render_style_parallel(Z, style)
        report["render_report"] = SW.render_report(images)
        png_dir = os.path.join(d, "example_pngs")
        os.makedirs(png_dir, exist_ok=True)
        from svgworld import raster
        for style in C.STYLES:
            mod = SW.STYLE_MODULES[style]
            for i in range(min(n_png_keep, n_scenes)):
                with open(os.path.join(png_dir, f"{style}_{i:05d}.png"), "wb") as f:
                    f.write(raster.svg_to_png_bytes(
                        SW.square_pad_svg(mod.generate_scene_svg(Z[i]))))

    Y = Z[:, C.POSITION_DIMS].astype(np.float64)
    labels = [C.LATENT_LABELS[i] for i in C.POSITION_DIMS]
    unit_hash = np.asarray([hash(Z[i].tobytes()) for i in range(len(Z))])
    Y_sens = {name: Z[:, dims].astype(np.float64)
              for name, dims in C.LATENT_SUBSETS.items()}

    features: dict[str, dict[str, np.ndarray]] = {}
    with stage("extract"):
        for mk in models:
            for variant in C.VARIANTS:
                res = E.extract_features_multi(
                    images, mk, variant, device,
                    batch_size=batch_size,
                    extra_poolings=C.EXTRA_POOLING.get(mk, ()), dtype=dtype)
                for style in C.STYLES:
                    key = f"{mk}|{variant}|{style}"
                    features[key] = res["features"][style]
                    report.setdefault("encoder_info", {})[key] = res["info"][style]
                    report.setdefault("feature_report", {})[key] = \
                        E.feature_report(res["features"][style])
                    if keep_features:
                        save_features(res["features"][style], d,
                                      key.replace("|", "_"))
        for mk in models:
            for style in C.STYLES:
                report.setdefault("variant_distinctness", {})[f"{mk}|{style}"] = \
                    E.variant_distinctness(features[f"{mk}|pre|{style}"]["pooled"],
                                           features[f"{mk}|rand|{style}"]["pooled"])

    with stage("diagnostics"):
        d3 = dict(kind="cross_style",
                  pairs=[(f"{mk}|{v}|island", f"{mk}|{v}|western")
                         for mk in models for v in C.VARIANTS]
                        + [(f"{mk}|{v}|western", f"{mk}|{v}|island")
                           for mk in models for v in C.VARIANTS])
        res = run_diagnostics(features, Y, labels, unit_hash, lane,
                              split_seeds=split_seeds, d3=d3)
        ci_pairs = []
        for mk in models:
            for s in C.STYLES:
                for block in sorted(features[f"{mk}|pre|{s}"]):
                    ci_pairs += [
                        (f"D1_raw|{mk}|{s}|{block}", f"d1|{mk}|pre|{s}|{block}",
                         f"d1|{mk}|rand|{s}|{block}"),
                        (f"forward|{mk}|{s}|{block}", f"fwd|{mk}|pre|{s}|{block}",
                         f"fwd|{mk}|rand|{s}|{block}"),
                    ]
            for block in sorted(features[f"{mk}|pre|island"]):
                ci_pairs.append((f"D3_island_to_western|{mk}|{block}",
                                 f"d3|{mk}|pre|island->{mk}|pre|western|{block}",
                                 f"d3|{mk}|rand|island->{mk}|rand|western|{block}"))
                ci_pairs.append((f"D3_western_to_island|{mk}|{block}",
                                 f"d3|{mk}|pre|western->{mk}|pre|island|{block}",
                                 f"d3|{mk}|rand|western->{mk}|rand|island|{block}"))
        ci_rows = paired_intervals(res["boot_terms"], lane, ci_pairs)

    with stage("sensitivity"):
        sens_rows = []
        for mk in models:
            for s in C.STYLES:
                X = features[f"{mk}|pre|{s}"]["pooled"]
                Xr = features[f"{mk}|rand|{s}"]["pooled"]
                tr, te = D.unit_split(len(Y), seed=split_seeds[0])
                for name, Ys in Y_sens.items():
                    for variant, Xv in (("pre", X), ("rand", Xr)):
                        r = D.d1_reverse(Ys[tr], Xv[tr], Ys[te], Xv[te])
                        sens_rows.append(dict(
                            lane=lane, model=mk, style=s, variant=variant,
                            latent_subset=name, n_latent_columns=Ys.shape[1],
                            **{k: v for k, v in r.items() if k != "terms"}))
        write_csv(os.path.join(d, "sensitivity_latent_subsets.csv"), sens_rows)

    _write_lane_outputs(d, lane, res, ci_rows, report)
    return report


# ---------------------------------------------------------------------------
# Lane: an added disentanglement dataset
# ---------------------------------------------------------------------------
def lane_dataset(key: str, n_samples: int, models: tuple[str, ...], device: str,
                 batch_size: int, dtype: str, split_seeds=C.SPLIT_SEEDS,
                 tag: str = "", keep_features: bool = True) -> dict:
    import datasets_extra as DX

    lane = f"{key}_{tag}" if tag else key
    d = out_dir(lane)
    report: dict = dict(lane=lane, dataset=key, n_samples=int(n_samples),
                        models=list(models), device=device,
                        batch_size=batch_size, dtype=dtype,
                        preproc_backend=C.PREPROC_BACKEND)

    with stage("dataset_prepare"):
        ds = DX.load_dataset(key, n_samples=n_samples)
        report["manifest"] = ds["manifest"]
        np.save(os.path.join(d, "source_index.npy"), ds["source_index"])
        np.save(os.path.join(d, "factors.npy"), ds["factors"])
        dump_json(ds["manifest"], os.path.join(d, "dataset_manifest.json"))
        print(json.dumps({k: v for k, v in ds["manifest"].items()
                          if k not in ("marginal_balance",)},
                         indent=1, default=_json_default), flush=True)

    spec = C.DATASETS[key]["factors"]
    F = ds["factors"]
    Y, labels, Y_cat, cat_labels = FA.encode_targets(F, spec)
    unit_hash = FA.combination_hash(F)
    tr_cb, te_cb, cb_info = FA.checkerboard_split(F, spec)
    report["checkerboard"] = cb_info
    print(f"[split] checkerboard over {cb_info['factor_a']} x {cb_info['factor_b']}: "
          f"{cb_info['n_train']} train / {cb_info['n_test']} test, "
          f"every marginal in train={cb_info['every_marginal_in_train']}, "
          f"combination overlap={cb_info['combination_overlap']}", flush=True)
    report["target_block"] = dict(labels=labels, n_columns=int(Y.shape[1]),
                                 categorical_labels=cat_labels,
                                 n_categorical_columns=int(Y_cat.shape[1]))

    images = ds["images"]
    features: dict[str, dict[str, np.ndarray]] = {}
    with stage("extract"):
        for mk in models:
            for variant in C.VARIANTS:
                res = E.extract_features(images, mk, variant, device,
                                         batch_size=batch_size, dtype=dtype)
                key_c = f"{mk}|{variant}"
                features[key_c] = res["features"]
                report.setdefault("encoder_info", {})[key_c] = res["info"]
                report.setdefault("feature_report", {})[key_c] = \
                    E.feature_report(res["features"])
                if keep_features:
                    save_features(res["features"], d, key_c.replace("|", "_"))
                print(f"[extract] {lane}/{key_c} "
                      f"{res['info']['images_per_second']} img/s", flush=True)
        for mk in models:
            report.setdefault("variant_distinctness", {})[mk] = \
                E.variant_distinctness(features[f"{mk}|pre"]["pooled"],
                                       features[f"{mk}|rand"]["pooled"])

    with stage("diagnostics"):
        res = run_diagnostics(features, Y, labels, unit_hash, lane,
                              split_seeds=split_seeds,
                              d3=dict(kind="checkerboard", train=tr_cb, test=te_cb,
                                      info=cb_info),
                              Y_cat=Y_cat, cat_labels=cat_labels)
        ci_pairs = []
        for mk in models:
            for block in sorted(features[f"{mk}|pre"]):
                ci_pairs += [
                    (f"D1_raw|{mk}|{block}", f"d1|{mk}|pre|{block}",
                     f"d1|{mk}|rand|{block}"),
                    (f"forward|{mk}|{block}", f"fwd|{mk}|pre|{block}",
                     f"fwd|{mk}|rand|{block}"),
                    (f"D3_checkerboard|{mk}|{block}", f"d3|{mk}|pre|{block}",
                     f"d3|{mk}|rand|{block}"),
                ]
        ci_rows = paired_intervals(res["boot_terms"], lane, ci_pairs)

    _write_lane_outputs(d, lane, res, ci_rows, report)
    return report


def _write_lane_outputs(d: str, lane: str, res: dict, ci_rows: list[dict],
                        report: dict):
    write_csv(os.path.join(d, "metrics_d1.csv"), res["d1"])
    write_csv(os.path.join(d, "metrics_d2.csv"), res["d2"])
    write_csv(os.path.join(d, "metrics_d2_curve.csv"), res["d2_curve"])
    write_csv(os.path.join(d, "metrics_forward.csv"), res["forward"])
    write_csv(os.path.join(d, "metrics_d3.csv"), res["d3"])
    write_csv(os.path.join(d, "intervals.csv"), ci_rows)
    write_csv(os.path.join(d, "split_audits.csv"), res["split_audits"])
    report["stage_seconds"] = dict(TIMERS)
    report["n_conditions"] = len({r["condition"] for r in res["d1"]})
    dump_json(report, os.path.join(d, "lane_report.json"))


def _check_json_reload(path: str, key: str) -> bool:
    try:
        with open(path) as f:
            obj = json.load(f)
        return bool(len(obj.get(key, [])) > 0)
    except Exception as e:
        print(f"[reload] FAILED {path}: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Lane: the Section 4 simulation on its own
# ---------------------------------------------------------------------------
def lane_simulation(device: str) -> dict:
    """The full Section 4 sweep plus the nondegeneracy audit.

    Split out from the SVG-World lane so it does not queue behind rendering and a
    seven-encoder panel.
    """
    import simulation as S

    lane = "simulation"
    d = out_dir(lane)
    with stage("simulation"):
        sim = S.run_sweep(device=device, out_dir=d,
                          widths=C.SIM["widths"],
                          weight_decays=C.SIM["weight_decays"],
                          n_seeds=C.SIM["n_seeds"])
    with stage("audit_summary"):
        summary = S.summarize_audit(sim, out_dir=d)
        print(json.dumps(summary, indent=1, default=_json_default), flush=True)
    report = dict(lane=lane, device=device,
                  lift_linearity_reverse_r2=sim["lift_linearity_reverse_r2"],
                  n_rows=len(sim["rows"]), audit=summary,
                  reload_ok=_check_json_reload(
                      os.path.join(d, "simulation_sweep.json"), "rows"),
                  stage_seconds=dict(TIMERS))
    dump_json(report, os.path.join(d, "lane_report.json"))
    return report


# ---------------------------------------------------------------------------
# Shared end-to-end smoke
# ---------------------------------------------------------------------------
def lane_smoke(device: str, n_scenes: int = 96, n_samples: int = 96,
               batch_candidates=(64, 128, 256)) -> dict:
    """One bounded pass through the real entrypoint, covering every dataset that
    passed preflight and every model family, plus the batch-size tuning and the
    autocast decision the production lanes inherit."""
    lane = "smoke"
    d = out_dir(lane)
    rep: dict = dict(lane=lane, device=device, n_scenes=n_scenes,
                     n_samples=n_samples, checks={})

    # --- render validation and a tiny SVG-World pass with the whole model panel
    sim_kw = dict(widths=(16, 128), weight_decays=(0.0, 1.0), n_seeds=1,
                  n_steps=200)
    sw = lane_svgworld(n_scenes, C.SVGWORLD_MODELS, device, batch_size=64,
                       dtype="fp32", lane="svgworld",
                       with_simulation=True, sim_kw=sim_kw,
                       split_seeds=(0, 1), n_png_keep=6, tag="smoke")
    rep["svgworld"] = sw

    # --- batch size and autocast, measured on the real images
    with stage("tune"):
        Z = SW.sample_latents(max(batch_candidates) * 2)
        imgs = render_style_parallel(Z, "island")
        Y = Z[:, C.POSITION_DIMS].astype(np.float64)
        rep["batch_tuning"] = {}
        for mk in ("dinov3_b16", "dinov3_l16", "resnet50"):
            rep["batch_tuning"][mk] = tune_batch_size(imgs, mk, device,
                                                      candidates=batch_candidates)
            print(json.dumps(rep["batch_tuning"][mk], default=_json_default),
                  flush=True)
        rep["autocast_validation"] = validate_autocast(
            imgs, "dinov3_b16", device, Y=Y,
            batch_size=min(128, len(imgs)))
        print(json.dumps(rep["autocast_validation"], indent=1,
                         default=_json_default), flush=True)
        rep["preprocessing_equivalence"] = {
            mk: E.preprocessing_equivalence(imgs, mk, device)
            for mk in C.SVGWORLD_MODELS}
        print(json.dumps(rep["preprocessing_equivalence"], indent=1,
                         default=_json_default), flush=True)

    # --- one example from every added dataset
    for key in ("shapes3d", "dsprites", "mpi3d_realistic"):
        try:  # each dataset writes to <dataset>_smoke, never the production dir
            rep[key] = lane_dataset(key, n_samples, C.CROSS_DATASET_MODELS,
                                    device, batch_size=64, dtype="fp32",
                                    split_seeds=(0, 1), tag="smoke")
        except Exception as e:
            import traceback
            traceback.print_exc()
            rep[key] = dict(failed=f"{type(e).__name__}: {e}")
            print(f"[smoke] dataset {key} FAILED: {type(e).__name__}: {e}",
                  flush=True)

    # --- does the resize backend matter on real 64x64 source images?
    # SVG-World renders at 224 so every model's resize is the identity there and
    # the two paths are bit-identical. The added datasets are 64x64, so the resize
    # is a real upsample and PIL's kernel differs slightly from torch's. This
    # measures the effect on the reported statistic, not on the pixels.
    with stage("preproc_backend_effect"):
        rep["preproc_backend_effect"] = []
        try:
            import datasets_extra as DX
            import factors as FA
            ds = DX.load_dataset("dsprites", n_samples=max(256, n_samples))
            spec = C.DATASETS["dsprites"]["factors"]
            Y, labels, _, _ = FA.encode_targets(ds["factors"], spec)
            tr, te = D.unit_split(len(Y), seed=0)
            for mk in C.CROSS_DATASET_MODELS:
                vals = {}
                for backend in ("torch", "pil"):
                    X = E.extract_features(ds["images"], mk, "pre", device,
                                           batch_size=64, dtype="fp32",
                                           preproc_backend=backend,
                                           log_every=0)["features"]["pooled"]
                    vals[backend] = D.d1_reverse(Y[tr], X[tr], Y[te], X[te])["d1_raw"]
                    vals[f"{backend}_X"] = X
                fd = float(np.abs(vals["torch_X"] - vals["pil_X"]).mean())
                scale = float(np.abs(vals["pil_X"]).mean()) + 1e-12
                row = dict(model=mk, d1_torch=vals["torch"], d1_pil=vals["pil"],
                           d1_abs_delta=abs(vals["torch"] - vals["pil"]),
                           mean_abs_feature_diff=fd,
                           relative_mean_abs_feature_diff=fd / scale,
                           n=int(len(Y)), source_hw=list(ds["images"].shape[1:3]))
                rep["preproc_backend_effect"].append(row)
                print(json.dumps(row, default=_json_default), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rep["preproc_backend_effect"] = dict(failed=f"{type(e).__name__}: {e}")

    # --- gates
    checks = rep["checks"]
    checks["render_checks_pass"] = bool(
        sw.get("render_checks", {}).get("all_checks_pass"))
    checks["simulation_reload_ok"] = bool(sw.get("simulation_reload_ok"))
    checks["all_features_finite"] = all(
        b["all_finite"] for cond in sw.get("feature_report", {}).values()
        for b in cond.values())
    checks["variants_distinct"] = all(
        v["distinct"] for v in sw.get("variant_distinctness", {}).values())
    checks["widths_match_declared"] = all(
        i["width_matches_declared"] for i in sw.get("encoder_info", {}).values())
    checks["preprocessing_equivalent"] = all(
        v["max_abs_diff"] < 0.15 for v in rep["preprocessing_equivalence"].values())
    eff = rep.get("preproc_backend_effect")
    checks["preproc_backend_delta_negligible"] = bool(
        isinstance(eff, list) and eff
        and all(r["d1_abs_delta"] < 5e-3 for r in eff))
    checks["datasets_ok"] = {k: ("failed" not in rep.get(k, {"failed": 1}))
                             for k in ("shapes3d", "dsprites", "mpi3d_realistic")}
    checks["n_datasets_ok"] = int(sum(checks["datasets_ok"].values()))
    rep["stage_seconds"] = dict(TIMERS)
    rep["gate_pass"] = bool(
        checks["render_checks_pass"] and checks["simulation_reload_ok"]
        and checks["all_features_finite"] and checks["variants_distinct"]
        and checks["widths_match_declared"] and checks["preprocessing_equivalent"]
        and checks["preproc_backend_delta_negligible"]
        and checks["n_datasets_ok"] >= 2)
    dump_json(rep, os.path.join(d, "smoke_report.json"))
    print("\n===== SMOKE GATES =====", flush=True)
    print(json.dumps(checks, indent=1, default=_json_default), flush=True)
    print(f"GATE_PASS {rep['gate_pass']}", flush=True)
    print(json.dumps(rep["stage_seconds"], indent=1), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True,
                    choices=["smoke", "simulation", "svgworld", "shapes3d",
                             "dsprites", "mpi3d_realistic", "pixel_control"])
    ap.add_argument("--n", type=int, default=None,
                    help="scenes (svgworld) or samples (datasets)")
    ap.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--dtype", default=C.COMPUTE_DTYPE,
                    choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--no_simulation", action="store_true")
    ap.add_argument("--keep_features", action="store_true",
                    help="also archive the float16 pooled features under "
                         "<lane>/features/ (off for the wide SVG-World panel, "
                         "whose 28 archives truncated the artifact upload)")
    ap.add_argument("--tag", default="",
                    help="suffix the lane output directory, so a rerun under a "
                         "different preprocessing backend does not overwrite the "
                         "previous one")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] lane={args.lane} tag={args.tag!r} device={device} "
          f"cpus={os.cpu_count()} torch={torch.__version__} "
          f"preproc_backend={C.PREPROC_BACKEND}", flush=True)
    if device == "cuda":
        print(f"[run] gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB",
              flush=True)

    t0 = time.time()
    if args.lane == "smoke":
        lane_smoke(device, n_scenes=args.n or 96, n_samples=args.n or 96)
    elif args.lane == "simulation":
        lane_simulation(device)
    elif args.lane == "pixel_control":
        import pixel_control as PC
        PC.run(n_scenes=args.n or C.N_SCENES_FULL,
               n_samples=args.n or C.N_DATASET_SAMPLES, side=32,
               render_style=render_style_parallel, write_csv=write_csv,
               dump_json=dump_json, out_dir=out_dir, stage=stage, timers=TIMERS)
    elif args.lane == "svgworld":
        lane_svgworld(args.n or C.N_SCENES_FULL, C.SVGWORLD_MODELS, device,
                      args.batch_size, args.dtype,
                      with_simulation=not args.no_simulation, tag=args.tag,
                      keep_features=args.keep_features)
    else:
        lane_dataset(args.lane, args.n or C.N_DATASET_SAMPLES,
                     C.CROSS_DATASET_MODELS, device, args.batch_size, args.dtype,
                     tag=args.tag, keep_features=True)
    print(f"\n[run] lane={args.lane} total {time.time() - t0:.1f}s", flush=True)
    print(json.dumps(dict(TIMERS), indent=1), flush=True)


if __name__ == "__main__":
    main()
