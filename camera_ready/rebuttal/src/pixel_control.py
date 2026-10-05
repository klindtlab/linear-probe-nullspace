"""Pixel-space control: do the target factors dominate IMAGE variance?

The cross-dataset inversion invites an explanation: in the added benchmarks the
generative factors are the dominant image variance, so even a random projection
retains them, whereas SVG-World's small objects on a fixed background are not. That
is a claim about pixels, and it is measurable directly rather than asserted. This
module runs the same D1 on raw downsampled pixels instead of encoder features.

Reading it: a high pixel-space D1 means a linear function of the factors explains
most of the image variance in that dataset; a low one means it does not. If the
added datasets come out high and SVG-World low, the account is supported by
measurement. If not, the account is wrong and the inversion needs another
explanation.

Pixels are area-averaged to `side` x `side` x 3 before the fit. Averaging is a
linear map, so it cannot manufacture linear structure that was not present. The
split, the target block and the estimator match the encoder lanes exactly.
"""

from __future__ import annotations

import os

import numpy as np

import diagnostics as D
import rebuttal_config as C


def downsample(imgs: np.ndarray, side: int) -> np.ndarray:
    """Area-average to (side, side, 3), flattened. Linear in the pixels."""
    n, h, w, c = imgs.shape
    fh, fw = max(h // side, 1), max(w // side, 1)
    keep_h, keep_w = fh * side, fw * side
    x = imgs[:, :keep_h, :keep_w, :].astype(np.float32)
    x = x.reshape(n, side, fh, side, fw, c).mean(axis=(2, 4))
    return x.reshape(n, -1)


def measure(tag: str, X: np.ndarray, Y: np.ndarray, labels: list[str],
            unit_hash: np.ndarray) -> dict:
    tr, te = D.unit_split(len(Y), seed=C.SPLIT_SEEDS[0])
    d1 = D.d1_reverse(Y[tr], X[tr], Y[te], X[te])
    fp = D.forward_probe(X[tr], Y[tr], X[te], Y[te], labels=labels)
    row = dict(dataset=tag, n=int(len(Y)), pixel_dim=int(X.shape[1]),
               n_targets=int(Y.shape[1]), target_rank=d1["target_rank"],
               pixel_d1_raw=d1["d1_raw"], pixel_d1_ceiling=d1["ceiling"],
               pixel_d1_normalized=d1["d1_normalized"],
               pixel_pc1_share=d1["pc1_share"],
               pixel_top12_share=d1["top12_share"],
               pixel_forward_r2=fp["r2_mean"],
               **D.audit_split(tr, te, unit_hash))
    print(f"[pixel {tag}] D1={row['pixel_d1_raw']:+.4f} "
          f"ceiling={row['pixel_d1_ceiling']:.4f} "
          f"norm={row['pixel_d1_normalized']:+.4f} "
          f"forward={row['pixel_forward_r2']:+.4f} "
          f"(pixels={row['pixel_dim']}, targets={row['n_targets']})", flush=True)
    return row


def run(n_scenes: int, n_samples: int, side: int, render_style,
        write_csv, dump_json, out_dir, stage, timers) -> dict:
    """Driven by run_lane so the rendering and IO helpers stay in one place."""
    import factors as FA
    import svgworld_data as SW

    lane = "pixel_control"
    d = out_dir(lane)
    rows: list[dict] = []
    report: dict = dict(lane=lane, side=side, n_scenes=int(n_scenes),
                        n_samples=int(n_samples))

    with stage("pixel_svgworld"):
        Z = SW.sample_latents(n_scenes)
        Y = Z[:, C.POSITION_DIMS].astype(np.float64)
        labels = [C.LATENT_LABELS[i] for i in C.POSITION_DIMS]
        uh = np.asarray([hash(Z[i].tobytes()) for i in range(len(Z))])
        for style in C.STYLES:
            imgs = render_style(Z, style)
            rows.append(measure(f"svgworld_{style}", downsample(imgs, side),
                                Y, labels, uh))
            del imgs

    import datasets_extra as DX
    for key in ("shapes3d", "dsprites", "mpi3d_realistic"):
        with stage(f"pixel_{key}"):
            try:
                ds = DX.load_dataset(key, n_samples=n_samples)
                spec = C.DATASETS[key]["factors"]
                Yd, labels_d, _, _ = FA.encode_targets(ds["factors"], spec)
                uh_d = FA.combination_hash(ds["factors"])
                rows.append(measure(key, downsample(ds["images"], side),
                                    Yd, labels_d, uh_d))
                del ds
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[pixel {key}] FAILED: {type(e).__name__}: {e}", flush=True)

    write_csv(os.path.join(d, "pixel_control.csv"), rows)
    report["rows"] = rows
    report["stage_seconds"] = dict(timers)
    dump_json(report, os.path.join(d, "lane_report.json"))
    return report
