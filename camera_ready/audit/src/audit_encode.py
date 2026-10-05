"""Feature extraction for the audit: several model instances per image pass.

The prior run extracted one pretrained encoder and one random initialization at
a fixed seed. The audit needs five random initializations (plus, for ResNet-50, a
BatchNorm-calibrated variant), which would be six or seven passes over the same
images if each instance were handled separately. Preprocessing is identical
across instances of one model, so this module preprocesses each batch once and
forwards it through every resident instance.

Preprocessing and pooled-readout logic are reused from the prior run's
`encoders.py` unchanged, so the features are the same objects the earlier numbers
were computed on; only the instance loop and the BatchNorm calibration are new.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import rebuttal_config as C
from encoders import _forward_pooled, preprocess_batch


def build_instance(model_key: str, variant: str, seed: int, device: str):
    """`variant` is "pre" or "rand"; `seed` sets the random initialization."""
    from transformers import AutoConfig, AutoModel

    spec = C.MODELS[model_key]
    cfg = AutoConfig.from_pretrained(spec["hf_id"], revision=spec["revision"])
    if variant == "pre":
        model = AutoModel.from_pretrained(spec["hf_id"], revision=spec["revision"])
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AutoModel.from_config(cfg)
    vis_cfg = cfg
    if spec["arch"] == "clip_vision":
        model = model.vision_model
        vis_cfg = cfg.vision_config
    n_reg = int(getattr(vis_cfg, "num_register_tokens", 0) or 0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = dict(model=model_key, variant=variant, init_seed=int(seed),
                hf_id=spec["hf_id"], revision=spec["revision"],
                readout=spec["readout"], n_register_tokens=n_reg,
                n_params=int(sum(p.numel() for p in model.parameters())))
    return model, n_reg, info


@torch.inference_mode()
def calibrate_batchnorm(model, images_u8: np.ndarray, model_key: str,
                        device: str, batch_size: int = 256,
                        max_images: int = 4096) -> dict:
    """Update BatchNorm running statistics on real images, without gradients.

    A freshly initialized ResNet carries ImageNet-irrelevant BatchNorm buffers
    (running_mean 0, running_var 1), which is not what "an untrained network of
    the same architecture" means for a normalization-dependent convnet: the
    features it emits are dominated by an arbitrary affine mismatch. This pass
    puts the normalization statistics on the evaluation images while leaving
    every weight untouched.
    """
    spec = C.MODELS[model_key]
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return dict(n_batchnorm_layers=0, n_images_used=0)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative moving average
        m.train()
    n = min(len(images_u8), max_images)
    for s0 in range(0, n, batch_size):
        px = preprocess_batch(images_u8[s0:min(s0 + batch_size, n)],
                              spec["preproc"]).to(device)
        model(pixel_values=px)
    for m in bns:
        m.eval()
    model.eval()
    return dict(n_batchnorm_layers=len(bns), n_images_used=int(n),
                mean_running_var=float(np.mean(
                    [float(m.running_var.mean()) for m in bns])))


def bn_modules(model):
    return [m for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def snapshot_bn(model) -> list[dict]:
    """Clone every BatchNorm buffer so a train-mode pass cannot leak into a
    later eval-mode pass."""
    snap = []
    for m in bn_modules(model):
        snap.append(dict(
            running_mean=None if m.running_mean is None else m.running_mean.detach().clone(),
            running_var=None if m.running_var is None else m.running_var.detach().clone(),
            num_batches_tracked=None if m.num_batches_tracked is None
            else m.num_batches_tracked.detach().clone(),
            momentum=m.momentum, training=m.training,
            track_running_stats=m.track_running_stats))
    return snap


def restore_bn(model, snap: list[dict]) -> float:
    """Restore buffers and report the largest absolute residual difference, which
    must be 0 for the restore to be exact."""
    worst = 0.0
    for m, s in zip(bn_modules(model), snap):
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            saved = s[name]
            cur = getattr(m, name)
            if saved is None or cur is None:
                continue
            cur.copy_(saved)
            worst = max(worst, float((cur.float() - saved.float()).abs().max()))
        m.momentum = s["momentum"]
        m.track_running_stats = s["track_running_stats"]
        m.train(s["training"])
    return worst


@torch.inference_mode()
def extract_instances(images_u8: np.ndarray, model_key: str,
                      instances: list[dict], device: str,
                      batch_size: int = 256,
                      extra_poolings: tuple[str, ...] = (),
                      calib_idx: np.ndarray | None = None) -> dict:
    """One preprocessing pass over the images, forwarded through each instance.

    `instances` is a list of dicts with keys `name`, `variant`, `seed`, and an
    optional `bn_calibrate` flag. Returns {name: {pooling: array}} plus info.
    """
    spec = C.MODELS[model_key]
    poolings = ("pooled",) + tuple(extra_poolings)
    built = []
    for inst in instances:
        model, n_reg, info = build_instance(model_key, inst["variant"],
                                           int(inst.get("seed", 42)), device)
        if inst.get("bn_calibrate"):
            # Calibration images are TRAINING rows only. Using every row would
            # put held-out images into the control's normalization statistics,
            # which is exactly the leakage the split exists to prevent.
            imgs = images_u8 if calib_idx is None else images_u8[calib_idx]
            info["bn_calibration"] = calibrate_batchnorm(
                model, imgs, model_key, device, batch_size)
            info["bn_calibration"]["train_rows_only"] = calib_idx is not None
        built.append((inst["name"], model, n_reg, info))

    n = images_u8.shape[0]
    buffers: dict[str, dict[str, np.ndarray]] = {name: {} for name, *_ in built}
    t0 = time.time()
    for s0 in range(0, n, batch_size):
        e = min(s0 + batch_size, n)
        px = preprocess_batch(images_u8[s0:e], spec["preproc"]).to(device)
        for name, model, n_reg, _ in built:
            res, _ = _forward_pooled(model, px, spec, n_reg, poolings)
            for k, v in res.items():
                arr = v.float().cpu().numpy()
                if k not in buffers[name]:
                    buffers[name][k] = np.empty((n, arr.shape[1]), np.float32)
                buffers[name][k][s0:e] = arr
    dt = time.time() - t0
    infos = {}
    for name, model, _, info in built:
        info = dict(info)
        info.update(n_images=int(n), seconds_shared_pass=round(dt, 2),
                    observed_width=int(buffers[name]["pooled"].shape[1]))
        infos[name] = info
    print(f"[extract] {model_key}: {len(built)} instances x {n} images in "
          f"{dt:.1f}s ({n * len(built) / max(dt, 1e-9):.0f} img-instances/s)",
          flush=True)
    for _, model, _, _ in built:
        del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(features=buffers, info=infos)


def feature_stats(X: np.ndarray) -> dict:
    """Spectrum descriptors used as covariates when reading a D1 comparison."""
    X = np.asarray(X, np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc.T @ Xc
    ev = np.maximum(np.linalg.eigvalsh(0.5 * (G + G.T))[::-1], 0.0)
    share = ev / max(ev.sum(), 1e-300)
    cum = np.cumsum(share)
    return dict(width=int(X.shape[1]),
                pc1_share=float(share[0]),
                effective_rank_99=int(np.searchsorted(cum, 0.99) + 1),
                top12_share=float(share[:12].sum()),
                participation_ratio=float(1.0 / (share ** 2).sum()),
                total_variance=float(ev.sum() / max(len(X) - 1, 1)),
                feature_norm_mean=float(np.linalg.norm(X, axis=1).mean()))
