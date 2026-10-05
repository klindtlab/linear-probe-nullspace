"""Frozen-encoder feature extraction for every model in the panel.

One pooled representation per (model, variant, image set), taken from the final
encoder block before any task or projection head:

  cls_last_hidden   last_hidden_state[:, 0] of a ViT-style encoder
  clip_pooled       vision_model pooler_output = post_layernorm(CLS), the pooled
                    state BEFORE CLIP's multimodal projection
  pooled_spatial    ResNet pooler_output (global average pool), flattened
  patches_mean      mean over patch tokens (register tokens skipped), used only
                    to reproduce the submitted DINOv3 ViT-B/16 analysis

Preprocessing is implemented from each model's pinned `preprocessor_config.json`
rather than through `AutoImageProcessor`, because transformers 5.12.1 requires
torchvision for both its fast and slow image-processor paths and the job image
has no torchvision. The transcribed settings live in
`rebuttal_config.MODELS[key]["preproc"]`; `preprocessing_report` records exactly
what was applied so the choice is auditable.

Random-initialization controls use the identical architecture and config, built
from the same seed, so a pretrained-minus-random gap cannot come from a shape or
capacity difference.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as TF

import rebuttal_config as C

_RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _pil_filter(code: int):
    from PIL import Image

    return {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR,
            3: Image.BICUBIC}[code]


_TORCH_MODE = {0: "nearest", 2: "bilinear", 3: "bicubic"}


def preprocess_batch_torch(imgs_u8: np.ndarray, pp: dict, device: str) -> torch.Tensor:
    """Device-side preprocessing: the same policy as `preprocess_batch`, done on
    the GPU so that resizing 64x64 source images to 224 does not spend sandbox
    CPU time. `preprocessing_equivalence` reports the gap against the PIL path.
    """
    x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    out = pp["out"]
    h, w = x.shape[-2:]
    if pp["mode"] == "square":
        target = (out, out)
    else:
        scale = pp["resize_to"] / min(h, w)
        target = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    if (h, w) != target:
        mode = _TORCH_MODE[pp["resample"]]
        kw = dict(mode=mode, align_corners=False, antialias=True) if mode != "nearest" else dict(mode=mode)
        x = TF.interpolate(x, size=target, **kw).clamp_(0.0, 1.0)
    if x.shape[-2:] != (out, out):
        top = (x.shape[-2] - out) // 2
        left = (x.shape[-1] - out) // 2
        x = x[..., top:top + out, left:left + out]
    mean = torch.as_tensor(pp["mean"], device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(pp["std"], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def preprocessing_equivalence(imgs_u8: np.ndarray, model_key: str,
                              device: str) -> dict:
    """Max absolute difference between the PIL reference path and the device path.

    Resizing is the only place the two can disagree (PIL vs torch interpolation
    kernels); when the source is already 224x224 they must agree to float noise.
    """
    pp = C.MODELS[model_key]["preproc"]
    ref = preprocess_batch(imgs_u8[:8], pp).numpy()
    got = preprocess_batch_torch(imgs_u8[:8], pp, device).float().cpu().numpy()
    d = np.abs(ref - got)
    return dict(model=model_key, source_hw=list(imgs_u8.shape[1:3]),
                max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                ref_std=float(ref.std()),
                resize_is_identity=bool(imgs_u8.shape[1] == pp["out"]
                                        and pp["mode"] == "square"))


def preprocess_batch(imgs_u8: np.ndarray, pp: dict) -> torch.Tensor:
    """uint8 (B,H,W,3) -> normalized float32 (B,3,out,out).

    Resize policy per the model's recorded config, then rescale by 1/255 and
    normalize per channel. Resizing is skipped when it would be the identity.
    """
    from PIL import Image

    out = pp["out"]
    h, w = imgs_u8.shape[1:3]
    if pp["mode"] == "square":
        target = (out, out)
    else:
        short = pp["resize_to"]
        scale = short / min(h, w)
        target = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        target = (target[1], target[0])  # (H, W)

    if (h, w) != target:
        f = _pil_filter(pp["resample"])
        resized = np.stack([
            np.asarray(Image.fromarray(a).resize((target[1], target[0]), f))
            for a in imgs_u8])
    else:
        resized = imgs_u8

    H, W = resized.shape[1:3]
    if (H, W) != (out, out):  # center crop
        top, left = (H - out) // 2, (W - out) // 2
        resized = resized[:, top:top + out, left:left + out, :]

    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray(pp["mean"], np.float32)) / np.asarray(pp["std"], np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def preprocessing_report(model_key: str, imgs_u8: np.ndarray) -> dict:
    pp = C.MODELS[model_key]["preproc"]
    t = preprocess_batch(imgs_u8[:4], pp)
    return dict(
        model=model_key, mode=pp["mode"], out=pp["out"],
        resize_to=pp.get("resize_to"), resample=_RESAMPLE[pp["resample"]],
        mean=pp["mean"], std=pp["std"],
        input_shape=list(imgs_u8.shape[1:]),
        tensor_shape=list(t.shape[1:]),
        tensor_min=float(t.min()), tensor_max=float(t.max()),
        tensor_mean=float(t.mean()),
    )


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def build_encoder(model_key: str, variant: str, device: str):
    from transformers import AutoConfig, AutoModel

    spec = C.MODELS[model_key]
    cfg = AutoConfig.from_pretrained(spec["hf_id"], revision=spec["revision"])
    if variant == "pre":
        model = AutoModel.from_pretrained(spec["hf_id"], revision=spec["revision"])
    elif variant == "rand":
        torch.manual_seed(C.RANDOM_INIT_SEED)
        np.random.seed(C.RANDOM_INIT_SEED)
        model = AutoModel.from_config(cfg)
    else:
        raise ValueError(variant)

    if spec["arch"] == "clip_vision":
        model = model.vision_model
        vis_cfg = cfg.vision_config
    else:
        vis_cfg = cfg

    n_reg = int(getattr(vis_cfg, "num_register_tokens", 0) or 0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = dict(
        model=model_key, variant=variant, hf_id=spec["hf_id"],
        revision=spec["revision"], readout=spec["readout"],
        declared_width=spec["width"], n_register_tokens=n_reg,
        config_class=type(cfg).__name__, module_class=type(model).__name__,
        n_params=int(sum(p.numel() for p in model.parameters())),
    )
    return model, info


@torch.inference_mode()
def _forward_pooled(model, px, spec, n_reg, poolings):
    out = model(pixel_values=px)
    res = {}
    if spec["arch"] == "resnet":
        p = out.pooler_output
        res["pooled"] = p.flatten(start_dim=1)
        return res, None
    if spec["readout"] == "clip_pooled":
        res["pooled"] = out.pooler_output
    else:
        res["pooled"] = out.last_hidden_state[:, 0, :]
    n_patch = None
    if "patches_mean" in poolings:
        patches = out.last_hidden_state[:, 1 + n_reg:, :]
        n_patch = int(patches.shape[1])
        res["patches_mean"] = patches.mean(dim=1)
    return res, n_patch


@torch.inference_mode()
def extract_features_multi(image_sets: dict[str, np.ndarray], model_key: str,
                           variant: str, device: str, batch_size: int = 128,
                           extra_poolings: tuple[str, ...] = (),
                           preproc_backend: str | None = None,
                           dtype: str = "fp32") -> dict:
    """Pooled features for one (model, variant) over SEVERAL image sets.

    The encoder is built once and reused across every set, so a seven-model panel
    over two visual styles costs 14 model loads rather than 28. Load time is pure
    overhead on a GPU-billed sandbox.
    """
    spec = C.MODELS[model_key]
    preproc_backend = preproc_backend or C.PREPROC_BACKEND
    model, info = build_encoder(model_key, variant, device)
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16,
                      "fp16": torch.float16}[dtype]
    poolings = ("pooled",) + tuple(extra_poolings)
    out, infos = {}, {}
    for set_name, images_u8 in image_sets.items():
        n = images_u8.shape[0]
        buffers: dict[str, np.ndarray] = {}
        n_patch = None
        t0 = time.time()
        for s0 in range(0, n, batch_size):
            e = min(s0 + batch_size, n)
            if preproc_backend == "torch":
                px = preprocess_batch_torch(images_u8[s0:e], spec["preproc"], device)
            else:
                px = preprocess_batch(images_u8[s0:e], spec["preproc"]).to(device)
            if autocast_dtype is None:
                res, np_seen = _forward_pooled(model, px, spec,
                                               info["n_register_tokens"], poolings)
            else:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    res, np_seen = _forward_pooled(model, px, spec,
                                                   info["n_register_tokens"], poolings)
            n_patch = np_seen if np_seen is not None else n_patch
            for k, v in res.items():
                arr = v.float().cpu().numpy()
                if k not in buffers:
                    buffers[k] = np.empty((n, arr.shape[1]), dtype=np.float32)
                buffers[k][s0:e] = arr
        dt = time.time() - t0
        out[set_name] = buffers
        si = dict(info)
        si.update(seconds=round(dt, 2), images_per_second=round(n / max(dt, 1e-9), 1),
                  batch_size=int(batch_size), preproc_backend=preproc_backend,
                  compute_dtype=dtype, n_patch=n_patch, image_set=set_name,
                  n_images=int(n),
                  observed_width=int(buffers["pooled"].shape[1]),
                  width_matches_declared=bool(
                      buffers["pooled"].shape[1] == spec["width"]))
        infos[set_name] = si
        print(f"[extract] {model_key}/{variant}/{set_name} {n} imgs "
              f"{si['images_per_second']} img/s width={si['observed_width']}",
              flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(features=out, info=infos)


@torch.inference_mode()
def extract_features(images_u8: np.ndarray, model_key: str, variant: str,
                     device: str, batch_size: int = 128,
                     extra_poolings: tuple[str, ...] = (),
                     log_every: int = 20, preproc_backend: str | None = None,
                     dtype: str = "fp32") -> dict:
    """Pooled features for one (model, variant) over one image set."""
    spec = C.MODELS[model_key]
    preproc_backend = preproc_backend or C.PREPROC_BACKEND
    pp = spec["preproc"]
    model, info = build_encoder(model_key, variant, device)
    poolings = ("pooled",) + tuple(extra_poolings)
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16,
                      "fp16": torch.float16}[dtype]
    n = images_u8.shape[0]
    buffers: dict[str, np.ndarray] = {}
    n_patch = None
    t0 = time.time()
    for bi, s in enumerate(range(0, n, batch_size)):
        e = min(s + batch_size, n)
        if preproc_backend == "torch":
            px = preprocess_batch_torch(images_u8[s:e], pp, device)
        else:
            px = preprocess_batch(images_u8[s:e], pp).to(device, non_blocking=True)
        if autocast_dtype is None:
            res, np_seen = _forward_pooled(model, px, spec,
                                           info["n_register_tokens"], poolings)
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                res, np_seen = _forward_pooled(model, px, spec,
                                               info["n_register_tokens"], poolings)
        n_patch = np_seen if np_seen is not None else n_patch
        for k, v in res.items():
            arr = v.float().cpu().numpy()
            if k not in buffers:
                buffers[k] = np.empty((n, arr.shape[1]), dtype=np.float32)
            buffers[k][s:e] = arr
        if log_every and (bi + 1) % log_every == 0:
            done = e
            print(f"[extract {model_key}/{variant}] {done}/{n} "
                  f"({done / max(time.time() - t0, 1e-9):.0f} img/s)", flush=True)
    dt = time.time() - t0
    info.update(seconds=round(dt, 2), images_per_second=round(n / max(dt, 1e-9), 1),
                batch_size=int(batch_size), preproc_backend=preproc_backend,
                compute_dtype=dtype,
                n_patch=n_patch,
                observed_width=int(buffers["pooled"].shape[1]),
                width_matches_declared=bool(
                    buffers["pooled"].shape[1] == spec["width"]))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(features=buffers, info=info)


def feature_report(feats: dict[str, np.ndarray]) -> dict:
    rep = {}
    for name, X in feats.items():
        rep[name] = dict(
            shape=list(X.shape), all_finite=bool(np.isfinite(X).all()),
            mean=float(X.mean()), std=float(X.std()),
            n_constant_dims=int((X.std(axis=0) < 1e-8).sum()),
            n_duplicate_rows=int(X.shape[0] - len(np.unique(
                np.round(X[:, :16].astype(np.float64), 4), axis=0))),
            l2_norm_mean=float(np.linalg.norm(X, axis=1).mean()),
            effective_rank_99=_effective_rank(X, 0.99),
        )
    return rep


def _effective_rank(X: np.ndarray, frac: float) -> int:
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] > 4000:
        Xc = Xc[:4000]
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, frac) + 1)


def variant_distinctness(f_pre: np.ndarray, f_rand: np.ndarray) -> dict:
    """Cheap guard that the pretrained and random conditions are not the same
    computation: their feature geometries must differ."""
    def corr_spectrum(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Xc[:2000], compute_uv=False)
        return (s ** 2) / (s ** 2).sum()

    a, b = corr_spectrum(f_pre), corr_spectrum(f_rand)
    k = min(len(a), len(b), 32)
    return dict(
        pc1_share_pre=float(a[0]), pc1_share_rand=float(b[0]),
        top32_share_pre=float(a[:32].sum()), top32_share_rand=float(b[:32].sum()),
        spectrum_l1_distance_top32=float(np.abs(a[:k] - b[:k]).sum()),
        mean_abs_feature_diff=float(np.abs(
            f_pre[:512].astype(np.float64) - f_rand[:512].astype(np.float64)).mean()),
        distinct=bool(np.abs(a[:k] - b[:k]).sum() > 1e-6),
    )
