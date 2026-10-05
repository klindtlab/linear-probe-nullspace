"""Locked configuration for the rebuttal run.

Every analysis choice is fixed here once, in one place, so that the simulation,
the SVG-World diagnostics and the figures all read the same definition.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Latent layout of SVG-World (from the submitted david/README.md)
# --------------------------------------------------------------------------
# 32 stored dims; 20 consumed by the renderer; 12 of those are positions.
POSITION_DIMS = [0, 1, 7, 8, 14, 15, 21, 22, 28, 29, 30, 31]
FACING_DIMS = [2, 9, 16, 23]
SKINHUE_DIMS = [3, 10, 17, 24]
ACTIVE_DIMS = sorted(POSITION_DIMS + FACING_DIMS + SKINHUE_DIMS)
ALL_DIMS = list(range(32))

# Headline target: the 12 position latents. The other two are appendix
# sensitivities (pin down which k the numbers refer to).
LATENT_SUBSETS = {
    "position12": POSITION_DIMS,
    "active20": ACTIVE_DIMS,
    "all32": ALL_DIMS,
}
HEADLINE_SUBSET = "position12"


def latent_labels() -> list[str]:
    labels = []
    per_person = ["x", "z", "facing", "skin_h",
                  "shirt_h_un", "shirt_s_un", "shirt_v_un"]
    for p in range(4):
        for d in per_person:
            labels.append(f"p{p}_{d}")
    labels += ["animal0_x", "animal0_z", "animal1_x", "animal1_z"]
    return labels


LATENT_LABELS = latent_labels()

# --------------------------------------------------------------------------
# Encoders. Revisions pinned so the run is reproducible.
# --------------------------------------------------------------------------
# Every entry is pinned to a commit. `readout` names the pooled representation
# taken from the final encoder block, before any task or projection head:
#   cls_last_hidden  last_hidden_state[:, 0] of a ViT-style encoder
#   clip_pooled      vision_model pooler_output = post_layernorm(CLS), i.e. the
#                    pooled state BEFORE the multimodal projection
#   pooled_spatial   ResNet pooler_output (global average pool), flattened
#
# `preproc` is transcribed from each model's pinned preprocessor_config.json:
#   square:   resize to (out, out)
#   shortest: resize shortest edge to `resize_to`, then center-crop `out`
# resample codes are PIL codes as recorded by HuggingFace (2 = bilinear,
# 3 = bicubic).
MODELS = {
    "dinov3_s16": {
        "hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "revision": "114c1379950215c8b35dfcd4e90a5c251dde0d32",
        "paradigm": "self-supervised (DINOv3, LVD-1689M)",
        "label": "DINOv3 ViT-S/16", "family": "dinov3", "width": 384,
        "arch": "vit", "readout": "cls_last_hidden",
        "preproc": {"mode": "square", "out": 224, "resample": 2,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
    "dinov3_b16": {
        "hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
        "paradigm": "self-supervised (DINOv3, LVD-1689M)",
        "label": "DINOv3 ViT-B/16", "family": "dinov3", "width": 768,
        "arch": "vit", "readout": "cls_last_hidden",
        "preproc": {"mode": "square", "out": 224, "resample": 2,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
    "dinov3_l16": {
        "hf_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
        "paradigm": "self-supervised (DINOv3, LVD-1689M)",
        "label": "DINOv3 ViT-L/16", "family": "dinov3", "width": 1024,
        "arch": "vit", "readout": "cls_last_hidden",
        "preproc": {"mode": "square", "out": 224, "resample": 2,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
    "dinov2_b14": {
        "hf_id": "facebook/dinov2-base",
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "paradigm": "self-supervised (DINOv2, LVD-142M)",
        "label": "DINOv2 ViT-B/14", "family": "dinov2", "width": 768,
        "arch": "vit", "readout": "cls_last_hidden",
        "preproc": {"mode": "shortest", "resize_to": 256, "out": 224, "resample": 3,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
    "deit_b16": {
        "hf_id": "facebook/deit-base-patch16-224",
        "revision": "fb2c78a54a5637dec350432794f7b93e31f910c9",
        "paradigm": "supervised (ImageNet-1k classification)",
        "label": "DeiT-B/16", "family": "deit", "width": 768,
        "arch": "vit", "readout": "cls_last_hidden",
        "preproc": {"mode": "square", "out": 224, "resample": 3,
                    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    },
    "clip_b16": {
        "hf_id": "openai/clip-vit-base-patch16",
        "revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "paradigm": "multimodal contrastive (CLIP, WIT-400M)",
        "label": "CLIP ViT-B/16", "family": "clip", "width": 768,
        "arch": "clip_vision", "readout": "clip_pooled",
        "preproc": {"mode": "shortest", "resize_to": 224, "out": 224, "resample": 3,
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "std": [0.26862954, 0.26130258, 0.27577711]},
    },
    "resnet50": {
        "hf_id": "microsoft/resnet-50",
        "revision": "34c2154c194f829b11125337b98c8f5f9965ff19",
        "paradigm": "supervised convnet (ImageNet-1k classification)",
        "label": "ResNet-50", "family": "resnet", "width": 2048,
        "arch": "resnet", "readout": "pooled_spatial",
        # ConvNextFeatureExtractor: size 224 with crop_pct 0.875 -> resize the
        # shortest edge to 256, then center-crop 224.
        "preproc": {"mode": "shortest", "resize_to": 256, "out": 224, "resample": 3,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
}

# Which encoders run on which lane.
SVGWORLD_MODELS = ("dinov3_s16", "dinov3_b16", "dinov3_l16", "dinov2_b14",
                   "deit_b16", "clip_b16", "resnet50")
CROSS_DATASET_MODELS = ("dinov3_b16", "clip_b16", "resnet50")

# DINOv3 B/16 additionally reproduces the submitted mean-patch pooling on
# SVG-World; every other condition uses the single pooled readout.
EXTRA_POOLING = {"dinov3_b16": ("patches_mean",)}

VARIANTS = ("pre", "rand")
STYLES = ("island", "western")
STYLE_LABELS = {"island": "AquaWorld", "western": "WestWorld"}

# --------------------------------------------------------------------------
# Added disentanglement datasets
# --------------------------------------------------------------------------
# Each entry describes the official source, the factor grid in the source's
# lexicographic index order, and how each factor is treated as a probe target:
#   linear      continuous / ordinal, used as-is (min-max mapped to [0, 1])
#   cyclic      encoded as a (sin, cos) pair over the factor's period
#   categorical one-hot, appendix sensitivity only
# `n_values` are in index-major order, so the flat source index is the
# lexicographic mixed-radix encoding of the factor tuple.
N_DATASET_SAMPLES = 20_000

DATASETS = {
    "svgworld": {
        "label": "SVG-World",
        "source": "rendered in-job from the submitted svg_island.py / svg_western.py",
        "license": "authors' own benchmark (submitted with the paper)",
        "d3": "cross_style",
    },
    "shapes3d": {
        "label": "3D Shapes",
        "url": "https://storage.googleapis.com/3d-shapes/3dshapes.h5",
        "source": "official DeepMind bucket gs://3d-shapes",
        "license": "Apache-2.0 (google-deepmind/3d-shapes)",
        "sha_size_bytes": 267573662,
        "image_shape": [64, 64, 3],
        "factors": [
            {"name": "floor_hue", "n_values": 10, "kind": "cyclic", "period": 10},
            {"name": "wall_hue", "n_values": 10, "kind": "cyclic", "period": 10},
            {"name": "object_hue", "n_values": 10, "kind": "cyclic", "period": 10},
            {"name": "scale", "n_values": 8, "kind": "linear"},
            {"name": "shape", "n_values": 4, "kind": "categorical"},
            {"name": "orientation", "n_values": 15, "kind": "linear"},
        ],
        "d3": "checkerboard",
    },
    "dsprites": {
        "label": "dSprites",
        "url": ("https://github.com/google-deepmind/dsprites-dataset/raw/master/"
                "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
        "source": "official google-deepmind/dsprites-dataset repository",
        "license": "Apache-2.0 (google-deepmind/dsprites-dataset)",
        "image_shape": [64, 64, 1],
        "factors": [
            {"name": "shape", "n_values": 3, "kind": "categorical"},
            {"name": "scale", "n_values": 6, "kind": "linear"},
            # A genuine rotation, and the bin grid is CLOSED, not open. Read from
            # the official npz's latents_values: the 40 stored angles are
            # linspace(0, 2*pi, 40) inclusive, spacing 2*pi/39 = 0.161107, so class
            # 39 is exactly 2*pi and renders the same image as class 0. The cyclic
            # period in bin units is therefore 39, not 40, and bin 39 is excluded
            # from sampling so the duplicated angle cannot enter twice. An earlier
            # run used period 40, which placed bin 39 at 351 degrees instead of 360
            # and gave ~2.5% of rows a target 9 degrees off.
            {"name": "orientation", "n_values": 40, "kind": "cyclic", "period": 39,
             "exclude_values": [39]},
            {"name": "pos_x", "n_values": 32, "kind": "linear"},
            {"name": "pos_y", "n_values": 32, "kind": "linear"},
        ],
        "d3": "checkerboard",
    },
    "mpi3d_realistic": {
        "label": "MPI3D-realistic",
        "url": ("https://storage.googleapis.com/mpi3d_disentanglement_dataset/"
                "data/mpi3d_realistic.npz"),
        "source": "official mpi3d_disentanglement_dataset bucket (Gondal et al. 2019)",
        "license": "Creative Commons Attribution 4.0 (rr-learning/disentanglement_dataset)",
        # single uncompressed zip member 'images.npy', so selected records are
        # fetched by HTTP range instead of downloading all 12.74 GB
        "zip_member": "images.npy",
        "zip_total_bytes": 12740198762,
        "member_uncompressed_bytes": 12740198528,
        "image_shape": [64, 64, 3],
        "factors": [
            {"name": "object_color", "n_values": 6, "kind": "categorical"},
            {"name": "object_shape", "n_values": 6, "kind": "categorical"},
            {"name": "object_size", "n_values": 2, "kind": "linear"},
            {"name": "camera_height", "n_values": 3, "kind": "linear"},
            {"name": "background_color", "n_values": 3, "kind": "categorical"},
            # Both axes are ordinal robot-arm degrees of freedom, NOT angles. An
            # earlier run encoded horizontal_axis as sin/cos, which asserts a
            # wrap-around between bin 39 and bin 0 that this factor does not have.
            {"name": "horizontal_axis", "n_values": 40, "kind": "linear"},
            {"name": "vertical_axis", "n_values": 40, "kind": "linear"},
        ],
        "d3": "checkerboard",
        "optional": True,
    },
}
# Batch sizes chosen from warmed throughput measured in the smoke on one H100
# (peak memory never exceeded 4.0 GB, so 256 is safe for every model here).
BATCH_SIZE = 256
# bf16 autocast was measured against fp32 on DINOv3 B/16 and REJECTED: relative
# mean absolute feature difference 9.066e-3 against a 5e-3 acceptance threshold
# (the D1 statistic itself moved only 5.239e-4). Production runs in fp32.
COMPUTE_DTYPE = "fp32"
# Resize backend. Measured in the smoke on real 64x64 dSprites images: the
# device-side interpolation moves D1 by up to 0.1059 (ResNet-50) against the PIL
# reference, which is the same order as the effects being reported. So the PIL
# path is the production path. Where the resize is an identity (SVG-World renders
# at 224 with a square-224 policy) the two are bit-identical and PIL costs nothing.
PREPROC_BACKEND = "pil"

SAMPLE_SEED = 1234          # deterministic factor-balanced sampling
CHECKERBOARD_SEED = 99      # parity offset for the held-out combination grid

# --------------------------------------------------------------------------
# Evaluation protocol
# --------------------------------------------------------------------------
N_SCENES_FULL = 10_000        # per style; same latent matrix in both styles
IMAGE_SIZE = 224
DATA_SEED = 42                # latent-matrix seed (matches the submission)
RANDOM_INIT_SEED = 42         # matched random-initialization seed

SPLIT_SEEDS = (0, 1, 2, 3, 4)  # five repeated 50/50 scene-level splits
TEST_FRACTION = 0.5
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
RIDGE_CV_FOLDS = 5            # inside the training half only
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 7
CI_LEVEL = 0.95
D2_KS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 767)

# --------------------------------------------------------------------------
# Section 4 simulation (recovered verbatim from 0_simulation.ipynb)
# --------------------------------------------------------------------------
SIM = dict(
    n_latent=2,
    n_ambient=256,
    n_train=8000,
    n_test=2000,
    lift_seed=7,
    lift_scale=3.0,
    z_train_seed=1,
    z_test_seed=2,
    widths=(4, 8, 16, 32, 64, 128, 256, 512, 1024),
    weight_decays=(0.0, 1.0),
    n_seeds=5,
    n_steps=2000,
    lr=2e-3,
    batch_size=256,
    ridge_alpha=1e-3,
    ood_norm_threshold=0.8,
    audit_width=1024,
    audit_seed=42,
    n_eigenvalues_reported=20,
)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def artifacts_dir() -> str:
    d = os.environ.get("ARTIFACTS_DIR") or "/tmp/rebuttal_artifacts"
    os.makedirs(d, exist_ok=True)
    return d


def results_dir() -> str:
    base = os.environ.get("EXPERIMENT_DIR", ".")
    d = os.path.join(base, "results")
    os.makedirs(d, exist_ok=True)
    return d


def sub(*parts: str) -> str:
    p = os.path.join(artifacts_dir(), *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
