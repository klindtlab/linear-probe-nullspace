"""
Extract DINOv3 last-layer features for one (theme, variant) combination.

Saves three files per call:
  scene_world/{theme}/features/cls_{variant}.npy      shape (N, 768)
  scene_world/{theme}/features/patches_{variant}.npy  shape (N, n_patch, 768)
  scene_world/{theme}/features/ids_{variant}.npy      shape (N,)

`n_patch` = 14*14 = 196 for DINOv3-B/16 at 224x224, after stripping the
CLS token and any register tokens reported by the model config.

Variants:
  --pretrained  load weights from facebook/dinov3-vitb16-pretrain-lvd1689m
  --random      same architecture, freshly initialized weights, seed-locked

Designed to be invoked four times (one per (theme, variant) pair); each call
fits comfortably on a single ~16 GB GPU at batch size 128.

Usage:
  python 2_extract_features.py --theme island  --pretrained
  python 2_extract_features.py --theme island  --random
  python 2_extract_features.py --theme western --pretrained
  python 2_extract_features.py --theme western --random

Storage budget at 10k images: ~30 MB CLS + ~6 GB patches per call,
so ~24 GB total across the four calls.
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoConfig

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class PNGDataset(Dataset):
    def __init__(self, png_dir, png_files, processor):
        self.png_dir = png_dir
        self.png_files = png_files
        self.processor = processor

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        fname = self.png_files[idx]
        img = Image.open(os.path.join(self.png_dir, fname)).convert("RGB")
        px = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        return fname.replace(".png", ""), px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", choices=["island", "western"], required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--root", default="scene_world")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="random init seed for the random variant")
    args = parser.parse_args()

    if args.pretrained == args.random:
        raise ValueError("specify exactly one of --pretrained / --random")

    variant = "pre" if args.pretrained else "rand"
    png_dir = os.path.join(args.root, args.theme, "pngs")
    out_dir = os.path.join(args.root, args.theme, "features")
    os.makedirs(out_dir, exist_ok=True)

    cls_path     = os.path.join(out_dir, f"cls_{variant}.npy")
    patches_path = os.path.join(out_dir, f"patches_{variant}.npy")
    ids_path     = os.path.join(out_dir, f"ids_{variant}.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.theme}/{variant}] device={device}")
    print(f"[{args.theme}/{variant}] loading model id={MODEL_ID}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    config    = AutoConfig.from_pretrained(MODEL_ID)
    n_reg     = getattr(config, "num_register_tokens", 0)
    print(f"[{args.theme}/{variant}] num_register_tokens={n_reg}")

    if args.pretrained:
        model = AutoModel.from_pretrained(MODEL_ID)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = AutoModel.from_config(config)
    model = model.to(device).eval()

    png_files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
    print(f"[{args.theme}/{variant}] found {len(png_files)} PNGs in {png_dir}")

    dataset = PNGDataset(png_dir, png_files, processor)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_ids, all_cls, all_patches = [], [], []
    with torch.no_grad():
        for batch_ids, batch_px in tqdm(
            loader, desc=f"{args.theme}/{variant}", ncols=80
        ):
            batch_px = batch_px.to(device, non_blocking=True)
            out = model(pixel_values=batch_px)
            hidden = out.last_hidden_state          # (B, 1+n_reg+n_patch, D)
            cls     = hidden[:, 0, :]                # (B, D)
            patches = hidden[:, 1 + n_reg:, :]       # (B, n_patch, D)

            all_cls.append(cls.cpu().numpy())
            all_patches.append(patches.cpu().numpy())
            all_ids.extend(batch_ids)

    X_cls     = np.concatenate(all_cls,     axis=0).astype(np.float32)
    X_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    ids       = np.array(all_ids)

    print(f"[{args.theme}/{variant}] X_cls     shape: {X_cls.shape}")
    print(f"[{args.theme}/{variant}] X_patches shape: {X_patches.shape}")
    print(f"[{args.theme}/{variant}] ids       shape: {ids.shape}")

    np.save(cls_path,     X_cls)
    np.save(patches_path, X_patches)
    np.save(ids_path,     ids)
    print(f"[{args.theme}/{variant}] wrote {cls_path}")
    print(f"[{args.theme}/{variant}] wrote {patches_path}")
    print(f"[{args.theme}/{variant}] wrote {ids_path}")


if __name__ == "__main__":
    main()
