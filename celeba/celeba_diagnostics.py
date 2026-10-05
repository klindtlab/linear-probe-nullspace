"""CelebA diagnostics for Sec. 9 of "Is a Linear Probe Evidence of a Linear
Representation?" (NeurIPS 2026).

Encoders: DINOv3-B/16 (facebook/dinov3-vitb16-pretrain-lvd1689m) and CLIP-B/16
(open_clip "ViT-B-16-quickgelu", OpenAI weights), each against five random
initializations of the same architecture. Factor set: the 40 binary CelebA
attributes, coded in {0,1}, on 20,000 images of the official training split.

Per encoder, weights and feature mode (CLS / mean of patch tokens) it computes,
on a 50/50 split (random_state=42) with ridge probes (alpha=1, intercept):
  - forward R^2 (features -> attributes), averaged over attributes
  - reverse R^2 (attributes -> features) on raw, standardized and PCA-whitened
    features; the whitened value at k = r is the spectral concentration SC
  - the ceiling C_r of Proposition 3 (top-r eigenvalue share, r = rank of the
    attribute matrix), R^2_rev / C_r and the participation ratio
  - a 95% bootstrap interval for raw reverse R^2 (1,000 resamples of
    held-out images)

The OpenAI CLIP weights were trained with QuickGELU, so pretrained model,
random initializations and preprocessing all use "ViT-B-16-quickgelu"; the
activation is asserted. --selftest checks on synthetic data that affine
recodings of the factors (0/1 -> +-1, z-scoring) leave the reverse statistics
unchanged up to the ridge penalty.

Features are cached in features_celeba/; delete the folder to re-extract.

Usage:
  python celeba_diagnostics.py --selftest
  python celeba_diagnostics.py --celeba-root /tmp/celeba_root --out results_celeba.csv
  (the root must contain celeba/img_align_celeba, list_attr_celeba.txt and
   list_eval_partition.txt; prepare_celeba.py builds this from the Kaggle zip)
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
# torchvision.datasets no longer needed: RawCelebA reads files directly
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 20_000
N_RANDOM_SEEDS = 5
ALPHA = 1.0                 # matches both probe scripts
SPLIT_SEED = 42             # matches both probe scripts
SUBSET_SEED = 0             # which 20k CelebA train images
BOOTSTRAP = 1000
DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# ------------------------------------------------------------ estimators
# reverse_r2: same estimator as the SVG-World experiments

def reverse_r2(Z_train, X_train, Z_test, X_test, alpha=ALPHA):
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(Z_train, X_train)
    X_pred = probe.predict(Z_test)
    total = X_test.var(axis=0).sum() + 1e-12
    resid = (X_test - X_pred).var(axis=0).sum()
    return 1.0 - resid / total


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def pca_whiten(X_train, X_test, n_components=None):
    if n_components is None:
        n_components = min(X_train.shape) - 1
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    return pca.fit_transform(X_train), pca.transform(X_test), pca


# forward probe: same estimator as the SVG-World experiments

def r2_per_dim(y_true, y_pred):
    tot = y_true.var(axis=0) + 1e-8
    res = (y_true - y_pred).var(axis=0)
    return 1.0 - res / tot


def forward_probe(X_train, X_test, y_train, y_test, alpha=ALPHA):
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_train, y_train)
    return float(r2_per_dim(y_test, probe.predict(X_test)).mean())


# Proposition 3 ceiling and participation ratio

def ceiling_top_r(X_train, r):
    lam = np.linalg.eigvalsh(np.cov(X_train.T))[::-1]
    lam = np.clip(lam, 0, None)
    return float(lam[:r].sum() / (lam.sum() + 1e-12))


def participation_ratio(X_train):
    lam = np.clip(np.linalg.eigvalsh(np.cov(X_train.T)), 0, None)
    return float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-12))


def rev_bootstrap_ci(Z_tr, X_tr, Z_te, X_te, alpha=ALPHA, n_boot=BOOTSTRAP,
                     seed=SPLIT_SEED):
    """Paired sample bootstrap over held-out rows for the raw reverse R^2."""
    probe = Ridge(alpha=alpha, fit_intercept=True).fit(Z_tr, X_tr)
    pred = probe.predict(Z_te)
    rng = np.random.default_rng(seed)
    n = len(X_te)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        Xb, Pb = X_te[b], pred[b]
        total = Xb.var(axis=0).sum() + 1e-12
        resid = (Xb - Pb).var(axis=0).sum()
        vals.append(1.0 - resid / total)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------ encoders

def build_dinov3(random_init: bool, seed: int):
    from transformers import AutoImageProcessor, AutoModel, AutoConfig
    processor = AutoImageProcessor.from_pretrained(DINOV3_ID)
    config = AutoConfig.from_pretrained(DINOV3_ID)
    n_reg = getattr(config, "num_register_tokens", 0)
    if random_init:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(DINOV3_ID)
    model = model.to(DEVICE).eval()

    def encode(imgs):  # list of PIL images -> (cls, patches_mean)
        px = processor(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            hidden = model(pixel_values=px).last_hidden_state
        cls = hidden[:, 0, :]
        pmean = hidden[:, 1 + n_reg:, :].mean(dim=1)
        return cls.cpu().numpy(), pmean.cpu().numpy()

    return encode


CLIP_ARCH = "ViT-B-16-quickgelu"   # OpenAI weights were trained with QuickGELU


def build_clip(random_init: bool, seed: int):
    import open_clip
    if random_init:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = open_clip.create_model(CLIP_ARCH, pretrained=None)
    else:
        model = open_clip.create_model(CLIP_ARCH, pretrained="openai")
    _, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained="openai")
    act = type(model.visual.transformer.resblocks[0].mlp.gelu).__name__
    assert act == "QuickGELU", f"expected QuickGELU activation, got {act}"
    model = model.to(DEVICE).eval()

    def encode(imgs):
        px = torch.stack([preprocess(im) for im in imgs]).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(px).float()
        e = emb.cpu().numpy()
        return e, e  # no patch tokens exposed; report image embedding only

    return encode


ENCODERS = {"dinov3_b16": build_dinov3, "clip_b16": build_clip}
# cache tag per encoder; bump when the feature extractor changes
CACHE_TAG = {"dinov3_b16": "dinov3_b16", "clip_b16": "clip_b16_qg"}


# ------------------------------------------------------------ data

class RawCelebA(torch.utils.data.Dataset):
    """Direct CelebA loader (train split), bypassing torchvision's MD5
    integrity check, which rejects Kaggle-sourced files. Yields
    (PIL image, attr in {0,1}^40); transforms live in the encoder.

    Expects under {root}/celeba/:
      img_align_celeba/*.jpg
      list_attr_celeba.txt      (original format: count line, header, rows)
      list_eval_partition.txt   (filename partition; 0=train)
    """

    def __init__(self, root):
        from PIL import Image  # noqa: F401 (ensures PIL present at init)
        base = os.path.join(root, "celeba")
        self.img_dir = os.path.join(base, "img_align_celeba")

        with open(os.path.join(base, "list_attr_celeba.txt")) as f:
            lines = f.read().strip().split("\n")
        self.attr_names = lines[1].split()
        attr_map = {}
        for ln in lines[2:]:
            parts = ln.split()
            attr_map[parts[0]] = np.array([int(v) for v in parts[1:]],
                                          dtype=np.int64)

        with open(os.path.join(base, "list_eval_partition.txt")) as f:
            part_lines = [ln.split() for ln in f.read().strip().split("\n")]
        self.files = [fn for fn, p in part_lines
                      if int(p) == 0 and fn in attr_map]
        self.attrs = np.stack([attr_map[fn] for fn in self.files])
        # original encoding is {-1,+1}; map to {0,1}
        self.attrs = ((self.attrs + 1) // 2 if self.attrs.min() < 0
                      else self.attrs)
        assert self.attrs.shape[1] == 40, self.attrs.shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(os.path.join(self.img_dir, self.files[i])).convert("RGB")
        return img, torch.from_numpy(self.attrs[i])


def collate(batch):
    imgs = [b[0] for b in batch]
    attrs = torch.stack([b[1] for b in batch])
    return imgs, attrs


def extract(encode, root, n=None, batch=256, workers=8):
    ds = RawCelebA(root)
    n = min(n or N_SAMPLES, len(ds))
    rng = np.random.default_rng(SUBSET_SEED)
    idx = rng.choice(len(ds), size=n, replace=False)
    dl = DataLoader(Subset(ds, idx.tolist()), batch_size=batch,
                    num_workers=workers, collate_fn=collate)
    C, P, Y = [], [], []
    for imgs, attrs in tqdm(dl, ncols=80):
        c, p = encode(imgs)
        C.append(c); P.append(p); Y.append(attrs.numpy())
    return (np.concatenate(C).astype(np.float32),
            np.concatenate(P).astype(np.float32),
            np.concatenate(Y).astype(np.float64))


# ------------------------------------------------------------ one cell

def run_cell(X, Y, tag):
    """One (encoder, weights, feature-mode) cell; returns rows for the three
    normalisations, matching 4_reverse_predictivity.py's structure."""
    X_tr, X_te, Z_tr, Z_te = train_test_split(
        X, Y, test_size=0.5, random_state=SPLIT_SEED)

    r = np.linalg.matrix_rank(Z_tr - Z_tr.mean(0))
    fwd = forward_probe(X_tr, X_te, Z_tr, Z_te)
    pr = participation_ratio(X_tr)
    ceil = ceiling_top_r(X_tr, r)

    rows = []

    r2_raw = reverse_r2(Z_tr, X_tr, Z_te, X_te)
    lo, hi = rev_bootstrap_ci(Z_tr, X_tr, Z_te, X_te)
    rows.append(dict(**tag, normalization="raw", whiten_k=0, rank_Y=r,
                     forward_r2=fwd, R2_rev=r2_raw, rev_lo=lo, rev_hi=hi,
                     ceiling=ceil, rev_over_ceiling=r2_raw / ceil,
                     part_ratio=pr))

    X_tr_s, X_te_s = standardize(X_tr, X_te)
    rows.append(dict(**tag, normalization="standardized", whiten_k=0, rank_Y=r,
                     forward_r2=fwd, R2_rev=reverse_r2(Z_tr, X_tr_s, Z_te, X_te_s),
                     rev_lo=np.nan, rev_hi=np.nan, ceiling=np.nan,
                     rev_over_ceiling=np.nan, part_ratio=pr))

    # pca_whitened at k = rank(Y): this is the spectral concentration SC
    X_tr_w, X_te_w, _ = pca_whiten(X_tr, X_te, n_components=r)
    rows.append(dict(**tag, normalization="pca_whitened", whiten_k=r, rank_Y=r,
                     forward_r2=fwd, R2_rev=reverse_r2(Z_tr, X_tr_w, Z_te, X_te_w),
                     rev_lo=np.nan, rev_hi=np.nan, ceiling=np.nan,
                     rev_over_ceiling=np.nan, part_ratio=pr))
    return rows


# ------------------------------------------------------------ self-test

def selftest(seed=0, n=20000, d=768, k=40, tol=1e-3):
    """Affine recodings of binary factors leave every reverse statistic
    unchanged: reverse R^2 (raw / standardized / whitened = SC) and the
    ceiling rank r. Runs on synthetic data, no GPU or CelebA needed."""
    rng = np.random.default_rng(seed)
    # binary factors with CelebA-like prevalences (some attributes are rare)
    Y = (rng.random((n, k)) < rng.uniform(0.02, 0.5, size=k)).astype(np.float64)
    X = (Y @ rng.normal(size=(k, d)) + 2.0 * rng.normal(size=(n, d))
         + np.tanh(rng.normal(size=(n, 3)) @ rng.normal(size=(3, d))))
    recodings = {"01": Y, "pm1": 2 * Y - 1,
                 "zscore": (Y - Y.mean(0)) / Y.std(0)}
    out = {}
    for name, Z in recodings.items():
        X_tr, X_te, Z_tr, Z_te = train_test_split(
            X, Z, test_size=0.5, random_state=SPLIT_SEED)
        r = np.linalg.matrix_rank(Z_tr - Z_tr.mean(0))
        X_tr_s, X_te_s = standardize(X_tr, X_te)
        X_tr_w, X_te_w, _ = pca_whiten(X_tr, X_te, n_components=r)
        out[name] = np.array([r,
                              reverse_r2(Z_tr, X_tr, Z_te, X_te),
                              reverse_r2(Z_tr, X_tr_s, Z_te, X_te_s),
                              reverse_r2(Z_tr, X_tr_w, Z_te, X_te_w),
                              ceiling_top_r(X_tr, r)])
    ref = out["01"]
    for name, v in out.items():
        # OLS would be exactly invariant; ridge (alpha=1) is not invariant to
        # rescaling Z, so recodings agree only up to a small penalty effect
        assert np.allclose(v, ref, atol=tol, rtol=0), (name, v, ref)
    diff = max(abs(v - ref).max() for v in out.values())
    print("selftest passed: affine recodings change R2_rev / SC / ceiling "
          f"by at most {diff:.1e} (ridge penalty effect; OLS would give 0)")
    return diff


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba-root", default=None)
    ap.add_argument("--out", default="results_celeba.csv")
    ap.add_argument("--feat-dir", default="features_celeba")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS),
                    choices=list(ENCODERS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--selftest", action="store_true",
                    help="run the affine-recoding check and exit")
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return
    if args.celeba_root is None:
        ap.error("--celeba-root is required unless --selftest is given")
    os.makedirs(args.feat_dir, exist_ok=True)

    rows = []
    for name in args.encoders:
        builder = ENCODERS[name]
        variants = [("pre", False, -1)] + [
            ("rand", True, s) for s in range(N_RANDOM_SEEDS)]

        for label, rnd, seed in variants:
            fbase = os.path.join(args.feat_dir,
                                 f"{CACHE_TAG[name]}_{label}{seed if rnd else ''}")
            if os.path.exists(fbase + "_cls.npy"):
                C = np.load(fbase + "_cls.npy")
                P = np.load(fbase + "_pmean.npy")
                Y = np.load(os.path.join(args.feat_dir, "attrs.npy"))
            else:
                encode = builder(random_init=rnd, seed=seed)
                C, P, Y = extract(encode, args.celeba_root, batch=args.batch)
                np.save(fbase + "_cls.npy", C)
                np.save(fbase + "_pmean.npy", P)
                np.save(os.path.join(args.feat_dir, "attrs.npy"), Y)
                del encode
                torch.cuda.empty_cache()

            for mode, X in (("cls", C), ("patches_mean", P)):
                if name == "clip_b16" and mode == "patches_mean":
                    continue
                tag = dict(encoder=name, weights=label, seed=seed, features=mode)
                rows.extend(run_cell(X, Y, tag))
                if label == "pre":
                    # real-data check of the appendix claim: recoding the binary
                    # attributes as +-1 leaves reverse R^2 unchanged up to ridge
                    X_tr, X_te, Z_tr, Z_te = train_test_split(
                        X, Y, test_size=0.5, random_state=SPLIT_SEED)
                    d01 = reverse_r2(Z_tr, X_tr, Z_te, X_te)
                    dpm = reverse_r2(2 * Z_tr - 1, X_tr, 2 * Z_te - 1, X_te)
                    print(f"  recoding check {name}/{mode}: |R2_rev(0/1) - "
                          f"R2_rev(+-1)| = {abs(d01 - dpm):.1e}")
                print(f"[{name}/{label}{seed if rnd else ''}/{mode}] done")

            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n=== summary (raw normalisation, CLS) ===")
    sub = df[(df.normalization == "raw") & (df.features == "cls")]
    print(sub.groupby(["encoder", "weights"])
             [["forward_r2", "R2_rev", "rev_over_ceiling", "part_ratio"]]
             .mean().round(4))

    print("\n=== reverse gap (pretrained minus mean random), raw CLS ===")
    for enc in sub.encoder.unique():
        pre = sub[(sub.encoder == enc) & (sub.weights == "pre")].R2_rev.iloc[0]
        rnd = sub[(sub.encoder == enc) & (sub.weights == "rand")].R2_rev.mean()
        print(f"  {enc}: {pre - rnd:+.4f}")


if __name__ == "__main__":
    main()
