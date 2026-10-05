"""
Reverse predictivity (D1) on Scene-World features.

For each (theme, variant), fit a ridge regression Z -> f(x). Report:
    R^2_rev = 1 - Var(f - V z) / Var(f)
which is the fraction of total feature variance linearly explained by the
ground-truth latents. This is a single scalar per (theme, variant).

We compute three variants with different feature normalizations:

  raw           : features as-is (the headline number, sensitive to scale)
  standardized  : each feature dim z-scored (mean 0, var 1) before fitting
  pca_whitened  : PCA-whiten so every principal direction has unit variance,
                  then fit. This isolates DIRECTION alignment from variance
                  concentration.

Why three variants:
  - Raw: what someone would do by default; sensitive to which dimensions
    of f happen to have large variance.
  - Standardized: each feature dim contributes equally regardless of its
    raw magnitude. Less affected by a few high-variance "spike" dims.
  - PCA-whitened: removes all variance-concentration structure; the only
    thing reverse R^2 can measure is how much of the variance is in
    z-aligned directions vs not, on equal footing.

If pretrained's reverse R^2 advantage shrinks under whitening, it means
much of the win is "the model concentrates variance into a few z-aligned
directions". If it survives whitening, it means the win is genuinely
about direction alignment, independent of how spread out the variance is.

Usage:
    python 4_reverse_predictivity.py
    python 4_reverse_predictivity.py --features patches_mean
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_features(root, theme, variant, mode, n_pca=None, fit_pca_on=None):
    """Mirrors 3_linear_probe.py's loader."""
    feat_dir = os.path.join(root, theme, "features")
    if mode == "cls":
        X = np.load(os.path.join(feat_dir, f"cls_{variant}.npy"))
    elif mode == "patches_mean":
        P = np.load(os.path.join(feat_dir, f"patches_{variant}.npy"))
        X = P.mean(axis=1)
    elif mode == "patches_concat":
        P = np.load(os.path.join(feat_dir, f"patches_{variant}.npy"))
        X = P.reshape(P.shape[0], -1)
        if fit_pca_on is None:
            pca = PCA(n_components=n_pca, random_state=0)
            X = pca.fit_transform(X)
        else:
            pca = fit_pca_on
            X = pca.transform(X)
    else:
        raise ValueError(mode)
    return X.astype(np.float32)


def reverse_r2(Z_train, X_train, Z_test, X_test, alpha=1.0):
    """Single scalar reverse R^2: fraction of total feature variance explained
    by a linear function of latents. Computed on test set."""
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(Z_train, X_train)
    X_pred = probe.predict(Z_test)
    # Total feature variance = sum_d Var(X_test[:, d])
    # Residual variance      = sum_d Var(X_test[:, d] - X_pred[:, d])
    total = X_test.var(axis=0).sum() + 1e-12
    resid = (X_test - X_pred).var(axis=0).sum()
    return 1.0 - resid / total


def standardize(X_train, X_test):
    """Z-score per dim, fit stats on train."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def pca_whiten(X_train, X_test, n_components=None, eps=1e-6):
    """PCA-whiten. n_components defaults to min(N, D); we set it conservatively
    to keep components with variance > eps so it's stable."""
    if n_components is None:
        n_components = min(X_train.shape) - 1
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    return pca.fit_transform(X_train), pca.transform(X_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="scene_world")
    parser.add_argument("--features",
                        choices=["cls", "patches_mean", "patches_concat"],
                        default="cls")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_pca", type=int, default=1024,
                        help="for patches_concat input")
    parser.add_argument("--n_pca_whiten", type=int, default=None,
                        help="dim cap for the PCA-whitening preprocessing "
                             "(None = use all reasonable components)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    Z = np.load(os.path.join(args.root, "Z.npy"))
    print(f"Z shape: {Z.shape}")
    print(f"Feature mode: {args.features}")
    print(f"Ridge alpha: {args.alpha}")

    rows = []  # (theme, variant, normalization, R^2_rev)

    for theme in ("island", "western"):
        for variant in ("pre", "rand"):
            X = load_features(args.root, theme, variant,
                              args.features, args.n_pca)
            # Single 50/50 split, shared across the three normalizations.
            X_tr, X_te, Z_tr, Z_te = train_test_split(
                X, Z, test_size=0.5, random_state=args.seed)

            # Raw
            r2_raw = reverse_r2(Z_tr, X_tr, Z_te, X_te, alpha=args.alpha)

            # Standardized
            X_tr_s, X_te_s = standardize(X_tr, X_te)
            r2_std = reverse_r2(Z_tr, X_tr_s, Z_te, X_te_s, alpha=args.alpha)

            # PCA-whitened. Use n_pca_whiten or default to min(N_train, D)-1.
            n_w = args.n_pca_whiten or (min(X_tr.shape) - 1)
            X_tr_w, X_te_w = pca_whiten(X_tr, X_te, n_components=n_w)
            r2_pw = reverse_r2(Z_tr, X_tr_w, Z_te, X_te_w, alpha=args.alpha)

            print(f"\n--- {theme} | {variant} ---")
            print(f"  R^2_rev (raw)                  = {r2_raw:+.4f}")
            print(f"  R^2_rev (standardized)         = {r2_std:+.4f}")
            print(f"  R^2_rev (PCA-whitened, k={n_w:>3}) = {r2_pw:+.4f}")

            rows.append((theme, variant, "raw",          0,    r2_raw))
            rows.append((theme, variant, "standardized", 0,    r2_std))
            rows.append((theme, variant, "pca_whitened", n_w,  r2_pw))

    # Aggregate summary across the two themes
    print("\n=== aggregate (mean over themes) ===")
    for variant in ("pre", "rand"):
        for norm in ("raw", "standardized", "pca_whitened"):
            vals = [r[4] for r in rows
                    if r[1] == variant and r[2] == norm]
            ks = sorted(set(r[3] for r in rows
                            if r[1] == variant and r[2] == norm))
            ks_str = "" if norm != "pca_whitened" else f", k={ks[0]}"
            print(f"  {variant:<5} {norm:<14}{ks_str}  mean R^2_rev = {np.mean(vals):+.4f}")

    # CSV (filename embeds whitening k so sweeps don't overwrite)
    n_w_for_name = args.n_pca_whiten or "full"
    out_path = args.out_csv or os.path.join(
        args.root, f"reverse_results_{args.features}_whitenk{n_w_for_name}.csv")
    with open(out_path, "w") as f:
        f.write("theme,variant,normalization,whiten_k,R2_rev\n")
        for theme, variant, norm, k, r2 in rows:
            f.write(f"{theme},{variant},{norm},{k},{r2:.6f}\n")
    print(f"\nwrote per-condition reverse R^2 to {out_path}")


if __name__ == "__main__":
    main()