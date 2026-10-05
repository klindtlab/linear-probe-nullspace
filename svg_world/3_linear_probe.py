"""
Linear probe: predict latents Z from DINOv3 features on Scene-World.

Reports per-latent ridge R^2 on a 50/50 train/test split. Default scope:
the 20 actually-used latent dims (4 people * {x, z, facing, skin_h} +
2 animals * {x, z}); the other 12 dims are not consumed by the renderer
(shirt h/s/v fixed per lane) and are uninformative by construction.

Within-domain: train and test on the same theme.
Cross-domain (--cross): fit on island train, evaluate on western test
(and the reverse). Features are mean-centered per domain before fitting,
so what's tested is whether the *direction* of latent encoding transfers
across visual style, not whether feature means happen to align.

Feature modes:
  --features cls            CLS token only (768-d)
  --features patches_mean   mean-pooled patch tokens (768-d)
  --features patches_concat all patches flattened (~150k-d), reduced to
                            n_pca dims with PCA (default 1024)

Usage:
  python 3_linear_probe.py
  python 3_linear_probe.py --features patches_mean
  python 3_linear_probe.py --cross
  python 3_linear_probe.py --features patches_concat --n_pca 1024 --cross
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------------
# Latent layout (must match svg_island.py / svg_western.py)
# ----------------------------------------------------------------------------
def _make_labels():
    labels = []
    per_person = ["x", "z", "facing", "skin_h", "shirt_h_un", "shirt_s_un", "shirt_v_un"]
    for p in range(4):
        for d in per_person:
            labels.append(f"p{p}_{d}")
    labels += ["dolphin_x", "dolphin_z", "turtle_x", "turtle_z"]
    return labels

LATENT_LABELS = _make_labels()                          # length 32

# 20 dims actually consumed by the renderer.
USED_DIMS = (
    [p * 7 + i for p in range(4) for i in (0, 1, 2, 3)]  # x, z, facing, skin_h
    + [28, 29, 30, 31]                                    # animal positions
)
USED_LABELS = [LATENT_LABELS[i] for i in USED_DIMS]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def r2_per_dim(y_true, y_pred):
    """1 - SSE/SST per output column. Equivalent to sklearn r2_score with
    multioutput='raw_values' but with a small variance floor for stability."""
    tot = y_true.var(axis=0) + 1e-8
    res = (y_true - y_pred).var(axis=0)
    return 1.0 - res / tot


def pearson_per_dim(y_true, y_pred):
    """Pearson correlation per output column. Insensitive to bias and scale,
    so a probe that has the right direction but wrong intercept/slope (e.g.
    cross-domain) shows high ρ even when R² is negative."""
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12
    return num / den


def load_features(root, theme, variant, mode, n_pca, fit_pca_on=None):
    """Returns an (N, d) feature matrix according to `mode`.

    For mode == 'patches_concat' a PCA fitted on `fit_pca_on` (or self,
    if None) is applied. fit_pca_on is the matrix used for fitting; that
    lets cross-domain probing fit PCA on the source theme and reuse it.
    Returns (X, pca_object_or_None).
    """
    fdir = os.path.join(root, theme, "features")
    if mode == "cls":
        return np.load(os.path.join(fdir, f"cls_{variant}.npy")), None

    patches = np.load(os.path.join(fdir, f"patches_{variant}.npy"))   # (N, P, D)
    if mode == "patches_mean":
        return patches.mean(axis=1), None
    if mode == "patches_concat":
        flat = patches.reshape(patches.shape[0], -1)                  # (N, P*D)
        if fit_pca_on is None:
            pca = PCA(n_components=n_pca, random_state=42).fit(flat)
            return pca.transform(flat).astype(np.float32), pca
        # reuse a previously-fit PCA
        return fit_pca_on.transform(flat).astype(np.float32), fit_pca_on
    raise ValueError(f"unknown features mode: {mode}")


def fit_eval(X_train, X_test, y_train, y_test, alpha):
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    return r2_per_dim(y_test, y_pred), pearson_per_dim(y_test, y_pred)


# ----------------------------------------------------------------------------
# Probe routines
# ----------------------------------------------------------------------------
def within_domain(root, theme, variant, mode, n_pca, Z, alpha, seed=42):
    X, _ = load_features(root, theme, variant, mode, n_pca)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, Z, test_size=0.5, random_state=seed
    )
    return fit_eval(X_tr, X_te, y_tr, y_te, alpha)


def cross_domain(root, src, tgt, variant, mode, n_pca, Z, alpha, seed=42):
    """Fit ridge on src train, evaluate on tgt test. Per-domain mean
    centering on the *fit* mean (i.e. subtract src train mean from src and
    tgt test mean from tgt) so we test direction transfer not bias transfer.
    """
    X_src, pca = load_features(root, src, variant, mode, n_pca)
    X_tgt, _   = load_features(root, tgt, variant, mode, n_pca, fit_pca_on=pca)

    # Same random_state -> same train/test indices in both themes; we use
    # src's train half to fit and tgt's test half to evaluate. Since the Z
    # matrices are identical across themes, train/test Z slices match the
    # corresponding feature halves.
    X_src_tr, _,         y_tr,   _    = train_test_split(X_src, Z, test_size=0.5, random_state=seed)
    _,        X_tgt_te,  _,      y_te = train_test_split(X_tgt, Z, test_size=0.5, random_state=seed)

    # Per-domain centering: remove the source train mean from source and
    # the target test mean from target. This isolates the direction.
    mu_src = X_src_tr.mean(axis=0, keepdims=True)
    mu_tgt = X_tgt_te.mean(axis=0, keepdims=True)
    X_src_tr = X_src_tr - mu_src
    X_tgt_te = X_tgt_te - mu_tgt

    return fit_eval(X_src_tr, X_tgt_te, y_tr, y_te, alpha)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(name, r2, rho, used_only=True):
    if used_only:
        r2_show  = r2[USED_DIMS]
        rho_show = rho[USED_DIMS]
        labels   = USED_LABELS
    else:
        r2_show  = r2
        rho_show = rho
        labels   = LATENT_LABELS

    # Per-group breakdown for quick scanning
    geom_idx   = [i for i, lbl in enumerate(labels) if lbl.endswith("_x") or lbl.endswith("_z")]
    facing_idx = [i for i, lbl in enumerate(labels) if "facing" in lbl]
    color_idx  = [i for i, lbl in enumerate(labels) if "skin_h" in lbl]

    print(f"\n--- {name} ---")
    print(f"  mean (all reported):       R^2 {r2_show.mean():>+.4f}   |rho| {np.abs(rho_show).mean():>+.4f}   rho {rho_show.mean():>+.4f}")
    if geom_idx:
        print(f"  positions x/z:             R^2 {r2_show[geom_idx].mean():>+.4f}   |rho| {np.abs(rho_show[geom_idx]).mean():>+.4f}   rho {rho_show[geom_idx].mean():>+.4f}")
    if facing_idx:
        print(f"  facing (capped):           R^2 {r2_show[facing_idx].mean():>+.4f}   |rho| {np.abs(rho_show[facing_idx]).mean():>+.4f}   rho {rho_show[facing_idx].mean():>+.4f}")
    if color_idx:
        print(f"  skin_h:                    R^2 {r2_show[color_idx].mean():>+.4f}   |rho| {np.abs(rho_show[color_idx]).mean():>+.4f}   rho {rho_show[color_idx].mean():>+.4f}")
    print()
    print(f"    {'dim':<14} {'R^2':>9}    {'rho':>7}")
    for lbl, r, p in zip(labels, r2_show, rho_show):
        print(f"    {lbl:<14} {r:>+9.4f}    {p:>+7.4f}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="scene_world")
    parser.add_argument("--features",
                        choices=["cls", "patches_mean", "patches_concat"],
                        default="cls")
    parser.add_argument("--cross", action="store_true",
                        help="also run cross-domain probing")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="ridge regularization")
    parser.add_argument("--n_pca", type=int, default=1024,
                        help="PCA dims for patches_concat mode")
    parser.add_argument("--all_dims", action="store_true",
                        help="report all 32 latent dims (default: 20 used)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", default=None,
                        help="Optional path to write per-dim metrics as a CSV. "
                             "Columns: condition, variant, dim, R2, rho.")
    args = parser.parse_args()

    Z = np.load(os.path.join(args.root, "Z.npy"))
    print(f"Z shape: {Z.shape}")
    print(f"Feature mode: {args.features}")
    print(f"Ridge alpha: {args.alpha}")
    print(f"Reporting: {'all 32 dims' if args.all_dims else '20 used dims'}")

    # Accumulate (condition, variant, dim_label, r2, rho) tuples for CSV output.
    rows = []

    def record(condition, variant, r2, rho):
        for i, lbl in enumerate(LATENT_LABELS):
            rows.append((condition, variant, lbl, float(r2[i]), float(rho[i])))

    # Within-domain
    for theme in ("island", "western"):
        for variant in ("pre", "rand"):
            r2, rho = within_domain(
                args.root, theme, variant,
                args.features, args.n_pca, Z, args.alpha, args.seed,
            )
            report(f"{theme} | {variant} | within-domain", r2, rho,
                   used_only=not args.all_dims)
            record(f"within_{theme}", variant, r2, rho)

    # Cross-domain
    if args.cross:
        for src, tgt in (("island", "western"), ("western", "island")):
            for variant in ("pre", "rand"):
                r2, rho = cross_domain(
                    args.root, src, tgt, variant,
                    args.features, args.n_pca, Z, args.alpha, args.seed,
                )
                report(
                    f"train={src} -> test={tgt} | {variant} | cross-domain",
                    r2, rho, used_only=not args.all_dims,
                )
                record(f"cross_{src}_to_{tgt}", variant, r2, rho)

    # Optional CSV output. Default name follows the feature mode if no path given.
    if args.out_csv is not None:
        out_path = args.out_csv
    else:
        out_path = os.path.join(args.root, f"probe_results_{args.features}.csv")
    with open(out_path, "w") as f:
        f.write("condition,variant,dim,R2,rho\n")
        for condition, variant, dim, r2, rho in rows:
            f.write(f"{condition},{variant},{dim},{r2:.6f},{rho:.6f}\n")
    print(f"\nwrote per-dim metrics to {out_path}")


if __name__ == "__main__":
    main()
