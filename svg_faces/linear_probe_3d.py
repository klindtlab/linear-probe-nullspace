import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from linear_probe_functions import load_dino_features, load_metadata_features

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def var_exp(y_true, y_pred):
    """R² per output dimension (column-wise)."""
    tot = y_true.var(axis=0) + 1e-6
    res = (y_true - y_pred).var(axis=0)
    return 1.0 - res / tot


def fit_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Fit a Ridge probe and return per-dim R²."""
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    preds = probe.predict(X_test)
    return var_exp(y_test, preds)          # shape (n_outputs,)


# ── per-latent-feature R² breakdown ──────────────────────────────────────────

def per_feature_r2(X_train, X_test, y_train, y_test, alpha=1.0):
    """
    For each of the 19 latent features individually, fit a probe that predicts
    all 768 activation dims.  Returns an array of shape (19, 768).

    This tells us: "how much does *this one geometric feature* alone explain
    each activation dimension?"
    """
    n_features = X_train.shape[1]       # 19
    n_acts     = y_train.shape[1]       # 768
    surface    = np.zeros((n_features, n_acts))

    for fi in range(n_features):
        xi_train = X_train[:, fi : fi + 1]   # keep 2-D
        xi_test  = X_test[:,  fi : fi + 1]
        surface[fi] = fit_probe(xi_train, y_train, xi_test, y_test, alpha)

    return surface   # (19, 768)


# ── 3-D surface plot ──────────────────────────────────────────────────────────

def plot_3d_surface(surface_pre, surface_rand, labels, save_path="r2_surface_3d.png"):
    """
    Two-panel 3-D surface:
      X axis  = activation dimension  (0 … 767)
      Y axis  = latent feature index  (0 … 18)
      Z axis  = R²

    Left  → Pretrained DINOv3
    Right → Random DINOv3

    A difference heatmap (rand − pre) is added below so you can see
    exactly where random wins.
    """
    n_feats, n_acts = surface_pre.shape
    act_idx  = np.arange(n_acts)
    feat_idx = np.arange(n_feats)
    ACT, FEAT = np.meshgrid(act_idx, feat_idx)   # both (19, 768)

    z_min = min(surface_pre.min(), surface_rand.min(), 0.0)
    z_max = max(surface_pre.max(), surface_rand.max())

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Reverse Linear Probe: R² Surface\n"
        "(Latent feature → Activation dimension)\n"
        "X = activation dim · Y = latent feature · Z = R²",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── panel 1 : Pretrained ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        ACT, FEAT, surface_pre,
        cmap="viridis", linewidth=0, antialiased=True, alpha=0.9,
    )
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel("Activation dim", fontsize=9, labelpad=6)
    ax1.set_ylabel("Latent feature", fontsize=9, labelpad=6)
    ax1.set_zlabel("R²", fontsize=9, labelpad=4)
    ax1.set_yticks(feat_idx)
    ax1.set_yticklabels(labels, fontsize=6)
    ax1.set_title("Pretrained DINOv3", fontsize=12, pad=8)
    fig.colorbar(surf1, ax=ax1, shrink=0.45, label="R²")

    # ── panel 2 : Random ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        ACT, FEAT, surface_rand,
        cmap="plasma", linewidth=0, antialiased=True, alpha=0.9,
    )
    ax2.set_zlim(z_min, z_max)
    ax2.set_xlabel("Activation dim", fontsize=9, labelpad=6)
    ax2.set_ylabel("Latent feature", fontsize=9, labelpad=6)
    ax2.set_zlabel("R²", fontsize=9, labelpad=4)
    ax2.set_yticks(feat_idx)
    ax2.set_yticklabels(labels, fontsize=6)
    ax2.set_title("Random DINOv3", fontsize=12, pad=8)
    fig.colorbar(surf2, ax=ax2, shrink=0.45, label="R²")

    # ── panel 3 : Difference heatmap (rand − pre) ────────────────────────────
    diff = surface_rand - surface_pre          # positive → random wins
    ax3 = fig.add_subplot(2, 2, 3)
    vabs = np.abs(diff).max()
    im = ax3.imshow(
        diff, aspect="auto", cmap="RdBu_r",
        vmin=-vabs, vmax=vabs,
        origin="lower",
    )
    ax3.set_xlabel("Activation dimension", fontsize=10)
    ax3.set_ylabel("Latent feature", fontsize=10)
    ax3.set_yticks(feat_idx)
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_title(
        "Difference  (Random − Pretrained)\nBlue = pretrained wins · Red = random wins",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax3, label="ΔR²")

    # ── panel 4 : Average R² per latent feature (bar chart) ──────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    avg_pre  = surface_pre.mean(axis=1)    # (19,)
    avg_rand = surface_rand.mean(axis=1)   # (19,)
    x = np.arange(n_feats)
    w = 0.35
    ax4.barh(x - w/2, avg_pre,  w, label="Pretrained", color="royalblue")
    ax4.barh(x + w/2, avg_rand, w, label="Random",     color="tomato")
    ax4.set_yticks(x)
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel("Mean R² across all 768 activation dims", fontsize=9)
    ax4.set_title("Which latent features are most recoverable?", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"\nPlot saved → {save_path}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def plot_4bar_summary(avg_std_pre, avg_std_rand, avg_rev_pre, avg_rev_rand,
                      save_path="summary_4bar.png"):
    """
    4 bars comparing both probe directions and both models.
      Left pair  : Activations → Latent   (standard)
      Right pair : Latent → Activations   (reverse)
    """
    from matplotlib.patches import Patch

    bars    = [avg_std_pre, avg_std_rand, avg_rev_pre, avg_rev_rand]
    colors  = ["royalblue", "tomato", "royalblue", "tomato"]
    xlabels = [
        "Pretrained\n(Act → Lat)",
        "Random\n(Act → Lat)",
        "Pretrained\n(Lat → Act)",
        "Random\n(Lat → Act)",
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    x     = np.arange(4)
    rects = ax.bar(x, bars, color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    for rect, val in zip(rects, bars):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.004,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ymax = max(bars) * 1.18
    ax.set_ylim(0, ymax)
    ax.text(0.5, 0.97, "Activations → Latent", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=10, color="dimgray", style="italic")
    ax.text(2.5, 0.97, "Latent → Activations", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=10, color="dimgray", style="italic")

    ax.legend(handles=[
        Patch(color="royalblue", label="Pretrained DINOv3"),
        Patch(color="tomato",    label="Random DINOv3"),
    ], fontsize=11, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("Mean R²", fontsize=12)
    ax.set_title("Linear Probe Summary\nBoth Directions · Both Models",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse mode: predict activations from latent features")
    parser.add_argument("--both", action="store_true",
                        help="Run both directions and plot a 4-bar summary")
    parser.add_argument("--cube", action="store_true",          # BUG FIX: was missing
                        help="Cube-transform the latent features before probing")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Ridge regularisation (default 10.0)")
    args = parser.parse_args()

    meta_dir = "svg_face_dataset_one_face/meta"

    # 1. Load features
    ids_pre,  X_layers_pre  = load_dino_features("dino_pretrained_10k.npz")
    ids_rand, X_layers_rand = load_dino_features("dino_random_10k.npz")
    assert ids_pre == ids_rand, "ID mismatch between pretrained and random!"

    X_pre  = X_layers_pre[:,  -1, :]   # last layer → (N, 768)
    X_rand = X_layers_rand[:, -1, :]

    # 2. Load latent metadata
    y, labels = load_metadata_features(meta_dir, ids_pre)   # (N, 19)
    if args.cube:
        y = y ** 3

    print(f"X_pre shape : {X_pre.shape}")
    print(f"y shape     : {y.shape}")

    # ── BOTH MODE : run both directions, just plot 4-bar summary ─────────────
    if args.both:
        print("\n── Direction 1: Activations → Latent ──")
        X_tr_pre,  X_te_pre,  y_tr, y_te = train_test_split(X_pre,  y, test_size=0.2, random_state=SEED)
        X_tr_rand, X_te_rand, _,    _    = train_test_split(X_rand, y, test_size=0.2, random_state=SEED)
        r2_std_pre  = fit_probe(X_tr_pre,  y_tr, X_te_pre,  y_te, args.alpha)  # (19,)
        r2_std_rand = fit_probe(X_tr_rand, y_tr, X_te_rand, y_te, args.alpha)

        print("\n── Direction 2: Latent → Activations ──")
        y_tr2, y_te2, X_tr_pre2,  X_te_pre2  = train_test_split(y, X_pre,  test_size=0.2, random_state=SEED)
        _,     _,     X_tr_rand2, X_te_rand2 = train_test_split(y, X_rand, test_size=0.2, random_state=SEED)
        r2_rev_pre  = fit_probe(y_tr2, X_tr_pre2,  y_te2, X_te_pre2,  args.alpha)  # (768,)
        r2_rev_rand = fit_probe(y_tr2, X_tr_rand2, y_te2, X_te_rand2, args.alpha)

        print(f"\n  Act→Lat   avg R²  |  Pretrained: {r2_std_pre.mean():.4f}  |  Random: {r2_std_rand.mean():.4f}")
        print(f"  Lat→Act   avg R²  |  Pretrained: {r2_rev_pre.mean():.4f}  |  Random: {r2_rev_rand.mean():.4f}")

        plot_4bar_summary(
            r2_std_pre.mean(), r2_std_rand.mean(),
            r2_rev_pre.mean(), r2_rev_rand.mean(),
        )
        return

    # ── STANDARD MODE : activations → latent ─────────────────────────────────
    if not args.reverse:
        print("\nStandard R²: Activations → Latent features")
        X_tr_pre,  X_te_pre,  y_tr, y_te = train_test_split(X_pre,  y, test_size=0.2, random_state=SEED)
        X_tr_rand, X_te_rand, _,    _    = train_test_split(X_rand, y, test_size=0.2, random_state=SEED)

        r2_pre  = fit_probe(X_tr_pre,  X_te_pre,  y_tr, y_te, args.alpha)
        r2_rand = fit_probe(X_tr_rand, X_te_rand, y_tr, y_te, args.alpha)

        print(f"\n{'Feature':<30} | {'Pretrained':>10} | {'Random':>10}")
        print("-" * 56)
        for i, lbl in enumerate(labels):
            print(f"{lbl:<30} | {r2_pre[i]:>10.4f} | {r2_rand[i]:>10.4f}")
        print(f"\n{'AVERAGE':<30} | {r2_pre.mean():>10.4f} | {r2_rand.mean():>10.4f}")

        # Simple bar chart for standard mode
        fig, ax = plt.subplots(figsize=(14, 6))
        x, w = np.arange(len(labels)), 0.35
        ax.bar(x - w/2, r2_pre,  w, label="Pretrained", color="royalblue")
        ax.bar(x + w/2, r2_rand, w, label="Random",     color="lightcoral")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("R²"); ax.set_title("Standard Linear Probe R²", fontsize=14)
        ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("variance_explained_standard.png", dpi=200)
        print("\nSaved → variance_explained_standard.png")
        plt.show()

    # ── REVERSE MODE : latent → activations ──────────────────────────────────
    else:
        print("\nReverse R²: Latent features → Activations (768 dims)")

        # Split on y (latent features are the input in reverse mode)
        y_tr, y_te, X_tr_pre,  X_te_pre  = train_test_split(y, X_pre,  test_size=0.2, random_state=SEED)
        _,    _,    X_tr_rand, X_te_rand  = train_test_split(y, X_rand, test_size=0.2, random_state=SEED)

        # Global R² (all 19 latent features together → each activation dim)
        print("  Fitting global probes (all 19 features at once)…")
        r2_pre_global  = fit_probe(y_tr, X_tr_pre,  y_te, X_te_pre,  args.alpha)  # (768,)
        r2_rand_global = fit_probe(y_tr, X_tr_rand, y_te, X_te_rand, args.alpha)

        print(f"\n  Global avg R²  →  Pretrained: {r2_pre_global.mean():.4f} | Random: {r2_rand_global.mean():.4f}")

        # Per-latent-feature R² surface (19 × 768)
        print("  Fitting per-feature probes (19 × 768 grid)… [this takes ~30s]")
        surface_pre  = per_feature_r2(y_tr, y_te, X_tr_pre,  X_te_pre,  args.alpha)
        surface_rand = per_feature_r2(y_tr, y_te, X_tr_rand, X_te_rand, args.alpha)

        print(f"\n  Surface shape: {surface_pre.shape}  (latent features × activation dims)")

        # Save raw surfaces so you can re-plot without re-running
        np.savez("r2_surfaces.npz",
                 surface_pre=surface_pre, surface_rand=surface_rand,
                 labels=labels)
        print("  Raw surfaces saved → r2_surfaces.npz")

        # 3D plot
        plot_3d_surface(surface_pre, surface_rand, labels,
                        save_path="r2_surface_3d.png")


if __name__ == "__main__":
    main()