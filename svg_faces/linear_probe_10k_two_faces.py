import os
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from linear_probe_functions import load_dino_features, load_metadata_features_two_faces

SEED = 42

def var_exp(y, y_):
    """Computes Variance Explained (R^2)"""
    tot_var = y.var(axis=0) + 1e-6
    res_var = (y - y_).var(axis=0)
    return 1 - (res_var / tot_var)

def main():
    parser = argparse.ArgumentParser(description="Linear Probe on DINOv3 Features for Two Faces")
    parser.add_argument("--reverse", action="store_true", help="Run in Reverse R^2 mode (predict activations from features)")
    args = parser.parse_args()

    meta_dir = "svg_face_dataset_two_face/meta"
    
    # 1. Load Pretrained Features
    ids_pre, X_layers_pre = load_dino_features("dino_pretrained_10k_twofaces.npz")
    # Take the last layer
    X_pre = X_layers_pre[:, -1, :]
    
    # 2. Load Random Features
    ids_rand, X_layers_rand = load_dino_features("dino_random_10k_twofaces.npz")
    X_rand = X_layers_rand[:, -1, :]
    
    assert ids_pre == ids_rand, "Mismatch in image IDs between pretrained and random models!"
    image_ids = ids_pre
    
    # 3. Load Target Metadata (Two Faces -> 38 Features)
    y, labels = load_metadata_features_two_faces(meta_dir, image_ids)
    
    print(f"X_pre shape: {X_pre.shape}, y shape (Two Faces): {y.shape}")
    
    # 4. Train/Test Split & Fit
    if args.reverse:
        print("Running Reverse R^2: Predicting Activations from Features.")
        # In reverse mode, y (38 features) becomes the input X, and X_pre/X_rand (768 dims) become the target y
        X_train, X_test, y_train_pre, y_test_pre = train_test_split(y, X_pre, test_size=0.2, random_state=SEED)
        _, _, y_train_rand, y_test_rand = train_test_split(y, X_rand, test_size=0.2, random_state=SEED)
        
        # Fit Ridge Probe for Pretrained
        print("Fitting Ridge Probe for Pretrained DINOv3...")
        probe_pre = Ridge(alpha=1.0)
        probe_pre.fit(X_train, y_train_pre)
        preds_pre = probe_pre.predict(X_test)
        r2_pre = var_exp(y_test_pre, preds_pre)
        
        # Fit Ridge Probe for Random
        print("Fitting Ridge Probe for Random DINOv3...")
        probe_rand = Ridge(alpha=1.0)
        probe_rand.fit(X_train, y_train_rand)
        preds_rand = probe_rand.predict(X_test)
        r2_rand = var_exp(y_test_rand, preds_rand)
        
        print("\nReverse R^2 completed. Skipping plots for now as requested.")
        
    else:
        print("Running Standard R^2: Predicting Features from Activations.")
        X_train_pre, X_test_pre, y_train, y_test = train_test_split(X_pre, y, test_size=0.2, random_state=SEED)
        X_train_rand, X_test_rand, _, _ = train_test_split(X_rand, y, test_size=0.2, random_state=SEED)
        
        # Fit Ridge Probe for Pretrained
        print("Fitting Ridge Probe for Pretrained DINOv3...")
        probe_pre = Ridge(alpha=1.0)
        probe_pre.fit(X_train_pre, y_train)
        preds_pre = probe_pre.predict(X_test_pre)
        r2_pre = var_exp(y_test, preds_pre)
        
        # Fit Ridge Probe for Random
        print("Fitting Ridge Probe for Random DINOv3...")
        probe_rand = Ridge(alpha=1.0)
        probe_rand.fit(X_train_rand, y_train)
        preds_rand = probe_rand.predict(X_test_rand)
        r2_rand = var_exp(y_test, preds_rand)
        
        print("\nVariance Explained (R^2) per Feature:")
        for i, label in enumerate(labels):
            print(f"{label:<30} | Pretrained: {r2_pre[i]:.4f} | Random: {r2_rand[i]:.4f}")
            
    avg_pre = np.mean(r2_pre)
    avg_rand = np.mean(r2_rand)
    print(f"\nAverage Variance Explained -> Pretrained: {avg_pre:.4f} | Random: {avg_rand:.4f}")

    if not args.reverse:
        import matplotlib.pyplot as plt
        
        # 8. Plot Comparison
        plt.figure(figsize=(24, 8)) # Wider figure for 38 features
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, r2_pre, width, label='Pretrained DINOv3', color='royalblue')
        plt.bar(x + width/2, r2_rand, width, label='Random DINOv3', color='lightcoral')
        
        plt.title("Linear Probe: Variance Explained ($R^2$) across Two-Face Features (Last Layer)", fontsize=16)
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Variance Explained ($R^2$)", fontsize=14)
        plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = "variance_explained_comparison_two_faces.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"\nPlot saved successfully to {plot_filename}!")


if __name__ == "__main__":
    main()
