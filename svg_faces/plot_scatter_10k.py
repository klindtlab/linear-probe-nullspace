import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from linear_probe_functions import load_dino_features, load_metadata_features

def plot_scatter_grid(y_true, y_pred, labels, title, filename):
    """
    Plots a grid of scatter plots (Ground Truth vs Predicted) for each feature.
    Calculates and displays the Pearson correlation coefficient (rho) in the title.
    """
    num_features = len(labels)
    # We have 19 features, so a 4x5 grid will fit them (20 subplots)
    cols = 5
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
    axes = axes.flatten()
    
    for i, label in enumerate(labels):
        ax = axes[i]
        gt = y_true[:, i]
        pred = y_pred[:, i]
        
        # Calculate Pearson correlation
        # Handle cases where variance is 0 (e.g., constant outputs)
        if np.std(pred) == 0 or np.std(gt) == 0:
            rho = 0.0
        else:
            rho, _ = pearsonr(gt, pred)
            
        ax.scatter(gt, pred, s=5, alpha=0.5, color='cornflowerblue')
        ax.set_title(f"{label}\n$\\rho={rho:.4f}$", fontsize=12)
        ax.set_xlabel("Ground Truth", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    # Hide any unused subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot grid to {filename}")
    plt.close()

def main():
    png_dir = "svg_face_dataset_one_face/pngs"
    meta_dir = "svg_face_dataset_one_face/meta"
    
    # 1. Load Pretrained Features
    ids_pre, X_layers_pre = load_dino_features("dino_pretrained_10k.npz")
    X_pre = X_layers_pre[:, -1, :]
    
    # 2. Load Random Features
    ids_rand, X_layers_rand = load_dino_features("dino_random_10k.npz")
    X_rand = X_layers_rand[:, -1, :]
    
    image_ids = ids_pre
    
    # 3. Load Target Metadata
    y, labels = load_metadata_features(meta_dir, image_ids)
    
    # 4. Train/Test Split
    X_train_pre, X_test_pre, y_train, y_test = train_test_split(X_pre, y, test_size=0.2, random_state=42)
    X_train_rand, X_test_rand, _, _ = train_test_split(X_rand, y, test_size=0.2, random_state=42)
    
    # 5. Fit Ridge Probe for Pretrained and plot
    print("Fitting Ridge Probe for Pretrained DINOv3...")
    probe_pre = Ridge(alpha=1.0)
    probe_pre.fit(X_train_pre, y_train)
    preds_pre = probe_pre.predict(X_test_pre)
    
    plot_scatter_grid(
        y_test, 
        preds_pre, 
        labels, 
        "Pretrained DINOv3: Predicted vs Ground Truth", 
        "scatter_pretrained.png"
    )
    
    # 6. Fit Ridge Probe for Random and plot
    print("Fitting Ridge Probe for Random DINOv3...")
    probe_rand = Ridge(alpha=1.0)
    probe_rand.fit(X_train_rand, y_train)
    preds_rand = probe_rand.predict(X_test_rand)
    
    plot_scatter_grid(
        y_test, 
        preds_rand, 
        labels, 
        "Random DINOv3: Predicted vs Ground Truth", 
        "scatter_random.png"
    )

if __name__ == "__main__":
    main()
