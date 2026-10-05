import numpy as np
import matplotlib.pyplot as plt

def main():
    
    feature_matrix = np.load("Z_10k_one_face.npy")
    
    # our 15 features that we need from the z matrix
    labels = [
        "face_radius", "face_cx", " face_cy", 
        "eye_radius", "eye_spacing", "eye_y_offset", 
        "mouth_width", "mouth_y_offset", "mouth_curve",
        "skin_h", "skin_s", "skin_v",
        "eye_h", "eye_s", "eye_v"
    ]
    
    print(f"Z matrix shape: {feature_matrix.shape}")
    
    # calculate the correlation matrix of the 15 features 
    corr = np.corrcoef(feature_matrix, rowvar=False)
    
    # plot the correlation matrix
    plt.figure(figsize=(10, 8))
    # Use coolwarm and set bounds to -1 and 1 for proper correlation colors
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    
    # use dynamic length for ticks
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.colorbar(label="Pearson Correlation")
    
    plt.title("Correlation Matrix of Generative Z Features", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")

if __name__ == "__main__":
    main()
