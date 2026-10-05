import os
from linear_probe_functions import extract_dino_features, save_dino_features

def main():
    png_dir = "svg_face_dataset_one_face/pngs"
    
    # Pretrained Model
    print("Extracting Pretrained DINOv3 Features...")
    ids_pretrained, features_pretrained = extract_dino_features(png_dir, pretrained=True)
    save_dino_features("dino_pretrained_10k.npz", ids_pretrained, features_pretrained)
    
    # Random Init Model
    print("Extracting Random DINOv3 Features...")
    ids_random, features_random = extract_dino_features(png_dir, pretrained=False)
    save_dino_features("dino_random_10k.npz", ids_random, features_random)
    
    print("All extractions completed successfully!")

if __name__ == "__main__":
    main()
