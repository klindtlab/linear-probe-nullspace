import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader

class PNGDataset(Dataset):
    """Custom PyTorch Dataset for loading and processing images for DINO."""
    def __init__(self, png_dir, png_files, processor):
        self.png_dir = png_dir
        self.png_files = png_files
        self.processor = processor

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        filename = self.png_files[idx]
        image_id = filename.replace(".png", "")
        img_path = os.path.join(self.png_dir, filename)
        
        image = Image.open(img_path).convert("RGB")
        # The processor adds a batch dimension of 1, we squeeze it out so DataLoader can batch it properly
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        return image_id, pixel_values


def var_exp(y, y_):
    """Computes Variance Explained"""
    tot_var = y.var() + 1e-6
    res_var = (y - y_).var()
    return 1 - res_var / tot_var

# def extract_dino_features(png_dir, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", pretrained=True, device=None):
#     """
#     Loads DINOv3 model and extracts intermediate layer [CLS] tokens for all images in the directory.
#     If pretrained=False, loads a randomly initialized model.
#     Returns:
#         all_ids: list of image string IDs
#         X_layers: NumPy array of shape (num_images, num_layers+1, hidden_dim)
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     print(f"Loading processor and model ({'pretrained' if pretrained else 'random'})...")
#     processor = AutoImageProcessor.from_pretrained(model_id)

#     if pretrained:
#         model = AutoModel.from_pretrained(model_id)
#     else:
#         config = AutoConfig.from_pretrained(model_id)
#         model = AutoModel.from_config(config)

#     model = model.to(device)
#     model.eval()

#     png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
#     print(f"Found {len(png_files)} PNG files in {png_dir}. Starting extraction...")

#     # tbd
#     # from torch import nn
#     # dino_dim = 768
#     # pixel_features = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, dino_dim)).to(device)
#     # rand_relu = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, dino_dim), nn.BatchNorm1D(dino_dim), nn.GeLU()).to(device)
#     # all_pixel, all_rand = [], []

#     all_ids = []
#     all_layer_cls = []

#     with torch.no_grad():
#         for filename in tqdm(png_files):
#             image_id = filename.replace(".png", "")
#             img_path = os.path.join(png_dir, filename)

#             image = Image.open(img_path).convert("RGB")
#             inputs = processor(images=image, return_tensors="pt").to(device)

#             # all_pixel.append(pixel_features(inputs).detach().cpu().numpy())
#             # all_rand.append(all_rand(inputs).detach().cpu().numpy())

#             outputs = model(**inputs, output_hidden_states=True)

#             layer_cls = []
#             for h in outputs.hidden_states:
#                 cls = h[:, 0, :].squeeze(0).cpu().numpy()
#                 layer_cls.append(cls)

#             all_layer_cls.append(np.stack(layer_cls))
#             all_ids.append(image_id)

#     X_layers = np.array(all_layer_cls)

#     print("Extraction complete.")
#     print("Number of images:", len(all_ids))
#     print("Layerwise CLS shape:", X_layers.shape)
    
#     return all_ids, X_layers

def extract_dino_features(png_dir, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", pretrained=True, device=None, batch_size=128):
    """
    Loads DINOv3 model and extracts intermediate layer [CLS] tokens for all images in the directory using batched inference.
    If pretrained=False, loads a randomly initialized model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor and model ({'pretrained' if pretrained else 'random'})...")
    processor = AutoImageProcessor.from_pretrained(model_id)

    if pretrained:
        model = AutoModel.from_pretrained(model_id)
    else:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_config(config)

    model = model.to(device)
    model.eval()

    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
    print(f"Found {len(png_files)} PNG files in {png_dir}. Starting batched extraction...")



    # Set up the DataLoader
    dataset = PNGDataset(png_dir, png_files, processor)
    # Using num_workers=4 to load images in parallel while GPU calculates
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_ids = []
    all_layer_cls = []

    with torch.no_grad():
        for batch_ids, batch_pixel_values in tqdm(dataloader, desc="Extracting Batches"):
            batch_pixel_values = batch_pixel_values.to(device)
            
            # Forward pass on the entire batch
            outputs = model(pixel_values=batch_pixel_values, output_hidden_states=True)

            batch_layer_cls = []
            for h in outputs.hidden_states:
                # h shape: (batch_size, sequence_length, hidden_dim)
                # We grab the [CLS] token (index 0) for the whole batch
                cls = h[:, 0, :].cpu().numpy()
                batch_layer_cls.append(cls)

            # Stack layers to get shape: (batch_size, num_layers, hidden_dim)
            batch_layer_cls = np.stack(batch_layer_cls, axis=1)
            
            all_layer_cls.append(batch_layer_cls)
            all_ids.extend(batch_ids)

    # Concatenate all batches together along the batch dimension
    X_layers = np.concatenate(all_layer_cls, axis=0)

    print("Extraction complete.")
    print("Number of images:", len(all_ids))
    print("Layerwise CLS shape:", X_layers.shape)
    
    return all_ids, X_layers

def extract_all_features(meta):
    """
    Given a single metadata dictionary loaded from JSON, 
    flattens all part features (e.g. face_base, eyes, mouth) into a dict. 
    """
    features = {}
    for part in meta.get("parts", []):
        name = part["id"]
        # Extract common fields
        features[name] = {k: v for k, v in part.items() if k != "id"}
        
        # Flatten coordinates
        if "center" in part:
            features[name]["x"] = part["center"][0]
            features[name]["y"] = part["center"][1]
            del features[name]["center"]

    return features

def load_metadata_features(meta_dir, image_ids):
    """
    Loops through image IDs, loads corresponding specific JSON metadata, 
    and extracts a 19-dimensional target feature vector matrix.
    Returns:
        feature_matrix: NumPy array of targets (N, 19)
        labels: List of label strings matching the columns
    """
    labels = [
        'face_base_x', 'face_base_y', 'face_base_radius', 
        'eye_left_x', 'eye_left_y', 'eye_left_radius',
        'eye_right_x', 'eye_right_y', 'eye_right_radius', 
        'mouth_x', 'mouth_y', 'mouth_width', 'mouth_curve',
        'skin_r', 'skin_g', 'skin_b',
        'eyes_r', 'eyes_g', 'eyes_b'
    ]
    print(f"Loading metadata features for {len(image_ids)} images from {meta_dir}...")
    all_features = []

    for image_id in image_ids:
        meta_path = os.path.join(meta_dir, f"{image_id}.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        f_parts = extract_all_features(meta)
        
        colors = meta.get("colors", {})
        skin_rgb = colors.get("skin", {}).get("rgb", [0, 0, 0])
        eyes_rgb = colors.get("eyes", {}).get("rgb", [0, 0, 0])
        
        row = [
            f_parts['face_base']['x'], f_parts['face_base']['y'], f_parts['face_base']['radius'],
            f_parts['eye_left']['x'],  f_parts['eye_left']['y'],  f_parts['eye_left']['radius'],
            f_parts['eye_right']['x'], f_parts['eye_right']['y'], f_parts['eye_right']['radius'],
            f_parts['mouth']['x'],     f_parts['mouth']['y'],     f_parts['mouth']['width'], f_parts['mouth']['curve'],
            skin_rgb[0], skin_rgb[1], skin_rgb[2],
            eyes_rgb[0], eyes_rgb[1], eyes_rgb[2]
        ]
        all_features.append(row)

    feature_matrix = np.array(all_features)
    print(f"Metadata feature matrix shape: {feature_matrix.shape}")
    return feature_matrix, labels

def save_dino_features(file_path, image_ids, X_layers):
    """
    Saves the extracted DINOv3 features and corresponding image IDs to a compressed .npz file.
    Example: save_dino_features("dino_pretrained_out.npz", ids, X_layers)
    """
    print(f"Saving features to {file_path}...")
    np.savez(file_path, all_ids=image_ids, X_layers=X_layers)
    print("Features saved successfully.")

def load_dino_features(file_path):
    """
    Loads saved DINOv3 features from an .npz file (e.g. dino_pretrained_out.npz)
    Returns:
        image_ids: List of image string IDs
        X_layers: NumPy array of layer features
    """
    print(f"Loading features from {file_path}...")
    loaded = np.load(file_path)
    X_layers = loaded['X_layers']
    all_ids = loaded['all_ids']
    
    # Standardize format back to list of strings
    if all_ids.ndim == 1:
        all_ids = all_ids.tolist()
        
    print(f"Loaded {len(all_ids)} images. Feature matrix shape: {X_layers.shape}")
    return all_ids, X_layers

def load_metadata_features_two_faces(meta_dir, image_ids):
    """
    Loops through image IDs, loads corresponding specific JSON metadata for TWO faces,
    and extracts a 38-dimensional target feature vector matrix (19 features per face).
    Returns:
        feature_matrix: NumPy array of targets (N, 38)
        labels: List of label strings matching the columns
    """
    labels = [
        'face_1_base_x', 'face_1_base_y', 'face_1_base_radius', 
        'face_1_eye_left_x', 'face_1_eye_left_y', 'face_1_eye_left_radius',
        'face_1_eye_right_x', 'face_1_eye_right_y', 'face_1_eye_right_radius', 
        'face_1_mouth_x', 'face_1_mouth_y', 'face_1_mouth_width', 'face_1_mouth_curve',
        'face_1_skin_r', 'face_1_skin_g', 'face_1_skin_b',
        'face_1_eyes_r', 'face_1_eyes_g', 'face_1_eyes_b',
        'face_2_base_x', 'face_2_base_y', 'face_2_base_radius', 
        'face_2_eye_left_x', 'face_2_eye_left_y', 'face_2_eye_left_radius',
        'face_2_eye_right_x', 'face_2_eye_right_y', 'face_2_eye_right_radius', 
        'face_2_mouth_x', 'face_2_mouth_y', 'face_2_mouth_width', 'face_2_mouth_curve',
        'face_2_skin_r', 'face_2_skin_g', 'face_2_skin_b',
        'face_2_eyes_r', 'face_2_eyes_g', 'face_2_eyes_b'
    ]
    print(f"Loading TWO-FACE metadata features for {len(image_ids)} images from {meta_dir}...")
    all_features = []

    for image_id in image_ids:
        meta_path = os.path.join(meta_dir, f"{image_id}.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        row = []
        for face_meta in meta["faces"]: # face_1 then face_2
            # Reuses the single part extractor
            f_parts = extract_all_features(face_meta)
            face_id = face_meta["face_id"] # "face_1" or "face_2"
            
            colors = face_meta.get("colors", {})
            skin_rgb = colors.get("skin", {}).get("rgb", [0, 0, 0])
            eyes_rgb = colors.get("eyes", {}).get("rgb", [0, 0, 0])
            
            row.extend([
                f_parts[f'{face_id}_base']['x'], f_parts[f'{face_id}_base']['y'], f_parts[f'{face_id}_base']['radius'],
                f_parts[f'{face_id}_eye_left']['x'],  f_parts[f'{face_id}_eye_left']['y'],  f_parts[f'{face_id}_eye_left']['radius'],
                f_parts[f'{face_id}_eye_right']['x'], f_parts[f'{face_id}_eye_right']['y'], f_parts[f'{face_id}_eye_right']['radius'],
                f_parts[f'{face_id}_mouth']['x'],     f_parts[f'{face_id}_mouth']['y'],     f_parts[f'{face_id}_mouth']['width'], f_parts[f'{face_id}_mouth']['curve'],
                skin_rgb[0], skin_rgb[1], skin_rgb[2],
                eyes_rgb[0], eyes_rgb[1], eyes_rgb[2]
            ])
        all_features.append(row)

    feature_matrix = np.array(all_features)
    print(f"Two-Face metadata feature matrix shape: {feature_matrix.shape}")
    return feature_matrix, labels

