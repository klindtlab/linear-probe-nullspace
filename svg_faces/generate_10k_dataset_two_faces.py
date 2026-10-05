import os
import json
from generate_svg import generate_two_faces_svg, convert_svgs_to_pngs

# --- PATH SETUP ---
BASE_DIR = "svg_face_dataset_two_face"
SVG_DIR = os.path.join(BASE_DIR, "svgs")
META_DIR = os.path.join(BASE_DIR, "meta")
PNG_DIR = os.path.join(BASE_DIR, "pngs")

os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def build_dataset(n_total=10000):
    print(f"Generating {n_total} SVGs locally inside {SVG_DIR}...")
    
    for i in range(n_total):
        sample_id = f"two_faces_{i:05d}" # format as 00000 to accomodate 10k
        svg_content, meta_data = generate_two_faces_svg(sample_id)

        # File paths
        svg_path = os.path.join(SVG_DIR, f"{sample_id}.svg")
        meta_path = os.path.join(META_DIR, f"{sample_id}.json")

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n_total} faces...")

    print("Success! Dataset generated.")

if __name__ == "__main__":
    # Generate the dataset of SVG:
    build_dataset(10000)
    
    # Run PNG conversion over all generated SVGs
    convert_svgs_to_pngs(SVG_DIR, PNG_DIR)
