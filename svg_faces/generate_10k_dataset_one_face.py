import os
import random
import colorsys
import json
from generate_svg import convert_svgs_to_pngs
import numpy as np
import cairosvg

# --- PATH SETUP ---
BASE_DIR = "svg_face_dataset_one_face"
SVG_DIR = os.path.join(BASE_DIR, "svgs")
META_DIR = os.path.join(BASE_DIR, "meta")
PNG_DIR = os.path.join(BASE_DIR, "pngs")

os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

CANVAS_W = 224
CANVAS_H = 224

def get_random_hsv(z):
    """Returns a tuple of (h, s, v) and the hex string for internal use."""
    h = z[0] #np.random.uniform(0, 1)
    s = 0.5 + z[1] * 0.5 # Vibrant range
    v = 0.5 + z[2] * 0.5 # Vibrant range
    
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    hex_val = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    
    return {
        "hsv": [h, s, v],
        "rgb": [r, g, b],
        "hex": hex_val
    }

def make_mouth_path(cx, cy, width, curve):
    """Creates a Quadratic Bezier path for the mouth based on curve value."""
    x1 = cx - width / 2
    x2 = cx + width / 2
    control_y = cy + curve
    # M = Move to start, Q = Quadratic Bezier (control point, end point)
    return f"M {x1:.2f} {cy:.2f} Q {cx:.2f} {control_y:.2f} {x2:.2f} {cy:.2f}"

def generate_face_svg(sample_id, seed=None):
    if seed is not None:
        np.random.seed(seed)

    z = np.random.uniform(0, 1, 15)

    # --- Geometry ---
    # face_radius = np.random.uniform(55, 75)
    start, end = 55, 75
    face_radius = start + z[0] * (end - start)
    # cx = np.random.uniform(face_radius + 10, CANVAS_W - face_radius - 10)
    start, end = face_radius + 10, CANVAS_W - face_radius - 10
    cx = start + z[1] * (end - start)
    # cy = np.random.uniform(face_radius + 10, CANVAS_H - face_radius - 10)
    start, end = face_radius + 10, CANVAS_H - face_radius - 10
    cy = start + z[2] * (end - start)

    # eye_radius = np.random.uniform(4, 10)
    start, end = 4, 10
    eye_radius = start + z[3] * (end - start)
    
    # eye_spacing = np.random.uniform(15, 30)
    start, end = 15, 30
    eye_spacing = start + z[4] * (end - start)
    
    # eye_y_offset = np.random.uniform(10, 25)
    start, end = 10, 25
    eye_y_offset = start + z[5] * (end - start)

    # mouth_width = np.random.uniform(25, 50)
    start, end = 25, 50
    mouth_width = start + z[6] * (end - start)
    
    # mouth_y_offset = np.random.uniform(15, 30)
    start, end = 15, 30
    mouth_y_offset = start + z[7] * (end - start)

    # mouth_curve = np.random.uniform(-20, 20)
    start, end = -20, 20
    mouth_curve = start + z[8] * (end - start)

    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset

    # --- Independent Colors for Interpretability ---
    skin = get_random_hsv(z[9:12])
    eye = get_random_hsv(z[12:15])
    stroke_color = "#111111"

    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)
    
    # z is a vector of 15 numbers here
    # we have to save it
    # z = np.array([
    #     face_radius,
    #     cx,
    #     cy,
    #     eye_radius,
    #     eye_spacing,
    #     eye_y_offset,
    #     mouth_width,
    #     mouth_y_offset,
    #     mouth_curve,
    #     # colors
    #     skin['hsv'][0],
    #     skin['hsv'][1],
    #     skin['hsv'][2],
    #     eye['hsv'][0],
    #     eye['hsv'][1],
    #     eye['hsv'][2]
    # ])
    # make sure that it is vecotr with 15 numbers here

    # --- SVG Generation ---
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <rect width="100%" height="100%" fill="white" />
  <g id="face">
    <circle id="face_base" cx="{cx}" cy="{cy}" r="{face_radius}" fill="{skin['hex']}" stroke="{stroke_color}" stroke-width="2" />
    <circle id="eye_left" cx="{left_eye_cx}" cy="{left_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <circle id="eye_right" cx="{right_eye_cx}" cy="{right_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <path id="mouth" d="{mouth_path}" fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" />
  </g>
</svg>'''

    # --- Metadata (Probing Target) ---
    metadata = {
        "id": sample_id,
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "colors": {
            "skin": {
                "hsv": skin["hsv"],
                "rgb": skin["rgb"]
            },
            "eyes": {
                "hsv": eye["hsv"],
                "rgb": eye["rgb"]
            }
        },
        "parts": [
            {"id": "face_base", "type": "circle", "center": [cx, cy], "radius": face_radius},
            {"id": "eye_left", "type": "circle", "center": [left_eye_cx, left_eye_cy], "radius": eye_radius},
            {"id": "eye_right", "type": "circle", "center": [right_eye_cx, right_eye_cy], "radius": eye_radius},
            {"id": "mouth", "type": "path", "center": [mouth_cx, mouth_cy], "width": mouth_width, "curve": mouth_curve}
        ]
    }

    return svg, z, metadata

if __name__ == "__main__":
    # --- EXECUTION ---
    n_total = 10000
    np.random.seed(42)
    seeds = np.random.choice(10 * n_total, n_total, replace=False)

    Z = np.zeros((n_total, 15))

    for i in range(n_total):
        sample_id = f"single_face_{i:05d}"
        svg_content, z, meta_data = generate_face_svg(sample_id, seeds[i])
        Z[i] = z

        # File paths
        svg_path = os.path.join(SVG_DIR, f"{sample_id}.svg")
        meta_path = os.path.join(META_DIR, f"{sample_id}.json")

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    np.save("Z_10k_one_face.npy", Z)

    print(f"Successfully generated/overwrote {n_total} files in {BASE_DIR}")

    # Run PNG conversion over all generated SVGs
    convert_svgs_to_pngs(SVG_DIR, PNG_DIR)

    # # test if the z is vector with 15 numbers here

    # svg_content, z, metadata = generate_face_svg("test_face_0", seed=seeds[0])
    # print(z.shape)
    # print(z)
    # print(metadata)
    # print(svg_content)



