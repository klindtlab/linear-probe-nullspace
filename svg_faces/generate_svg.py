import os
import json
import random
import colorsys
import cairosvg

CANVAS_W = 224 # Dinov3 input size
CANVAS_H = 224 # Dinov3 input size

def get_random_hsv():
    """Returns a tuple of (h, s, v) and the hex string for internal use."""
    h = random.random()
    s = random.uniform(0.5, 1.0) # Vibrant range
    v = random.uniform(0.5, 1.0) 
    
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

def generate_random_face(cx, cy, face_radius, face_id):
    """Generates the SVG group and metadata for a single face."""
    eye_radius = random.randint(3, 7)
    eye_spacing = random.randint(10, 20)
    eye_y_offset = random.randint(5, 15)

    mouth_width = random.randint(15, 30)
    mouth_y_offset = random.randint(10, 20)
    mouth_curve = random.randint(-15, 15) 

    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset

    skin = get_random_hsv()
    eye = get_random_hsv()
    stroke_color = "#111111"

    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)

    svg_group = f'''  <g id="{face_id}">
    <circle id="{face_id}_base" cx="{cx}" cy="{cy}" r="{face_radius}" fill="{skin['hex']}" stroke="{stroke_color}" stroke-width="2" />
    <circle id="{face_id}_eye_left" cx="{left_eye_cx}" cy="{left_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <circle id="{face_id}_eye_right" cx="{right_eye_cx}" cy="{right_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <path id="{face_id}_mouth" d="{mouth_path}" fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" />
  </g>'''

    metadata = {
        "face_id": face_id,
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
            {"id": f"{face_id}_base", "type": "circle", "center": [cx, cy], "radius": face_radius},
            {"id": f"{face_id}_eye_left", "type": "circle", "center": [left_eye_cx, left_eye_cy], "radius": eye_radius},
            {"id": f"{face_id}_eye_right", "type": "circle", "center": [right_eye_cx, right_eye_cy], "radius": eye_radius},
            {"id": f"{face_id}_mouth", "type": "path", "center": [mouth_cx, mouth_cy], "width": mouth_width, "curve": mouth_curve}
        ]
    }
    return svg_group, metadata

def generate_two_faces_svg(sample_id):
    faces_metadata = []
    svg_groups = []

    
    while True:
        r1 = random.randint(25, 45)
        cx1 = random.randint(r1 + 10, CANVAS_W - r1 - 10)
        cy1 = random.randint(r1 + 10, CANVAS_H - r1 - 10)
        
        r2 = random.randint(25, 45)
        cx2 = random.randint(r2 + 10, CANVAS_W - r2 - 10)
        cy2 = random.randint(r2 + 10, CANVAS_H - r2 - 10)
        
        # Check for overlap (distance between centers vs sum of radii + padding)
        dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
        min_dist = r1 + r2 + 5 # 5 pixel padding
        if dist_sq >= min_dist**2:
            break

    # Once we have valid positions, generate the SVGs
    svg1, meta1 = generate_random_face(cx1, cy1, r1, "face_1")
    faces_metadata.append(meta1)
    svg_groups.append(svg1)
    
    svg2, meta2 = generate_random_face(cx2, cy2, r2, "face_2")
    faces_metadata.append(meta2)
    svg_groups.append(svg2)

    # --- SVG Generation ---
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <rect width="100%" height="100%" fill="white" />
{svg_groups[0]}
{svg_groups[1]}
</svg>'''

    # --- Metadata ---
    metadata = {
        "id": sample_id,
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "faces": faces_metadata
    }

    return svg_content, metadata

def convert_svgs_to_pngs(svg_dir, png_dir):
    print(f"Starting conversion: SVG -> PNG ({png_dir})")
    os.makedirs(png_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(svg_dir):
        if filename.endswith(".svg"):
            svg_path = os.path.join(svg_dir, filename)
            png_name = filename.replace(".svg", ".png")
            png_path = os.path.join(png_dir, png_name)

            # Skip conversion if PNG already exists to allow resuming
            if not os.path.exists(png_path):
                try:
                    cairosvg.svg2png(
                        url=svg_path,
                        write_to=png_path,
                        output_width=224,
                        output_height=224
                    )
                    count += 1
                    if count % 1000 == 0:
                        print(f"Converted {count} new images...")
                except Exception as e:
                    print(f"WARNING: Skipping corrupted SVG file {filename}")

    print(f"Successfully converted remaining files! New converted: {count}")
