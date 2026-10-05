"""Lay out the Kaggle CelebA download the way celeba_diagnostics.py expects.

  kaggle datasets download -d jessicali9530/celeba-dataset    # ~1.4 GB
  python prepare_celeba.py celeba-dataset.zip /tmp/celeba_root

Result:
  /tmp/celeba_root/celeba/img_align_celeba/000001.jpg ...
  /tmp/celeba_root/celeba/list_attr_celeba.txt      (original text format)
  /tmp/celeba_root/celeba/list_eval_partition.txt

Then run with CELEBA_ROOT=/tmp/celeba_root. Unzip to local disk (e.g. /tmp):
writing 202,599 small files to network storage takes far longer.
"""
import csv
import os
import shutil
import sys
import zipfile

zip_path, root = sys.argv[1], sys.argv[2]
raw = os.path.join(root, "download")
dst = os.path.join(root, "celeba")
os.makedirs(raw, exist_ok=True)
os.makedirs(dst, exist_ok=True)

print("unzipping ...")
with zipfile.ZipFile(zip_path) as z:
    z.extractall(raw)

# images: the Kaggle zip nests them as img_align_celeba/img_align_celeba/*.jpg
img_dir = None
for d, _, files in os.walk(raw):
    if "000001.jpg" in files:
        img_dir = d
        break
assert img_dir, "000001.jpg not found in the zip"
target = os.path.join(dst, "img_align_celeba")
if not os.path.exists(target):
    shutil.move(img_dir, target)

# attributes: CSV (-1/+1, header image_id,...) -> original text format
rows = list(csv.reader(open(os.path.join(raw, "list_attr_celeba.csv"))))
header, data = rows[0], rows[1:]
with open(os.path.join(dst, "list_attr_celeba.txt"), "w") as f:
    f.write(f"{len(data)}\n" + " ".join(header[1:]) + "\n")
    for r in data:
        f.write(r[0] + " " + " ".join(r[1:]) + "\n")

part = list(csv.reader(open(os.path.join(raw, "list_eval_partition.csv"))))[1:]
with open(os.path.join(dst, "list_eval_partition.txt"), "w") as f:
    for r in part:
        f.write(f"{r[0]} {r[1]}\n")

n_img = len(os.listdir(target))
n_train = sum(r[1] == "0" for r in part)
print(f"{n_img} images, {len(data)} attribute rows, {n_train} train images")
assert (n_img, len(data), n_train) == (202599, 202599, 162770), "unexpected counts"
print(f"ready: export CELEBA_ROOT={root}")
