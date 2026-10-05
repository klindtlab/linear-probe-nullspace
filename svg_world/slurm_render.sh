#!/bin/bash
#SBATCH --job-name=sw_render
#SBATCH --output=logs/render_%j.out
#SBATCH --error=logs/render_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# Environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Sanity check
python -c "import cairosvg, numpy, tqdm; print('cairosvg ok')"

# svg_island.py and svg_western.py must be importable; either run from a
# directory that contains them, or add their location to PYTHONPATH:
# export PYTHONPATH="$PWD/path/to/svg_modules:${PYTHONPATH:-}"

mkdir -p logs

python 1_render_dataset.py \
    --n_total 10000 \
    --seed 42 \
    --output_size 224 \
    --out_root scene_world

echo "render done"
