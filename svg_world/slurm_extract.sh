#!/bin/bash
#SBATCH --job-name=sw_extract
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Environment
module load cuda12.3/toolkit/12.3.2
eval "$(conda shell.bash hook)"
conda activate pytorch

# Sanity check
python -c "
import torch, transformers
print('cuda available :', torch.cuda.is_available())
print('cuda device    :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('torch version  :', torch.__version__)
print('transformers   :', transformers.__version__)
"

mkdir -p logs

# Four extraction calls. Each loads its own model from scratch but they're
# fast (~6 min each at batch_size=128 on a V100; faster on A100/H100).
for theme in island western; do
    for variant_flag in --pretrained --random; do
        echo "=== ${theme} ${variant_flag} ==="
        python 2_extract_features.py \
            --theme  "${theme}" \
            ${variant_flag} \
            --root   scene_world \
            --batch_size 128 \
            --num_workers 4 \
            --seed 42
    done
done

echo "extract done"
