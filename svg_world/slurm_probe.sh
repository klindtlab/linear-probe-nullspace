#!/bin/bash
#SBATCH --job-name=sw_probe
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs

echo ""
echo "######################################################################"
echo "# CLS, within-domain"
echo "######################################################################"
python 3_linear_probe.py --features cls

echo ""
echo "######################################################################"
echo "# CLS, within + cross-domain"
echo "######################################################################"
python 3_linear_probe.py --features cls --cross

echo ""
echo "######################################################################"
echo "# Mean-pooled patches, within + cross-domain"
echo "######################################################################"
python 3_linear_probe.py --features patches_mean --cross

# Patches_concat is heavier; uncomment when ready.
# echo ""
# echo "######################################################################"
# echo "# Concatenated patches with PCA(1024), within + cross-domain"
# echo "######################################################################"
# python 3_linear_probe.py --features patches_concat --n_pca 1024 --cross

echo "probe done"
