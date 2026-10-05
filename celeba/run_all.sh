#!/usr/bin/env bash
# CelebA diagnostics: self-test, run, checks, figure.
#   export CELEBA_ROOT=/tmp/celeba_root     # built by prepare_celeba.py
#   nohup bash run_all.sh > run_all.out 2>&1 &
set -euo pipefail
: "${CELEBA_ROOT:?set CELEBA_ROOT to the folder that contains celeba/}"
cd "$(dirname "$0")"
echo "CELEBA_ROOT=$CELEBA_ROOT"
mkdir -p outputs

echo "== 1/4 self-test (synthetic, no GPU)"
python celeba_diagnostics.py --selftest | tee outputs/selftest.log

echo "== 2/4 CelebA run"
python celeba_diagnostics.py --celeba-root "$CELEBA_ROOT" \
  --out outputs/results_celeba.csv --batch 512 2>&1 | tee outputs/run.log

echo "== 3/4 checks + numbers"
if [ -f reference/results_celeba.csv ]; then REF="--ref reference/results_celeba.csv"; else REF=""; fi
python celeba_report.py outputs/results_celeba.csv $REF --log outputs/run.log --out outputs/report \
  || echo "!! some checks failed: see outputs/report.txt"

echo "== 4/4 figure"
python make_fig_celeba.py outputs/results_celeba.csv outputs/fig_celeba_results_vertical.pdf --h=3.6
echo "done: outputs/"
