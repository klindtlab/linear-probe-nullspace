#!/usr/bin/env bash
# Job-time dependency setup, sourced at the top of each cluster job.
#
# Installs the few packages the render path needs on top of an existing
# torch/transformers environment, into a job-local directory, pinned and with
# --no-deps so the environment's torch/numpy stack is left untouched:
#
#   resvg-py  self-contained SVG rasterizer (used when libcairo is unavailable)
#   pillow    PNG encode/decode and the reference resize path
#   h5py      3D Shapes ships as a single HDF5 file
#
# Plotting stays out of the job: figures are rendered afterwards from the CSVs.
set -euo pipefail

export UV_OFFLINE=0
# Dependencies and caches go to job-local scratch.
JOB_SCRATCH="${TMPDIR:-/tmp}/job-$$"
JOB_DEPS="$JOB_SCRATCH/job-deps"
export UV_CACHE_DIR="$JOB_SCRATCH/uv-cache"
mkdir -p "$JOB_DEPS" "$UV_CACHE_DIR"
uv pip install --quiet --target "$JOB_DEPS" --no-deps \
    "resvg-py==0.3.3" "pillow==12.3.0" "h5py==3.16.0"
export PYTHONPATH="$JOB_DEPS:$PWD/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$JOB_SCRATCH/mpl"
echo "[prelude] job deps -> $JOB_DEPS"
python -c "import resvg_py, PIL, h5py; print('[prelude] resvg_py', '+ PIL', PIL.__version__, '+ h5py', h5py.__version__)"
