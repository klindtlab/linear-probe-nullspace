# Camera-Ready Experiments

Experiments added for the camera-ready version of "Is a Linear Probe Evidence of
a Linear Representation?". They answer four questions:

1. **Is the Section 4 simulation a nontrivial linear world model, or does the
   ReLU MLP degenerate to an affine map?** Measured, not asserted: distinct ReLU
   regions on the data, gating fraction, dead units, active fraction, Jacobian
   variation, and the linear-scale eigenspectrum, across widths, weight decays,
   and seeds.
2. **Are the diagnostics precisely defined?** One definition each for D1, D2 and
   D3, fixed in `src/diagnostics.py` and used by every number, table and figure.
   The submitted code computed the "spectral concentration" panel of Figure 3 and
   the Figure 7 curve with different formulas under one name; both now use the
   Section 3.3 definition, and the superseded statistic is kept for comparison.
3. **Are the experiments reproducible?** Every metric comes from a CSV this
   pipeline writes. Nothing is hand-transcribed.
4. **How far do the conclusions generalize?** Seven encoders on SVG-World and
   three encoders on three further disentanglement datasets, each against a
   matched random initialization.

## Layout

```
src/rebuttal_config.py      every fixed choice: targets, splits, encoders, datasets
src/diagnostics.py          D1, D2, D3 definitions + split audit + paired bootstrap
src/fast_ridge.py           ridge with a cheap alpha path (matches sklearn to 1e-9)
src/simulation.py           Section 4 simulation: sweep, integrity audit, spectra
src/simulation_controls.py  audit calibration and the affine-selector control (CPU)
src/svgworld_data.py        shared latent matrix + SVG-World rendering
src/svgworld/               the SVG-World scene generators (imports made lazy)
src/svgworld/raster.py      rasterizer backend selection (cairosvg, resvg fallback)
src/render_checks.py        geometry/determinism validation of the rasterizer swap
src/encoders.py             preprocessing + pooled feature extraction for all models
src/factors.py              factor-balanced sampling, target encoding, checkerboard
src/datasets_extra.py       3D Shapes, dSprites, MPI3D-realistic loaders
src/pixel_control.py        D1 on raw pixels, the pixel-space control
src/run_lane.py             the one entrypoint, one "lane" per experiment
src/synthesize.py           joins the lanes' CSVs into the aggregated tables
src/aggregate_svgworld.py   SVG-World tables
src/refresh_dataset_numbers.py  cross-dataset aggregates and the pixel-vs-gap table
src/job_prelude.sh          job-time dependency setup for cluster jobs
tests/                      toy-fixture tests for the ridge path and the diagnostics
results/                    metrics tables
figures/                    one directory per figure: plot.py + small data + render
```

## Fixed definitions

| Choice | Value |
| --- | --- |
| Headline target | the 12 position latents; 20 active and all 32 stored dims are appendix sensitivities |
| Split | five repeated 50/50 splits over split units (scene ids, or source factor-combination hashes); paired styles share the split |
| Ridge alpha | 5-fold CV inside the training half only, over 1e-3 to 1e5 |
| D1 | held-out reverse R^2, reported raw, with the top-r eigenvalue-share (Ky Fan) ceiling and the ratio; r = effective rank of the target block on train |
| D2 | S(k) = reverse R^2 after PCA and whitening fitted on train only; scalar is S(r_eff) |
| D3 | SVG-World: cross-style transfer, both directions, features centered per style, reported as Pearson rho (transfer R^2 is uninterpretable under per-style scale mismatch). Added datasets: held-out checkerboard over the two highest-cardinality varying factors |
| Cyclic factors | encoded as sin/cos over the period implied by each dataset's own stored angle grid, verified in the loader against that grid. dSprites orientation: closed grid of 40 angles spanning [0, 2pi] inclusive, so period 39 and class 39 (a duplicate of class 0) excluded. 3D Shapes hues: open grid of 10, period 10. MPI3D both axes: ordinal, not cyclic |
| Intervals | paired bootstrap over held-out units, 10,000 resamples, probe held fixed. Applies to forward, D1 and D2 only; cross-style rho uncertainty is the spread over five splits and two directions |
| Random control | identical architecture and config, seed 42 |

## How to reproduce

```bash
cd camera_ready/rebuttal
export PYTHONPATH="$PWD/src"

# end-to-end smoke test (all encoders, all datasets, tiny)
python src/run_lane.py --lane smoke --n 96

# full runs, one GPU each, independent
python src/run_lane.py --lane simulation
python src/run_lane.py --lane svgworld            # 7 encoders x {pre,rand} x 2 styles
python src/run_lane.py --lane shapes3d            # 3 encoders x {pre,rand}
python src/run_lane.py --lane dsprites
python src/run_lane.py --lane mpi3d_realistic
python src/run_lane.py --lane pixel_control

# tables and figure data (CPU), once the lane CSVs are in results/
python src/synthesize.py --results results --out results
python figures/build_figure_data.py --results results

# tests (CPU)
python -m pytest tests -q
```

Each lane writes to `$ARTIFACTS_DIR/<lane>/` (default `/tmp/rebuttal_artifacts`).
On a SLURM cluster, `source src/job_prelude.sh` at the top of each job installs
the rendering dependencies.

**Estimator convention.** Every SVG-World and added-dataset value is measured on
the shared 50/50 split that the paired bootstrap runs on, so a point estimate and
its interval describe the same fit. Simulation values are means over five seeds.
The five-split mean differs from the shared split by at most 0.0025 on D1.

## Environment notes

- Tested with torch 2.11.0 (CUDA 12.8), transformers 5.12.1, scikit-learn 1.9.0,
  numpy 2.5.0 and pandas 3.0.3; `pyproject.toml` lists the remaining packages.
- **Rasterizer**: when `libcairo` is not available, rendering uses `resvg_py`
  instead of `cairosvg`. `src/render_checks.py` validates that the replacement
  reproduces the scene geometry the generators encode.
- **Preprocessing**: `AutoImageProcessor` in transformers 5.12.1 requires
  torchvision on both its fast and slow paths, so preprocessing is implemented
  directly from each model's pinned `preprocessor_config.json`, and the
  device-side path is checked against a PIL reference in the smoke run.

## Simulation Controls

`src/simulation_controls.py` (CPU, ~2 min, 5 seeds at width 1024) writes
`results/simulation/simulation_controls.csv` (per-seed rows) and
`results/simulation/simulation_controls.json` (aggregated summary):

- **Audit calibration.** Forcing every pre-activation positive makes the network
  affine on its data. The audit then reports exactly 1 linear region, active
  fraction 1.000, and Jacobian CV 2.36e-15 in all 5 seeds, so its null result ("no
  seed produced a single region") is falsifiable.
- **Affine-selector counterexample.** A high reverse R^2 from a lift only
  0.2452-linear in the latents does NOT imply the network is nonaffine in x: by the
  paper's own Proposition 1 an affine map can select the latent-decodable subspace.
  Projecting x onto the top-2 directions of its ridge-fitted latent-linear component
  is affine in x by construction and reaches reverse R^2 0.8992, against 0.9936 for
  the trained network in the same run. Nondegeneracy therefore rests on the
  activation statistics alone.

## Results

Large outputs (rendered images, extracted features) are not included. `results/`
holds the small tables:

| experiment | directory | what it holds |
| --- | --- | --- |
| Section 4 simulation | `results/simulation/` | nondegeneracy audit by width, top-20 eigenvalues, D2 curves, and the two simulation controls |
| SVG-World panel | `results/svgworld_pil/` and `results/svgworld_*.csv` | 7 encoders x {pretrained, random} x 2 styles, D1/D2/D3, intervals |
| 3D Shapes | `results/shapes3d_pil/` | 3 encoders x {pretrained, random}, 20,000 samples |
| dSprites | `results/dsprites_pil3/` | orientation period 39 with the duplicated bin 39 excluded |
| MPI3D-realistic | `results/mpi3d_realistic_pil2/` | `horizontal_axis` encoded as ordinal |
| pixel-space control | `results/pixel_control/`, `results/pixel_vs_gap.csv` | D1 on raw 32x32x3 images for all four datasets |
| resize-backend sensitivity | `results/sensitivity_device_backend/`, `results/backend_robustness.csv` | earlier device-resize runs, kept as a sensitivity analysis |

`results/backend_robustness.csv` compares the two resize backends on 3D Shapes, the
one dataset whose target encoding did not also change: 9 statistics, no sign
changes, largest absolute shift 0.0016. The dSprites and MPI3D backend rows were
dropped because their encodings were corrected after those runs, so a comparison
would confound two changes; backend robustness of those two results under the
corrected encodings is therefore unmeasured. `results/preprocessing_reconciliation.json`
checks every encoder's coded preprocessing against its pinned config.

### Provenance

| item | value |
| --- | --- |
| encoders | 7, pinned by revision in `src/rebuttal_config.py` |
| datasets | SVG-World (rendered in the job), 3D Shapes, dSprites, MPI3D-realistic, all from official sources |
| image scale | 20,000 SVG-World scenes (10,000 latent vectors x 2 styles), 20,000 samples per added dataset |
| precision | fp32; bf16 autocast measured and rejected (relative feature difference 9.066e-3 against a 5e-3 threshold) |
| resize path | PIL reference, per each model's pinned `preprocessor_config.json`; the device path shifted D1 by up to 0.1059 on 64-pixel sources |
| batch size | 256, from warmed throughput on one H100, peak memory below 4.0 GB |
| rasterizer | `resvg_py` 0.3.3; geometry validated in `src/render_checks.py` |
| seeds | data seed 42, random-init seed 42, split seeds 0 to 4, sampling seed 1234, bootstrap seed 7 |
