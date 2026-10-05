# Independent Audit

An independent recomputation of the camera-ready evidence for "Is a Linear Probe
Evidence of a Linear Representation?". The question is which claims survive when
the diagnostics are reimplemented from the manuscript's equations, the submitted
recipe is walked to the revised one one factor at a time, and the cross-dataset
result is rerun across model initializations, dataset samples and target
encodings. The verdict is in `results/audit_verdict.md`.

The code under audit is `../rebuttal/src/`. It is vendored unchanged under
`src/prior/` so its data loaders, renderer and encoders are the same objects; its
`diagnostics.py` is used only as a second opinion in the identical-array parity
check.

## Layout

```
src/audit_calc.py       D1, D2, D3 written from Eq. (3), Prop. 2 and Sec 3.3,
                        plus C(k), the Ky Fan ceiling, and the hierarchical
                        bootstrap. Imports nothing from src/prior/.
src/legacy_calc.py      the submitted recipe, transcribed from
                        svg_world/4_reverse_predictivity.py
src/audit_encode.py     several model instances per preprocessing pass, and
                        BatchNorm calibration for the random ResNet control
src/run_audit.py        the one entrypoint: --stage smoke|svgworld|dataset
src/sim_definitions.py  S(k) against C(k) on the Section 4 simulation (CPU)
src/aggregate.py        reads the stage CSVs and writes the audit tables
src/prior/              ../rebuttal/src, vendored unchanged
src/job_prelude.sh      job-time dependency setup for cluster jobs
tests/test_oracles.py   oracle tests (code) and validity tests (definitions)
results/                audit tables and verdict
figures/                one directory per figure: plot.py + data + render
```

## How to reproduce

```bash
cd camera_ready/audit
export PYTHONPATH="$PWD/src:$PWD/src/prior"

# smoke test, the production entrypoint at tiny size
python src/run_audit.py --stage smoke --tag smoke

# full runs, one GPU each
python src/run_audit.py --stage svgworld --tag prod --n 10000 --full-d2
python src/run_audit.py --stage dataset --tag prod --dataset shapes3d --all-encodings
python src/run_audit.py --stage dataset --tag prod --dataset dsprites --all-encodings
python src/run_audit.py --stage dataset --tag prod --dataset mpi3d_realistic --all-encodings

# definition validity and estimator oracles (CPU)
python -m pytest tests -q
python src/sim_definitions.py --seeds 5
```

Each stage writes to `$ARTIFACTS_DIR/<tag>/<stage>/` (default `/tmp/rebuttal_artifacts`).
On a SLURM cluster, `source src/job_prelude.sh` at the top of each job installs
the rendering dependencies.

## Audit axes

| axis | submitted | revised | what the audit adds |
| --- | --- | --- | --- |
| D1 estimator | residual variance, float32, alpha 1, one row split seed 42, 32 target columns | held-out SSE, float64, CV alpha, five unit splits, 12 position latents | each factor's contribution measured alone and cumulatively |
| implementation | one implementation | one implementation | a second implementation written from the equations, compared on identical arrays |
| rasterizer | cairosvg | resvg | identical SVG strings rasterized by a third engine, pixel and D1 agreement |
| random control | one initialization | one initialization, uncalibrated ResNet BatchNorm | five initializations, plus BatchNorm calibrated on the evaluation images |
| dataset sample | one sample seed | one sample seed | three independent factor-balanced samples |
| targets | one encoding | one encoding | canonical, plain ordinal, and canonical plus one-hot |
| uncertainty | none | paired bootstrap over held-out units | hierarchical bootstrap over units and control initializations |
| D2 | S(k), inconsistently implemented | S(k), one definition | S(k) against the monotone C(k), with the invariances tested |

## Findings

| axis | verdict |
| --- | --- |
| implementation | two independent implementations agree to 5.551e-16 on identical arrays (8 conditions) |
| submitted numbers | reproduce within 0.007; no recipe factor moves reverse predictivity by more than 0.0037, the full walk by more than 0.0042 |
| definitions | three printed statements corrected: the Definition 1 residual condition is not scale-invariant, Proposition 2 needs a pseudoinverse at rank-deficient latent covariance, and the spectral-concentration curve does not approach 1 from the latent rank onward |
| spectral concentration | neither the printed statistic (0.7127 random against 0.9998 trained) nor the monotone alternative (0.8517 against 1.0000) discriminates in the Section 4 simulation |
| SVG-World | holds on every axis: five initializations, all intervals excluding zero, renderer sensitivity 0.0077 against a gap of 0.19 |
| added benchmarks | reproduce to four decimals; robust to initialization and dataset sample (0.0204), but 2 of 6 transformer encoder-benchmark pairs change sign across defensible target encodings |
| mechanism | random controls are near low-rank (participation ratio 2.31 and 3.75 against 16.57 and 8.69) while forward predictivity is tied |

## Results

Large outputs (extracted features) are not included. `results/` holds the tables:

| item | location | contents |
| --- | --- | --- |
| verdict | `results/audit_verdict.md` | claim-by-claim status with the evidence for each |
| SVG-World | `results/prod/svgworld/` | recipe ladder, implementation parity, five-seed D1/D2/D3 panel, hierarchical intervals |
| added datasets | `results/prod3/<dataset>/`, `results/prodmin/<dataset>/` | per-instance metrics and hierarchical intervals |
| simulation definitions | `results/sim_definitions.csv` | S(k) and C(k) on the Section 4 simulation, five seeds |
| simulation statistics | `results/prior_source/` | five-seed activation statistics from `../rebuttal/results/simulation/` |

Provenance: seven pinned encoders (see `src/prior/rebuttal_config.py`); random
initialization seeds 42 to 46; dataset sample seeds 1234, 7 and 2024; split seeds
0 to 4; bootstrap seed 7; fp32; PIL reference resize path; resvg 0.3.3 rasterizer.
