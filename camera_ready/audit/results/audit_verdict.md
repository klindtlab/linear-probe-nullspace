# Audit Verdict

Independent recomputation of the rebuttal evidence. Each row states what was
tested, what the audit found, and what the manuscript should say.
Numbers come from this experiment's tables, not from the prior run's.

Status vocabulary. **Confirmed** the claim reproduces independently.
**Corrected** the claim is wrong as stated and the correct statement is given.
**Weakened** the claim survives only in a narrower form. **Unsupported** the
evidence does not carry the claim. **Untested** the audit could not test it.

## A. Implementation And Recipe

| # | Claim under audit | Status | Evidence |
| --- | --- | --- | --- |
| A1 | The diagnostics are implemented as the manuscript defines them | Confirmed | A second implementation written from Eq. (3), Prop. 2 and Sec 3.3 agrees with the prior run's module on identical feature arrays to at most 3.331e-16 across eight conditions (D1, the Ky Fan ceiling, the D2 curve, the forward probe). `results/prod/svgworld/implementation_parity.csv` |
| A2 | The submitted numbers are reproducible | Confirmed | The submitted estimator on 10,000 scenes gives CLS D1 0.2142 / 0.1619 by style, theme mean 0.1880 against the reported 0.1812; random init 0.0190 against 0.0194; mean-patch 0.2827 against 0.2759 and 0.0205 against 0.0225. `results/prod/svgworld/recipe_ladder.csv` |
| A3 | The revised recipe changes the SVG-World conclusion | Unsupported, and no such claim is needed | Each recipe factor alone moves D1 by at most 0.0037; the whole walk from submitted to revised moves it by at most 0.0042. Estimator choice is not why any number differs. |
| A4 | Ridge alpha, dtype and split choice matter | Unsupported | Alone: float64 exactly 0.0000, CV alpha at most 0.0014, split at most 0.0025, SSE-vs-variance formula at most 0.0007. |

## B. Definitions

| # | Claim under audit | Status | Evidence |
| --- | --- | --- | --- |
| B1 | Definition 1's residual condition is well posed | Corrected | The bound sigma^2 ||W||_F^2 and D1 itself are not invariant to a rescaling of individual feature coordinates: inflating a residual-only block by 20x moves D1 by more than 0.05 while aggregate forward R^2 moves by less than 0.02. State the definition on whitened latents, or use residual variance over signal-plus-residual variance. `tests/test_oracles.py::test_validity_d1_is_not_invariant_to_coordinate_rescaling` |
| B2 | Prop. 2's Eq. (4) is usable as written | Corrected | With a rank-deficient Cov(z) the expression needs a pseudoinverse; the plain inverse is unusable (condition number above 1e12). The pseudoinverse form matches the held-out estimator to 0.01. |
| B3 | "A strict linear world model has S(k) approaching 1 from k = k_lat onward" | Corrected | False for any nonzero residual. On an exact f = Wz + mu + eps with small isotropic noise, S(3) exceeds 0.95 and S(64) falls below 0.30, because whitening rescales residual directions to unit variance. The decay the paper attributes to a kernel machine is a property of the statistic. |
| B4 | A monotone statistic fixes D2 | Weakened | C(k), the share of held-out latent-predictable feature variance inside the training top-k principal subspace, does satisfy the claimed invariances (monotone, in [0,1], C(r) = 1 for a strict linear world model, C(2) below 0.1 when the latent subspace is not leading). But on the Section 4 simulation it gives random init 0.8517 against 1.0000 trained, so it does not restore the discrimination that S(k) loses. |
| B5 | The D1/ceiling ratio is a meaningful normalization | Weakened | The Ky Fan ceiling bounds D1 but is not attainable, so the ratio normalizes against a loose bound. Report it beside raw D1, never instead of it. |
| B6 | Cross-style transfer R^2 is interpretable | Corrected | Zero-shot cross-style R^2 is -58.78 for DINOv3 B/16 and about -2e4 for random init: per-style scale mismatch dominates. Report direction-only rho (0.3808 and 0.5334 pretrained against 0.0736 and 0.0988 random, five-initialization means) and target-calibrated R^2 (0.1637 and 0.3134 against 0.0110 and 0.0175 (five-initialization means)) as separate quantities. `results/prod/svgworld/panel_d1_d2_d3.csv` |

## C. The Section 4 Simulation

| # | Claim under audit | Status | Evidence |
| --- | --- | --- | --- |
| C1 | The trained network is not degenerate | Confirmed (prior run's measurement, not re-measured here) | Not re-run in this audit; the prior run's activation statistics stand, including the forced-positive calibration that makes the null falsifiable. |
| C2 | Reverse predictivity separates trained from random | Confirmed | Width 1024, five seeds: D1 0.9935 trained (weight decay 1) against 0.2133 random. `prod/sim_definitions.csv` |
| C3 | Spectral concentration separates trained from random | Unsupported in the simulation | S(k_lat) 0.7127 random against 0.9998 trained, and the corrected C(k_lat) 0.8517 against 1.0000. Both statistics put random init close to trained. The k needed to hold 90% of latent-predictable variance separates them weakly (2 trained against 4 random in four of five seeds). |
| C4 | Forward decodability is uninformative here | Confirmed | Forward R^2 is 1.0000 in all three conditions, including random init. |

## D. SVG-World

| # | Claim under audit | Status | Evidence |
| --- | --- | --- | --- |
| D1 | Pretrained DINOv3 B/16 exceeds its matched random initialization on reverse predictivity | Confirmed, and now with initialization uncertainty | Five random seeds (42 to 46): gap 0.1867 [0.1823, 0.1912] and 0.1432 [0.1386, 0.1477] for CLS by style, 0.2492 [0.2438, 0.2545] and 0.2713 [0.2654, 0.2768] for mean-patch. All intervals exclude zero. Random-initialization reverse predictivity spans 0.0163 to 0.0298 over all seeds, styles and poolings (0.0177 to 0.0272 for CLS alone). |
| D2s | Spectral concentration separates them on real features | Confirmed | S(r) 0.2887 against 0.0321, C(r) 0.8436 against 0.5358, k90 17 against 55 to 59 (AquaWorld, CLS). Note this contradicts C3: D2 discriminates on SVG-World but not in the simulation. |
| D3s | Cross-style transfer separates them | Confirmed on direction and calibrated transfer | See B6. |
| D4 | The result is a rendering artifact | Unsupported | Sub-pixel coverage changes move D1 by +0.0077 (pretrained) and -0.0048 (random) against a gap of about 0.19. Reported as a sensitivity check, not a cairosvg reproduction. |

## E. The Added-Benchmark Ordering

| # | Claim under audit | Status | Evidence |
| --- | --- | --- | --- |
| E1 | On 3D Shapes, dSprites and MPI3D-realistic, pretrained encoders have lower reverse predictivity than their matched random initializations in 7 of 9 encoder-dataset pairs | Confirmed as a reproduction | Single-seed gaps reproduce the prior run to four decimals: DINOv3-B -0.5478 / -0.5235 / -0.2203, CLIP -0.7483 / +0.0792 / +0.0566, ResNet-50 -0.3142 / -0.3211 / -0.5674. An independent implementation gives the same numbers, so the ordering is not an implementation artifact. |
| E2 | The ordering is robust to the control initialization | Confirmed | Five initializations per encoder; hierarchical intervals over held-out units and control seeds exclude zero in every cell. DINOv3-B: -0.5593 [-0.5706, -0.5489], -0.5160 [-0.5250, -0.5053], -0.2133 [-0.2355, -0.1882]. |
| E3 | The ordering is robust to the dataset sample | Confirmed | Three independent factor-balanced samples per benchmark (seeds 1234, 7, 2024). The largest movement of any gap across samples is 0.0204. |
| E4 | The ordering is robust to the target encoding | **Refuted for a third of the transformer cells** | Across canonical, plain-ordinal and canonical-plus-one-hot encodings, 4 of 6 transformer encoder-dataset pairs keep their sign and 2 do not. DINOv3-B on 3D Shapes: -0.5598, +0.0725, -0.0960. CLIP on MPI3D-realistic: +0.0541, -0.2156, -0.4170. Adding one-hot identity columns moves CLIP on MPI3D-realistic by 0.47. These two cells are therefore recipe-dependent and are reported as such. |
| E5 | The ordering reflects the variance-share mechanism rather than representation quality | Supported | On 3D Shapes the random controls are near low-rank while forward predictivity is tied: participation ratio 2.31 (CLIP) and 3.75 (DINOv3-B) random against 16.57 and 8.69 pretrained; forward R^2 0.9574 pretrained against 0.9591 random for DINOv3-B. |
| E6 | ResNet-50 belongs in the same count as the transformers | Refuted | Its untrained control is constructed differently: BatchNorm running statistics are themselves untrained state. Under the data-free control (random weights, eval mode, untouched statistics) its gaps are -0.3161, -0.2967, -0.6611 and +0.1052 on SVG-World. A train-only warm-up pass over the training images, which is no longer a data-free control, gives -0.0673, +0.0567, +0.1028 and raises the random participation ratio from 1.23 to 5.12. Only the data-free control is reported, and ResNet-50 is excluded from every count. |

**What the audit concludes about the added benchmarks.** The prior run's numbers
are correct and reproduce exactly. What the audit adds is that the direction is
robust to initialization and to dataset sampling but not uniformly robust to how
the factor set is written down, and that the mechanism is the feature spectrum
rather than representation quality. Both findings point the same way: a
reverse-predictivity comparison is defined only against a fully specified factor
set, and cannot be read as an encoder-quality score.

## F. Limitations Of This Audit

- The renderer axis is a sub-pixel-coverage sensitivity check plus the prior
  run's geometry validation, not a reproduction of the submitted cairosvg
  pipeline: the job image has neither libcairo nor libEGL, and no
  self-contained alternative rasterizer wheel loads in it.
- The Section 4 nondegeneracy audit was not re-measured; only its diagnostics
  were recomputed. The prior run's activation statistics are taken as given.
- Encoders other than DINOv3 B/16 were not rerun on SVG-World; the prior run's
  seven-encoder panel is unaudited beyond the shared estimator check.
