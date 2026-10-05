# linear-probe-nullspace

Code for *Is a Linear Probe Evidence of a Linear Representation?* (NeurIPS 2026).

A high linear-probe score is consistent with two very different geometries: a
linear world model, where the latent-aligned subspace dominates the
representation, or a kernel machine, where the latents lie linearly inside a much
larger curved manifold. This repository contains the code for the three
diagnostics that tell them apart (reverse predictivity, spectral concentration,
out-of-distribution generalization), the simulation, the SVG-World benchmark, and
the experiments on pretrained vision encoders.

## Repository Structure

```
linear-probe-nullspace/
├── svg_world/          SVG-World benchmark: AquaWorld and WestWorld renderers,
│                       rendering, DINOv3 feature extraction, forward and reverse
│                       probes, paper figure scripts, SLURM scripts
├── notebooks/
│   ├── 0_simulation.ipynb     the fully observed simulation
│   └── 1_dino_dsprites.ipynb  DINOv3 linear probe on a dSprites trajectory (Fig. 1)
├── svg_faces/          SVG-Faces, the single-object precursor benchmark (appendix)
├── celeba/             CelebA: DINOv3 and CLIP against random initializations
└── camera_ready/
    ├── rebuttal/       camera-ready experiments: seven encoders on SVG-World,
    │                   3D Shapes, dSprites, MPI3D-realistic, simulation controls
    └── audit/          independent reimplementation and audit of those results
```

## Getting Started

Each part has its own instructions:

| Part | Start here |
| --- | --- |
| SVG-World | `svg_world/README.md` |
| Simulation and Fig. 1 | the notebooks in `notebooks/` (they were run on Colab; adjust the data paths in the first cells) |
| SVG-Faces | run the scripts from inside `svg_faces/`: generate the dataset, extract features, then probe |
| CelebA | `celeba/README.md` |
| Camera-ready experiments | `camera_ready/rebuttal/README.md` |
| Audit | `camera_ready/audit/README.md` |

The DINOv3 weights on Hugging Face are gated: accept the license for
`facebook/dinov3-vitb16-pretrain-lvd1689m` and run `huggingface-cli login` before
extracting features.

## License

MIT, see `LICENSE`.
