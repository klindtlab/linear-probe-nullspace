# Scene-World pipeline

End-to-end DINOv3 linear-probing on the island and western Scene-World
datasets. Three stages, intended to be run as separate Slurm jobs.

## Layout

```
scene_world/
  1_render_dataset.py     stage 1: SVG -> 224x224 PNGs for both themes
  2_extract_features.py   stage 2: DINOv3 features (one call per theme/variant)
  3_linear_probe.py       stage 3: ridge probes, within and cross-domain
  slurm_render.sh         submit script for stage 1 (CPU)
  slurm_extract.sh        submit script for stage 2 (1 GPU)
  slurm_probe.sh          submit script for stage 3 (CPU)
  svg_island.py           island scene generator (AquaWorld)
  svg_western.py          western scene generator (WestWorld)
```

`svg_island.py` and `svg_western.py` are required and must be importable
from the working directory or via `PYTHONPATH`. The pipeline does not
modify them.

## Output layout

```
scene_world/
  Z.npy                       (N, 32) float32, latents shared across themes
  latent_layout.json          metadata: which dims are used vs unused
  island/
    svgs/island_NNNNN.svg     padded to 600x600 viewBox
    pngs/island_NNNNN.png     224x224
    features/
      cls_pre.npy        (N, 768)
      cls_rand.npy       (N, 768)
      patches_pre.npy    (N, 196, 768)    ~6 GB
      patches_rand.npy   (N, 196, 768)    ~6 GB
      ids_pre.npy
      ids_rand.npy
  western/                  ... same structure
```

Total disk for the default 10k-per-theme run: ~25 GB (mostly patch
tokens). If scratch is tight, comment out the `np.save(patches_path, ...)`
line in `2_extract_features.py` and rerun probes with `--features cls`
only.

## Running

Adjust paths and conda env name at the top of each `slurm_*.sh` if needed,
then submit in order:

```
sbatch slurm_render.sh    # ~10 min CPU
sbatch slurm_extract.sh   # ~30 min, 1 GPU
sbatch slurm_probe.sh     # ~5 min CPU
```

Each stage produces files the next stage reads; you can re-run stage 3
many times with different feature modes without re-extracting.

## Latent layout

32 dims total, only **20 are consumed by the renderer**:

| Dims     | Meaning                                          |
|----------|--------------------------------------------------|
| 0..3     | person 0: lane_x, lane_z, facing, skin_h         |
| 4..6     | person 0: shirt h/s/v *(unused, fixed per lane)* |
| 7..10    | person 1: lane_x, lane_z, facing, skin_h         |
| 11..13   | person 1 shirt *(unused)*                        |
| 14..17   | person 2: lane_x, lane_z, facing, skin_h         |
| 18..20   | person 2 shirt *(unused)*                        |
| 21..24   | person 3: lane_x, lane_z, facing, skin_h         |
| 25..27   | person 3 shirt *(unused)*                        |
| 28, 29   | dolphin/horse: lane_x, lane_z                    |
| 30, 31   | turtle/cow:    lane_x, lane_z                    |

`facing` is `int(z * 3.999)` inside the renderer, so its R^2 is upper-bounded
even for a perfect model. Treat it as a sanity check, not a clean continuous
target. The 12 unused dims are excluded from the default report.

## Expected sanity outcomes

With CLS-only probes, geometric latents (positions of large objects) tend
to be recoverable from pretrained features but harder from random features.
Per-person `lane_x` is the cleanest target. If pretrained CLS R^2 on
`lane_x` is below ~0.5, switch to `--features patches_mean` or
`--features patches_concat`: object positions can be hard to extract from
a single CLS pooling on multi-object scenes.

## Cross-domain probing

`--cross` fits a ridge probe on island train features, evaluates on
western test features (and the reverse). Per-domain mean-centering is
applied so we test direction transfer, not bias transfer. If pretrained
within-domain R^2 is high but cross-domain R^2 is much lower, the probe
is reading style-specific features rather than the latent itself.
