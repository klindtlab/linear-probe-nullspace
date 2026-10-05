# CelebA diagnostics

Reproduces the CelebA results of Sec. 9 of *Is a Linear Probe Evidence of a
Linear Representation?* (NeurIPS 2026): DINOv3-B/16 and CLIP-B/16, pretrained vs.
five random initializations each, with the 40 binary CelebA attributes as the
factor set.

## Files

| File | Purpose |
|---|---|
| `celeba_diagnostics.py` | feature extraction and all diagnostics (forward R², reverse R², ceiling, SC, PR, bootstrap) |
| `prepare_celeba.py` | lays out the Kaggle CelebA download as the loader expects |
| `run_all.sh` | self-test → run → checks → figure, in one command |
| `celeba_report.py` | checks a run; prints the table rows and summary numbers |
| `make_fig_celeba.py` | the figure, read from the results CSV |
| `reference/results_celeba.csv` | our results, for comparison |

## 1. Environment

Python 3.10, one GPU. The full run takes about 20-30 min on an H100.

```bash
pip install torch transformers open_clip_torch scikit-learn pandas matplotlib tqdm pillow kaggle
```

The DINOv3 weights on Hugging Face are gated: accept the license on the model page
(`facebook/dinov3-vitb16-pretrain-lvd1689m`), then log in once with
`huggingface-cli login`. The CLIP weights download automatically through `open_clip`.

## 2. Data

CelebA from Kaggle (needs a Kaggle API token):

```bash
kaggle datasets download -d jessicali9530/celeba-dataset      # ~1.4 GB
python prepare_celeba.py celeba-dataset.zip /tmp/celeba_root  # 202599 images, 162770 train
```

Put the unzipped data on local disk (e.g. `/tmp`): writing 202,599 small files to
network storage is very slow. The script always draws the same 20,000 training
images (seed 0 over the official training split).

## 3. Run

```bash
export CELEBA_ROOT=/tmp/celeba_root
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/tmp
nohup bash run_all.sh > run_all.out 2>&1 &
tail -f run_all.out
```

Single steps:

```bash
python celeba_diagnostics.py --selftest        # synthetic check, no GPU
python celeba_diagnostics.py --celeba-root $CELEBA_ROOT --out outputs/results_celeba.csv --batch 512
python celeba_report.py outputs/results_celeba.csv --ref reference/results_celeba.csv --out outputs/report
python make_fig_celeba.py outputs/results_celeba.csv outputs/fig_celeba_results_vertical.pdf --h=3.6
```

Features are cached in `features_celeba/` (one file per encoder, weights and
seed); a rerun reuses them. Delete the folder to re-extract.

## 4. Output

Everything goes to `outputs/`.

- `report.txt` ends with `ALL CHECKS PASSED` or names the failed check. It checks
  that pretrained > random in every cell under all three feature normalizations
  (raw, standardized, PCA-whitened), that CLIP loaded with its QuickGELU
  activation, and that the results match `reference/results_celeba.csv`
  (fresh runs on other hardware agree to float precision).
- `report_table.tex`: the rows of the CelebA table.
- `results_celeba.csv`: one row per encoder × weights × seed × features ×
  normalization. Columns: `forward_r2`, `R2_rev`, `rev_lo`/`rev_hi` (95% bootstrap
  interval, 1,000 resamples of held-out images; raw rows only), `ceiling`
  (C_r with r = 40), `rev_over_ceiling`, `part_ratio`. The `pca_whitened` rows are
  the spectral concentration SC (whitening at k = r).

Expected, harmless messages: sklearn `LinAlgWarning` (ridge on near-low-rank
random-init features); `No pretrained weights loaded ... initialized randomly`
for the random CLIP controls; `OSError: [Errno 16] ... .nfs*` during DataLoader
cleanup if the working directory is on network storage.

## Notes

- The OpenAI CLIP weights were trained with QuickGELU, so the model is built as
  open_clip `ViT-B-16-quickgelu`; the script asserts the activation.
- The attributes are binary, so recoding them (0/1 → ±1, z-scoring) is affine
  and changes the reverse statistics only through the ridge penalty (≤ 1e-5;
  `--selftest` checks this). The robustness check therefore varies the feature
  normalization rather than the factor encoding.
