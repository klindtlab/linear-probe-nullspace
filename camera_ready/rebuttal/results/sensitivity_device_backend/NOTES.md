# Device-Backend Sensitivity Analysis (Not The Reported Numbers)

These are the pretrained-minus-random intervals from the first run of the three
added-dataset lanes, which preprocessed images with the device-side (torch)
resize rather than the PIL reference path. They are retained because the smoke
measured that the two backends disagree materially on 64-pixel source images
(D1 shifted by up to 0.1059 on ResNet-50), so this is the recorded sensitivity
analysis for that choice, not a set of results.

The reported numbers come from the `*_pil` lanes, which use the PIL reference
path described in each model's pinned `preprocessor_config.json`.

Provenance caveat: only `intervals.csv` was copied out of the artifact store
before the re-smoke overwrote the untagged lane directories (the smoke wrote its
96-sample dataset lanes to the same paths). That defect is fixed in
`src/run_lane.py`: the smoke now tags every sub-lane, so it can never overwrite a
production lane again. The corresponding per-split tables for this backend were
lost to that overwrite and are not recoverable without a rerun, which is why this
directory holds the aggregate intervals only.


## Scope Narrowed After The Encoding Corrections

The backend comparison is only interpretable where the target encoding did not also
change. dSprites (orientation period 39, duplicated bin excluded) and MPI3D
(horizontal_axis ordinal) were both rerun with corrected encodings after these
device-backend runs, so their device-versus-PIL rows would confound two changes at
once and have been dropped from `results/backend_robustness.csv`. That table now
covers 3D Shapes only, whose encoding was unchanged: 9 statistics, 0 sign changes,
largest absolute shift 0.0015. The retained interval files here are kept for the
record but should not be read as current results.
