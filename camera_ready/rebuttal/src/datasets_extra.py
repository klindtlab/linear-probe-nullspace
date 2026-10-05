"""Loaders for the three added disentanglement datasets.

Each loader returns the same contract:

    dict(images=uint8 (N, 64, 64, 3),
         factors=int64 (N, n_factors),   values in source index order
         source_index=int64 (N,),        flat index into the source grid
         manifest=dict)                  provenance, retained as an artifact

All three sources are the official author-attributable releases. Sampling is
deterministic and factor-balanced (see factors.sample_factor_balanced), and the
sampled source indices are written out so the exact subset is reproducible
without re-running the sampler.

MPI3D-realistic is 12.74 GB as published. Its .npz holds a single UNCOMPRESSED
zip member, so this module fetches only the selected records by HTTP range,
about 246 MB for 20,000 samples, instead of downloading the whole archive.
"""

from __future__ import annotations

import os
import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import factors as FA
import rebuttal_config as C


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: str, chunk=1 << 22) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[data] cached {dest} ({os.path.getsize(dest)} bytes)", flush=True)
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        got = 0
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            got += len(b)
            if total and got % (1 << 26) < chunk:
                print(f"[data] {os.path.basename(dest)} "
                      f"{got / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    os.replace(tmp, dest)
    print(f"[data] downloaded {dest} ({os.path.getsize(dest)} bytes, "
          f"{time.time() - t0:.1f}s)", flush=True)
    return dest


def _to_rgb_u8(imgs: np.ndarray) -> np.ndarray:
    """(N,H,W) or (N,H,W,1) -> (N,H,W,3) uint8; binary masks scaled to 0/255."""
    a = np.asarray(imgs)
    if a.ndim == 3:
        a = a[..., None]
    if a.dtype != np.uint8:
        mx = float(a.max()) if a.size else 1.0
        a = (a * (255.0 if mx <= 1.0 else 1.0)).astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.ascontiguousarray(a)


def _base_manifest(key: str, source_index: np.ndarray, F: np.ndarray) -> dict:
    spec = C.DATASETS[key]
    return dict(
        dataset=key, label=spec["label"], source=spec["source"],
        license=spec["license"], url=spec.get("url"),
        n_samples=int(len(source_index)),
        factor_names=[f["name"] for f in spec["factors"]],
        factor_n_values=[f["n_values"] for f in spec["factors"]],
        factor_kinds=[f["kind"] for f in spec["factors"]],
        sample_seed=C.SAMPLE_SEED,
        source_grid_size=int(np.prod([f["n_values"] for f in spec["factors"]])),
        marginal_balance=FA.marginal_balance_report(F, spec["factors"]),
        n_distinct_combinations=int(len(np.unique(F, axis=0))),
    )


# ---------------------------------------------------------------------------
# 3D Shapes
# ---------------------------------------------------------------------------
def load_shapes3d(n_samples: int = C.N_DATASET_SAMPLES, cache_dir: str = "/tmp/data"):
    import h5py

    spec = C.DATASETS["shapes3d"]
    n_values = [f["n_values"] for f in spec["factors"]]
    idx = FA.sample_factor_balanced(n_values, n_samples,
                                    exclude=FA.exclusions_from_spec(spec["factors"]))
    path = _download(spec["url"], os.path.join(cache_dir, "3dshapes.h5"))
    with h5py.File(path, "r") as h:
        assert h["images"].shape[0] == int(np.prod(n_values)), h["images"].shape
        images = h["images"][idx]          # fancy indexing on a sorted index list
        labels = np.asarray(h["labels"][idx])
    F = FA.index_to_factors(idx, n_values)
    # Cross-check the assumed lexicographic layout against the stored labels:
    # the label columns must be monotone in the decoded factor values.
    checks = {}
    for j, f in enumerate(spec["factors"]):
        vals = np.unique(labels[:, j])
        order = np.argsort([np.median(labels[F[:, j] == v, j])
                            for v in range(f["n_values"])
                            if (F[:, j] == v).any()])
        checks[f["name"]] = dict(
            n_unique_label_values=int(len(vals)),
            monotone_in_decoded_index=bool(np.all(np.diff(order) > 0)))
    man = _base_manifest("shapes3d", idx, F)
    man["file_bytes"] = int(os.path.getsize(path))
    man["layout_checks"] = checks
    man["label_columns_min"] = [float(v) for v in labels.min(axis=0)]
    man["label_columns_max"] = [float(v) for v in labels.max(axis=0)]
    return dict(images=_to_rgb_u8(images), factors=F, source_index=idx,
                manifest=man)


# ---------------------------------------------------------------------------
# dSprites
# ---------------------------------------------------------------------------
def load_dsprites(n_samples: int = C.N_DATASET_SAMPLES, cache_dir: str = "/tmp/data"):
    spec = C.DATASETS["dsprites"]
    n_values = [f["n_values"] for f in spec["factors"]]
    idx = FA.sample_factor_balanced(n_values, n_samples,
                                    exclude=FA.exclusions_from_spec(spec["factors"]))
    path = _download(spec["url"], os.path.join(cache_dir, "dsprites.npz"))
    with np.load(path, allow_pickle=True, encoding="latin1") as z:
        imgs = z["imgs"]
        assert imgs.shape[0] == int(np.prod(n_values)), imgs.shape
        images = imgs[idx]
        classes = np.asarray(z["latents_classes"])[idx]
        # Non-circular check of the cyclic period: read the stored angles rather
        # than re-evaluating our own encoding. dSprites column 3 is orientation.
        lv = np.asarray(z["latents_values"])
        angles = np.unique(lv[:, 3])
        lc_all = np.asarray(z["latents_classes"])
        ori_spec = [f for f in spec["factors"] if f["name"] == "orientation"][0]
        period_declared = int(ori_spec.get("period", ori_spec["n_values"]))
        spacing = float(np.diff(angles).mean())
        period_from_file = int(round(2 * np.pi / spacing))
        closed = bool(np.isclose(angles[-1] - angles[0], 2 * np.pi))
        angle_checks = dict(
            n_stored_angles=int(len(angles)),
            first_angle=float(angles[0]), last_angle=float(angles[-1]),
            spacing=spacing, period_from_file=period_from_file,
            period_declared=period_declared,
            grid_is_closed=closed,
            period_matches_file=bool(period_from_file == period_declared),
            class0_angle=float(lv[lc_all[:, 3] == 0, 3][0]),
            class_last_angle=float(lv[lc_all[:, 3] == len(angles) - 1, 3][0]),
            excluded_values=sorted(ori_spec.get("exclude_values", [])),
        )
        assert angle_checks["period_matches_file"], angle_checks
    F = FA.index_to_factors(idx, n_values)
    # latents_classes carries the source factor indices; column 0 is a constant
    # 'color' factor that is not part of the varying grid.
    stored = classes[:, 1:] if classes.shape[1] == len(n_values) + 1 else classes
    man = _base_manifest("dsprites", idx, F)
    man["file_bytes"] = int(os.path.getsize(path))
    man["decoded_matches_stored_classes"] = bool(np.array_equal(stored, F))
    man["stored_class_columns"] = int(classes.shape[1])
    man["orientation_angle_checks"] = angle_checks
    man["excluded_factor_values"] = {
        f["name"]: sorted(f["exclude_values"]) for f in spec["factors"]
        if f.get("exclude_values")}
    return dict(images=_to_rgb_u8(images), factors=F, source_index=idx,
                manifest=man)


# ---------------------------------------------------------------------------
# MPI3D-realistic, read by HTTP range from the uncompressed zip member
# ---------------------------------------------------------------------------
def _npy_member_data_offset(url: str) -> tuple[int, tuple[int, ...], np.dtype]:
    """Byte offset of the first record, plus the array shape and dtype.

    Reads the zip local header and the .npy header of the single stored member.
    """
    head = _http_range(url, 0, 4095)
    assert head[:4] == b"PK\x03\x04", head[:4]
    nlen, elen = struct.unpack("<HH", head[26:30])
    npy_at = 30 + nlen + elen
    assert head[npy_at:npy_at + 6] == b"\x93NUMPY", head[npy_at:npy_at + 8]
    major = head[npy_at + 6]
    if major == 1:
        hlen = struct.unpack("<H", head[npy_at + 8:npy_at + 10])[0]
        hstart = npy_at + 10
    else:
        hlen = struct.unpack("<I", head[npy_at + 8:npy_at + 12])[0]
        hstart = npy_at + 12
    header = head[hstart:hstart + hlen].decode("latin1")
    meta = eval(header, {"__builtins__": {}}, {"False": False, "True": True})
    assert not meta["fortran_order"], meta
    return hstart + hlen, tuple(meta["shape"]), np.dtype(meta["descr"])


def _http_range(url: str, start: int, end: int, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:  # transient GCS/network hiccup
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range request failed {start}-{end}: {last}")


def load_mpi3d(n_samples: int = C.N_DATASET_SAMPLES, cache_dir: str = "/tmp/data",
               max_gap: int = 64, n_threads: int = 24):
    """Fetch only the selected records. Nearby indices are coalesced into one
    range request; `max_gap` is how many unwanted records are worth reading to
    avoid an extra round trip."""
    spec = C.DATASETS["mpi3d_realistic"]
    url = spec["url"]
    n_values = [f["n_values"] for f in spec["factors"]]
    idx = FA.sample_factor_balanced(n_values, n_samples,
                                    exclude=FA.exclusions_from_spec(spec["factors"]))

    data_off, shape, dtype = _npy_member_data_offset(url)
    total = int(np.prod(n_values))
    assert shape[0] == total, (shape, total)
    rec = int(np.prod(shape[1:])) * dtype.itemsize
    print(f"[mpi3d] member shape={shape} dtype={dtype} record={rec}B "
          f"data_offset={data_off}", flush=True)

    groups: list[list[int]] = [[int(idx[0])]]
    for i in idx[1:]:
        i = int(i)
        if i - groups[-1][-1] - 1 <= max_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    print(f"[mpi3d] {len(idx)} records in {len(groups)} range requests", flush=True)

    # Destination row for each source index, so threads can write disjointly.
    dest = {int(v): j for j, v in enumerate(idx)}
    out = np.empty((len(idx), *shape[1:]), dtype=dtype)
    t0 = time.time()
    counters = {"bytes": 0, "groups": 0, "rows": 0}
    lock = threading.Lock()

    def fetch(g):
        lo, hi = g[0], g[-1]
        raw = _http_range(url, data_off + lo * rec, data_off + (hi + 1) * rec - 1)
        block = np.frombuffer(raw, dtype=dtype).reshape(hi - lo + 1, *shape[1:])
        for i in g:
            out[dest[i]] = block[i - lo]
        with lock:
            counters["bytes"] += len(raw)
            counters["groups"] += 1
            counters["rows"] += len(g)
            if counters["groups"] % 200 == 0:
                print(f"[mpi3d] {counters['groups']}/{len(groups)} groups, "
                      f"{counters['bytes'] / 1e6:.0f} MB, "
                      f"{time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(fetch, groups))
    fetched = counters["bytes"]
    assert counters["rows"] == len(idx), (counters["rows"], len(idx))
    print(f"[mpi3d] fetched {fetched / 1e6:.1f} MB in {time.time() - t0:.1f}s",
          flush=True)

    F = FA.index_to_factors(idx, n_values)
    man = _base_manifest("mpi3d_realistic", idx, F)
    man["bytes_fetched"] = int(fetched)
    man["range_requests"] = int(len(groups))
    man["fetch_seconds"] = round(time.time() - t0, 1)
    man["fetch_threads"] = int(n_threads)
    man["member_shape"] = list(shape)
    man["member_dtype"] = str(dtype)
    man["data_offset"] = int(data_off)
    return dict(images=_to_rgb_u8(out), factors=F, source_index=idx, manifest=man)


LOADERS = {"shapes3d": load_shapes3d, "dsprites": load_dsprites,
           "mpi3d_realistic": load_mpi3d}


def load_dataset(key: str, n_samples: int = C.N_DATASET_SAMPLES, **kw):
    d = LOADERS[key](n_samples=n_samples, **kw)
    imgs = d["images"]
    d["manifest"]["image_report"] = dict(
        shape=list(imgs.shape[1:]), dtype=str(imgs.dtype),
        pixel_mean=float(imgs.mean()), pixel_std=float(imgs.std()),
        n_constant_images=int((imgs.reshape(len(imgs), -1).std(axis=1) < 1e-6).sum()),
        n_duplicate_images=_n_duplicate_rows(imgs.reshape(len(imgs), -1)),
    )
    return d

def _n_duplicate_rows(flat: np.ndarray) -> int:
    """Byte-exact duplicate count. Hashing the FULL row matters: subsampling
    every 97th pixel reported 29 false duplicates out of 96 dSprites images,
    whose sprites are sparse and share most background pixels."""
    seen = {hash(r.tobytes()) for r in np.ascontiguousarray(flat)}
    return int(flat.shape[0] - len(seen))
