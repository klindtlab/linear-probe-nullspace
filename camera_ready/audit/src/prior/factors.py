"""Generative-factor handling for the added disentanglement datasets.

Three jobs:

1. Deterministic factor-balanced sampling. A factor combination is drawn by
   sampling each factor's value uniformly and independently, so every factor's
   marginal is flat by construction. Combinations are deduplicated, so each
   sampled source record is a distinct factor combination and appears once.

2. Target encoding. Continuous and ordinal factors are min-max mapped to [0, 1];
   cyclic factors become a (sin, cos) pair over their period, so the probe is not
   asked to predict a discontinuity; categorical factors are one-hot and are kept
   out of the headline target block (appendix sensitivity only).

3. The D3 checkerboard. Over the two highest-cardinality varying factors, hold
   out the combinations whose index sum has odd parity. Every marginal value of
   both factors still appears in training, but half of their joint combinations
   are unseen, so what is tested is composition of known factor values rather
   than extrapolation past the training range.
"""

from __future__ import annotations

import numpy as np

import rebuttal_config as C


# ---------------------------------------------------------------------------
# Index arithmetic over the source factor grid (lexicographic, index-major)
# ---------------------------------------------------------------------------
def radix_multipliers(n_values: list[int]) -> np.ndarray:
    mult = np.ones(len(n_values), dtype=np.int64)
    for i in range(len(n_values) - 2, -1, -1):
        mult[i] = mult[i + 1] * n_values[i + 1]
    return mult


def factors_to_index(F: np.ndarray, n_values: list[int]) -> np.ndarray:
    return (F.astype(np.int64) * radix_multipliers(n_values)).sum(axis=1)


def index_to_factors(idx: np.ndarray, n_values: list[int]) -> np.ndarray:
    mult = radix_multipliers(n_values)
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty((len(idx), len(n_values)), dtype=np.int64)
    for j in range(len(n_values)):
        out[:, j] = (idx // mult[j]) % n_values[j]
    return out


def sample_factor_balanced(n_values: list[int], n_samples: int,
                           seed: int = C.SAMPLE_SEED,
                           exclude: dict[int, set[int]] | None = None) -> np.ndarray:
    """Distinct factor combinations, each factor's marginal uniform by design.

    `exclude` maps a factor position to values that must not be sampled. It exists
    for closed bin grids: dSprites stores orientation as linspace(0, 2*pi, 40)
    inclusive, so bin 39 is the same angle as bin 0 and would enter the sample as a
    pixel-identical duplicate carrying its own target.
    """
    exclude = exclude or {}
    total = int(np.prod(n_values))
    allowed = [np.asarray([v for v in range(nv) if v not in exclude.get(j, set())])
               for j, nv in enumerate(n_values)]
    if any(len(a) == 0 for a in allowed):
        raise ValueError("a factor has no allowed values after exclusion")
    if n_samples >= int(np.prod([len(a) for a in allowed])):
        grid = np.stack(np.meshgrid(*allowed, indexing="ij"), axis=-1)
        return np.sort(factors_to_index(grid.reshape(-1, len(n_values)), n_values))
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    order: list[int] = []
    while len(order) < n_samples:
        need = n_samples - len(order)
        draw = np.stack([rng.choice(a, size=need * 2) for a in allowed], axis=1)
        for i in factors_to_index(draw, n_values):
            i = int(i)
            if i not in chosen:
                chosen.add(i)
                order.append(i)
                if len(order) == n_samples:
                    break
    assert total >= len(order)
    return np.sort(np.asarray(order, dtype=np.int64))


def exclusions_from_spec(spec: list[dict]) -> dict[int, set[int]]:
    """Per-factor excluded values declared in the dataset spec."""
    return {j: set(f["exclude_values"]) for j, f in enumerate(spec)
            if f.get("exclude_values")}


def marginal_balance_report(F: np.ndarray, spec: list[dict]) -> dict:
    """Observed counts per factor value, to be checkpointed with the data review."""
    rep = {}
    for j, f in enumerate(spec):
        counts = np.bincount(F[:, j], minlength=f["n_values"])
        rep[f["name"]] = dict(
            n_values=int(f["n_values"]),
            counts=[int(c) for c in counts],
            min_count=int(counts.min()), max_count=int(counts.max()),
            relative_spread=float((counts.max() - counts.min()) / max(counts.mean(), 1)),
        )
    return rep


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def encode_targets(F: np.ndarray, spec: list[dict]):
    """Build the headline target block and the categorical appendix block.

    Returns (Y_headline, headline_labels, Y_categorical, categorical_labels).
    """
    cols, labels = [], []
    cat_cols, cat_labels = [], []
    for j, f in enumerate(spec):
        v = F[:, j].astype(np.float64)
        nv = int(f["n_values"])
        if f["kind"] == "linear":
            cols.append((v / max(nv - 1, 1))[:, None])
            labels.append(f["name"])
        elif f["kind"] == "cyclic":
            period = float(f.get("period", nv))
            ang = 2 * np.pi * v / period
            cols.append(np.stack([np.sin(ang), np.cos(ang)], axis=1))
            labels.extend([f"{f['name']}_sin", f"{f['name']}_cos"])
        elif f["kind"] == "categorical":
            oh = np.zeros((len(v), nv))
            oh[np.arange(len(v)), F[:, j].astype(int)] = 1.0
            cat_cols.append(oh)
            cat_labels.extend([f"{f['name']}={i}" for i in range(nv)])
        else:
            raise ValueError(f["kind"])
    Y = np.concatenate(cols, axis=1) if cols else np.zeros((len(F), 0))
    Yc = np.concatenate(cat_cols, axis=1) if cat_cols else np.zeros((len(F), 0))
    return Y, labels, Yc, cat_labels


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def combination_hash(F: np.ndarray) -> np.ndarray:
    """Split unit: a stable hash of the source factor combination."""
    return np.asarray([hash(row.tobytes()) for row in np.ascontiguousarray(F)])


def random_split(n: int, seed: int, test_fraction: float = C.TEST_FRACTION):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def pick_checkerboard_factors(F: np.ndarray, spec: list[dict]) -> tuple[int, int]:
    """The two highest-cardinality factors that actually vary in the sample.

    Deterministic tie-break: higher observed cardinality first, then lower
    factor index.
    """
    varying = [(len(np.unique(F[:, j])), j) for j in range(F.shape[1])]
    varying = [(c, j) for c, j in varying if c > 1]
    varying.sort(key=lambda t: (-t[0], t[1]))
    if len(varying) < 2:
        raise ValueError("need two varying factors for a checkerboard split")
    return varying[0][1], varying[1][1]


def checkerboard_split(F: np.ndarray, spec: list[dict],
                       parity: int = 0, seed: int = C.CHECKERBOARD_SEED):
    """Hold out half the joint combinations of the two chosen factors.

    Membership depends only on the parity of (rank_a + rank_b + offset), so every
    marginal value of each factor appears on both sides of the split.
    """
    ja, jb = pick_checkerboard_factors(F, spec)
    ra = {v: i for i, v in enumerate(np.unique(F[:, ja]))}
    rb = {v: i for i, v in enumerate(np.unique(F[:, jb]))}
    a = np.asarray([ra[v] for v in F[:, ja]])
    b = np.asarray([rb[v] for v in F[:, jb]])
    offset = int(np.random.default_rng(seed).integers(0, 2))
    held = ((a + b + offset) % 2) != parity
    train_idx = np.where(~held)[0]
    test_idx = np.where(held)[0]
    info = dict(
        factor_a=spec[ja]["name"], factor_b=spec[jb]["name"],
        n_train=int(len(train_idx)), n_test=int(len(test_idx)),
        n_combinations_total=int(len(ra) * len(rb)),
        parity_offset=offset,
        train_marginal_coverage_a=int(len(np.unique(a[train_idx]))),
        train_marginal_coverage_b=int(len(np.unique(b[train_idx]))),
        marginal_values_a=int(len(ra)), marginal_values_b=int(len(rb)),
        held_out_combinations=int(len(np.unique(
            np.stack([a[test_idx], b[test_idx]], axis=1), axis=0))),
        train_combinations=int(len(np.unique(
            np.stack([a[train_idx], b[train_idx]], axis=1), axis=0))),
    )
    info["every_marginal_in_train"] = bool(
        info["train_marginal_coverage_a"] == info["marginal_values_a"]
        and info["train_marginal_coverage_b"] == info["marginal_values_b"])
    info["combination_overlap"] = int(len(
        set(map(tuple, np.stack([a[train_idx], b[train_idx]], axis=1)))
        & set(map(tuple, np.stack([a[test_idx], b[test_idx]], axis=1)))))
    return train_idx, test_idx, info
