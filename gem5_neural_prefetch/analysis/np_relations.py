"""Fit relations between a producer load's value and a consumer load's address
(label L2 of the design memo).

The slicer says which earlier loads flow into each load's address. This module
decides whether the flow is one of two exact forms, with a base that repeats
for the same static pair of load PCs:

  element of an array    address = (value << shift) + base, shift 0 to 3
  word of a bit vector   address = ((value >> k) << (k - 3)) + base, for words
                         of 64, 32 and 8 bits (k = 6, 5, 3)

The bit-vector forms are tried only for loads that no array form explains.
They are reported with a negative shift code, -k.
"""

import numpy as np

NONE = np.uint64(0xFFFFFFFFFFFFFFFF)
SHIFTS = (0, 1, 2, 3)
BIT_VECTOR = (-6, -5, -3)  # the word that holds bit `value`, for 64, 32 and 8-bit words
MIN_SUPPORT = 16  # instances that must share one base
MIN_SHARE = 0.10  # of the static pair's instances
MIN_DISTINCT = 4  # different addresses one relation must name


def transform(value, code):
    """The address offset a value contributes under one form."""
    if code >= 0:
        return value << np.uint64(code)
    k = -code
    return (value >> np.uint64(k)) << np.uint64(k - 3)


def read_edges(path):
    """Rows: ld, count, then four (producer_ld, info) pairs."""
    return np.fromfile(path, dtype="<u8").reshape(-1, 10)


def _group_ids(*columns):
    """Dense id for every distinct combination of the given columns."""
    stacked = np.stack([np.asarray(c, dtype=np.uint64) for c in columns], axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def _modes(group, value):
    """Most frequent value in each group: (mode, its count, group size)."""
    n_groups = int(group.max()) + 1 if len(group) else 0
    size = np.bincount(group, minlength=n_groups)
    order = np.lexsort((value, group))
    g, v = group[order], value[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = (g[1:] != g[:-1]) | (v[1:] != v[:-1])
    first = np.flatnonzero(start)
    count = np.diff(np.append(first, len(g)))
    run_group, run_value = g[first], v[first]
    best = np.lexsort((count, run_group))
    last_of_group = np.ones(len(best), dtype=bool)
    last_of_group[:-1] = run_group[best][1:] != run_group[best][:-1]
    pick = best[last_of_group]
    mode = np.zeros(n_groups, dtype=np.uint64)
    mode_count = np.zeros(n_groups, dtype=np.int64)
    mode[run_group[pick]] = run_value[pick]
    mode_count[run_group[pick]] = count[pick]
    return mode, mode_count, size


def _distinct(group, value, mask):
    """Number of distinct values per group among the masked rows."""
    n_groups = int(group.max()) + 1 if len(group) else 0
    g, v = group[mask], value[mask]
    if not len(g):
        return np.zeros(n_groups, dtype=np.int64)
    order = np.lexsort((v, g))
    g, v = g[order], v[order]
    new = np.ones(len(g), dtype=bool)
    new[1:] = (g[1:] != g[:-1]) | (v[1:] != v[:-1])
    return np.bincount(g[new], minlength=n_groups)


def fit(load, edges):
    """Best affine explanation of every load's address.

    Returns arrays over loads:
      derived   True when some producer explains the address
      producer  load index of that producer (-1 if none)
      origin    earliest load that carried the same value as the producer
      shift, base, depth, through_memory
      relation  dense id of the static relation (-1 if none)
    and `relations`, one row per static relation with its support.
    """
    n = len(load["pc"])
    pc = load["pc"]
    vaddr = load["vaddr"]
    value = load["value"]

    derived = np.zeros(n, dtype=bool)
    producer = np.full(n, -1, dtype=np.int64)
    shift = np.full(n, -1, dtype=np.int8)
    base = np.zeros(n, dtype=np.uint64)
    depth = np.full(n, -1, dtype=np.int32)
    through_memory = np.zeros(n, dtype=bool)
    key_columns = []  # per explained load: producer pc, consumer pc, depth, shift, base

    consumer = edges[:, 0].astype(np.int64)
    # Array forms first, for every producer. Then bit-vector forms for what is
    # left. Within a pass the nearest producers are tried first, so the
    # shallowest explanation wins.
    for forms, slot in [(SHIFTS, k) for k in range(4)] + [(BIT_VECTOR, k) for k in range(4)]:
        prod = edges[:, 2 + 2 * slot]
        info = edges[:, 3 + 2 * slot]
        valid = prod != NONE
        if not valid.any():
            continue
        c = consumer[valid]
        p = prod[valid].astype(np.int64)
        d = (info[valid] & np.uint64(0xFFFF)).astype(np.int64)
        mem = ((info[valid] >> np.uint64(16)) & np.uint64(1)) != 0
        group = _group_ids(pc[p], pc[c], d)
        for s in forms:
            residual = vaddr[c] - transform(value[p], s)
            mode, mode_count, size = _modes(group, residual)
            fits = residual == mode[group]
            # A constant producer value "explains" a constant address with
            # any shift. A relation must name several different addresses.
            varied = _distinct(group, vaddr[c], fits) >= MIN_DISTINCT
            ok = (
                fits
                & (mode_count[group] >= MIN_SUPPORT)
                & (mode_count[group] >= MIN_SHARE * size[group])
                & varied[group]
                & ~derived[c]
            )
            if not ok.any():
                continue
            cc = c[ok]
            derived[cc] = True
            producer[cc] = p[ok]
            shift[cc] = s
            base[cc] = residual[ok]
            depth[cc] = d[ok]
            through_memory[cc] = mem[ok]

    # The origin is the earliest load that carried the same value. A pointer
    # reloaded from the stack names the same address as the load that first
    # produced it, and that first load is where a traversal could start.
    origin = producer.copy()
    for slot in range(4):
        prod = edges[:, 2 + 2 * slot]
        valid = (prod != NONE) & derived[consumer]
        c = consumer[valid]
        p = prod[valid].astype(np.int64)
        same = value[p] == value[producer[c]]
        earlier = same & (p < origin[c])
        origin[c[earlier]] = p[earlier]

    explained = np.flatnonzero(derived)
    relation = np.full(n, -1, dtype=np.int64)
    relations = {}
    if len(explained):
        ids = _group_ids(
            pc[producer[explained]],
            pc[explained],
            depth[explained].astype(np.uint64),
            (shift[explained].astype(np.int64) + 16).astype(np.uint64),
            base[explained],
        )
        relation[explained] = ids
        n_rel = int(ids.max()) + 1
        first = np.full(n_rel, -1, dtype=np.int64)
        first[ids[::-1]] = explained[::-1]
        relations = {
            "producer_pc": pc[producer[first]],
            "consumer_pc": pc[first],
            "depth": depth[first],
            "shift": shift[first],
            "base": base[first],
            "instances": np.bincount(ids, minlength=n_rel),
        }
    return {
        "derived": derived,
        "producer": producer,
        "origin": origin,
        "shift": shift,
        "base": base,
        "depth": depth,
        "through_memory": through_memory,
        "relation": relation,
        "relations": relations,
    }
