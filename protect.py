"""Stored-bit faults and the protections both sides may use (round four).

mode   none    raw bits
       parity  one parity bit per 32-bit chunk: odd flips detected, even flips slip through
       secded  8 check bits per 64-bit chunk (Hamming-style): 1 flip corrected, 2 detected,
               3 or more slip through
policy on a detected but uncorrected record:
       drop    the record is not used (a nucleus goes silent, a network's chunk reads as zero)
       keep    the corrupted record is used as it is
Flips are applied to the form actually stored (index list or mask, as priced), check bits
included. Each contender is scored with its best (mode, policy) per flip rate."""
import numpy as np


def _faults(n, rate, r, mode):
    """Return (flip mask to apply, detected?) for one record of n bits."""
    f = r.random(n) < rate
    if mode == "none":
        return f, False
    det = False
    if mode == "parity":
        for c in range(0, n, 32):
            if (int(f[c:c + 32].sum()) + int(r.random() < rate)) % 2 == 1:
                det = True
        return f, det
    for c in range(0, n, 64):                                     # secded
        seg = f[c:c + 64]
        k = int(seg.sum()) + int(np.sum(r.random(8) < rate))
        if k == 1:
            seg[:] = False
        elif k == 2:
            det = True
    return f, det


def _to_bits(v, nbits):
    return ((int(v) >> np.arange(nbits)) & 1).astype(np.uint8)


def _from_bits(b):
    return int(np.sum(b.astype(np.int64) << np.arange(len(b))))


def flip_mycel(m, rate, seed, mode="none", policy="drop"):
    from dmycel import MycelCo
    r = np.random.default_rng(seed)
    L = m.L; ib = int(np.ceil(np.log2(max(L, 2))))
    rows = m.inc.reshape(-1, m.W).copy()
    alive = m.alive.reshape(-1)
    for j in range(len(rows)):
        if alive[j] == 0:
            continue
        mask = np.unpackbits(rows[j].view(np.uint8), bitorder="little")[:L]
        lits = np.flatnonzero(mask)
        index_form = 8 + len(lits) * ib < L
        rec = (np.concatenate([_to_bits(len(lits), 8)] + [_to_bits(l, ib) for l in lits]) if index_form else mask.copy())
        f, det = _faults(len(rec), rate, r, mode)
        if det and policy == "drop":
            alive[j] = 0; continue
        nb = rec ^ f.astype(np.uint8)
        new = np.zeros(m.W * 64, np.uint8)
        if index_form:
            k = min(_from_bits(nb[:8]), len(lits))
            for i in range(k):
                l = _from_bits(nb[8 + i * ib: 8 + (i + 1) * ib])
                if l < L:
                    new[l] = 1
        else:
            new[:L] = nb
        rows[j] = np.packbits(new, bitorder="little").view(np.uint64)
    m.inc = rows.reshape(m.inc.shape).copy()
    if isinstance(m, MycelCo):
        for j in range(m.P):
            if alive[j] == 0:
                continue
            rec = np.unpackbits(m.w8[:, j].copy().view(np.uint8), bitorder="little")
            f, det = _faults(len(rec), rate, r, mode)
            if det and policy == "drop":
                alive[j] = 0; continue
            m.w8[:, j] = np.packbits(rec ^ f.astype(np.uint8), bitorder="little").view(np.int8)
    elif m.weighted:
        wg = m.wgt.reshape(-1)
        for j in range(len(wg)):
            if alive[j] == 0:
                continue
            rec = np.unpackbits(np.array([wg[j]], np.uint8), bitorder="little")
            f, det = _faults(8, rate, r, mode)
            if det and policy == "drop":
                alive[j] = 0; continue
            wg[j] = np.packbits(rec ^ f.astype(np.uint8), bitorder="little")[0]


def flip_mlp(m, rate, seed, mode="none", policy="drop"):
    """Networks: the stored weight arrays, chunk by chunk (int8: 8 weights per 64 bits)."""
    r = np.random.default_rng(seed)
    arrs = [q[0] for q in m.q] if m.q is not None else [m.W1, m.W2]
    step = 64 if mode == "secded" else 32
    for a in arrs:
        flat = a.reshape(-1).view(np.uint8)
        bits = np.unpackbits(flat, bitorder="little")
        for c in range(0, len(bits), step):
            seg = bits[c:c + step]
            f, det = _faults(len(seg), rate, r, mode)
            if det and policy == "drop":
                seg[:] = 0
            else:
                seg ^= f.astype(np.uint8)
        flat[:] = np.packbits(bits, bitorder="little")


SCHEMES = [("none", "keep"), ("parity", "drop"), ("secded", "drop"), ("secded", "keep")]
OVERHEAD = {"none": 0.0, "parity": 1 / 32, "secded": 8 / 64}       # extra stored bits (and reads)
