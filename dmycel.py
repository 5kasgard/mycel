"""Mycel-D: the digital-native translation.

Ports      each feature -> thermometer bits at training quantiles, plus complements (dual rail)
Nuclei     bitmask of open tubes; fires when every open tube sees a 1 (coincidence detection)
Tubes      8-bit saturating counters (use it or lose it); open at >= 128
Contests   one yes/no contest per class; vote = popcount(YES fired) - popcount(NO fired)
Learning   bounded feedback (vote margin sets how often nuclei learn), reinforce / block
Growth     optional fixed tube budget per nucleus; dead tubes regrow toward active inputs
Consolidation  optional: freeze tubes of nuclei that matter for a learned task
"""
import ctypes, os, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, "libdcore.so"))
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
_lib.d_seed.argtypes = [ctypes.c_uint64]
_lib.d_set_quorum.argtypes = [ctypes.c_int]
_lib.d_predict.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 5
_lib.d_epoch.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p] * 7 + [ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_double, ctypes.c_int]
_lib.d_rebuild.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4


class Encoder:
    """Binary ports: B thermometer thresholds per feature at training quantiles, plus complements.
    Optional oblique ports: R random +/-1 sums of the inputs (adds only, no multiplies)."""
    def __init__(self, B, R=0, seed=0):
        self.B, self.R, self.seed = B, R, seed

    def _feat(self, X):
        if self.R == 0:
            return X
        return np.concatenate([X, X @ self.proj], 1)

    def fit(self, X):
        self.d = X.shape[1]
        if self.R:
            self.proj = np.random.default_rng(self.seed + 999).choice([-1.0, 1.0], (self.d, self.R))
        X = self._feat(X)
        qs = np.linspace(0, 1, self.B + 2)[1:-1]
        self.th = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.L = 2 * sum(len(t) for t in self.th)
        self.W = (self.L + 63) // 64
        return self

    def bits(self, X):
        X = self._feat(X)
        cols = [(X[:, [j]] >= t[None, :]) for j, t in enumerate(self.th)]
        b = np.concatenate(cols, 1)
        return np.concatenate([b, ~b], 1).astype(np.uint8)

    def pack(self, Xb):
        n = len(Xb)
        pad = np.zeros((n, self.W * 64), np.uint8); pad[:, :self.L] = Xb
        return np.packbits(pad.reshape(n, -1, 8)[:, :, ::-1], axis=2).reshape(n, self.W, 8).view(np.uint64).reshape(n, self.W).copy()

    def compares(self):
        """Input-side logic, in 32-bit-add units: one compare per thermometer bit, d adds per oblique port."""
        return self.L // 2 + self.R * self.d


class MycelD:
    def __init__(self, K, C=40, T=None, s=3.9, B=8, M=0, grow=0, epochs=50, seed=0, weighted=0, R=0, q=100, open0=0.5):
        self.weighted, self.R, self.q, self.open0 = weighted, R, q, open0
        self.K, self.C, self.s, self.B, self.M0, self.grow, self.epochs, self.seed = K, C, s, B, M, grow, epochs, seed
        self.T = T if T is not None else max(2, C // 2)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- build
    def _init(self, X):
        self.enc = Encoder(self.B, self.R, self.seed).fit(X)
        L, W, K, C = self.enc.L, self.enc.W, self.K, self.C
        self.L, self.W = L, W
        self.M = L if (self.M0 <= 0 or self.M0 >= L) else self.M0
        if self.M == L:
            self.lit = np.tile(np.arange(L, dtype=np.uint16), (K, C, 1))
        else:
            self.lit = np.stack([np.stack([self.rng.choice(L, self.M, replace=False) for _ in range(C)]) for _ in range(K)]).astype(np.uint16)
        self.cnt = np.full((K, C, self.M), 127, np.uint8)
        self.frz = np.zeros((K, C, self.M), np.uint8)
        self.alive = np.ones((K, C), np.uint8)
        self.wgt = np.ones((K, C), np.uint8)
        self.inc = np.zeros((K, C, W), np.uint64); self.has = np.zeros((K, C, W), np.uint64)
        self.rebuild()
        _lib.d_seed(self.seed + 1)

    def rebuild(self):
        _lib.d_rebuild(self.K, self.C, self.M, self.W, P(self.lit), P(self.cnt), P(self.inc), P(self.has))

    def _prep(self, X):
        Xb = np.ascontiguousarray(self.enc.bits(X)); Xp = np.ascontiguousarray(self.enc.pack(Xb))
        return Xb, Xp

    def fit(self, X, y, epochs=None):
        if not hasattr(self, "enc"):
            self._init(X)
        Xb, Xp = self._prep(X)
        self.pon = Xb.mean(0)
        y = np.ascontiguousarray(y, np.int32)
        _lib.d_set_quorum(int(self.q))
        for _ in range(epochs or self.epochs):
            order = self.rng.permutation(len(y)).astype(np.int32)
            _lib.d_epoch(len(y), self.K, self.C, self.M, self.W, self.L, P(self.lit), P(self.cnt), P(self.inc),
                         P(self.has), P(self.alive), P(self.frz), P(self.wgt), int(self.weighted), P(Xp), P(Xb), P(y), P(order),
                         int(self.T), float(self.s), int(self.grow))
        return self

    def votes(self, X):
        Xb, Xp = self._prep(X); self._lastXb = Xb
        out = np.zeros((len(X), self.K), np.int32)
        _lib.d_set_quorum(int(self.q))
        _lib.d_predict(len(X), self.K, self.C, self.W, P(self.inc), P(self.alive), P(self.wgt), P(Xp), P(out))
        return out

    def predict(self, X):
        v = self.votes(X)
        return np.argmax(v + 1e-3 * self.rng.random(v.shape), 1)   # ties broken at random

    # ---------------------------------------------------------------- cost
    def _nucleus_bits(self):
        """Each live nucleus stored in its cheaper form: a full mask (L bits) or a list of open-tube
        indices (8-bit length + ceil(log2 L) bits each). Returns bits per live nucleus and includes."""
        k = np.array([[int(sum(bin(int(w)).count("1") for w in self.inc[c, j])) for j in range(self.C)] for c in range(self.K)])
        idx = 8 + k * int(np.ceil(np.log2(max(self.L, 2))))
        b = np.minimum(self.L, idx) + (8 if self.weighted else 0)     # + vote weight
        return np.where(self.alive > 0, b, 0), np.where(self.alive > 0, k, 0)

    def model_bits(self):
        """Bits read to answer (polarity is implicit in position)."""
        b, _ = self._nucleus_bits()
        return int(b.sum())

    def energy(self, X=None):
        """Energy per answer (pJ), pre-registered model (Horowitz 2014, 45 nm). Memory priced per
        64-bit word touched, by the store the model fits in; logic priced as 32-bit adds (0.1 pJ)."""
        Xb = self.enc.bits(X) if X is not None else self._lastXb
        live = (self.alive > 0).reshape(-1)
        w, t = words_touched(self.inc.reshape(-1, self.W)[live], self.L, Xb, self.pon)
        tier = cache_read_pJ(self.model_bits())
        read = tier * w.sum() + 10.0 * self.W + (tier * live.sum() * 8 / 64 if self.weighted else 0)
        read += self._port_read()
        ops = 0.1 * t.sum() * self._test_ops(t) + self.enc.compares() * 0.1 + live.sum() * 0.1 + self.K * 0.1
        return float(read + ops)

    def _port_read(self):
        """Audit fix: raw input (8-bit) and the stored thresholds (8-bit) are read every answer."""
        nth = self.enc.L // 2
        return 10.0 * np.ceil(self.enc.d * 8 / 64) + 10.0 * np.ceil(nth * 8 / 64)

    def _test_ops(self, t):
        """Audit fix: testing one index against a multi-word input takes ~3 ops (shift, load, mask)."""
        return 3.0 if self.W > 1 else 1.0

    # ---------------------------------------------------------------- damage
    def kill(self, frac, seed):
        r = np.random.default_rng(seed)
        self.alive[r.random(self.alive.shape) < frac] = 0

    def retrain(self, X, y, passes):
        self.fit(X, y, passes)

    # ---------------------------------------------------------------- consolidation
    def consolidate(self, X, y, top=0.5):
        """Freeze the open tubes of the nuclei that fired most usefully on the task just learned."""
        Xb, Xp = self._prep(X)
        use = np.zeros((self.K, self.C))
        for c in range(self.K):
            for j in range(self.C):
                inc = self.inc[c, j]
                if not inc.any():
                    continue
                f = ~np.any(inc[None, :] & ~Xp, axis=1)
                pol = 1 if j < self.C // 2 else -1
                use[c, j] = np.sum(f & ((y == c) == (pol == 1))) - np.sum(f & ((y == c) != (pol == 1)))
        cut = np.quantile(use, 1 - top)
        keep = use >= max(cut, 1)
        self.frz[keep] = (self.cnt[keep] >= 128).astype(np.uint8)
        self.alive[keep & (self.alive > 0)] = 2


def cache_read_pJ(bits):
    """Energy of one 64-bit read, by the size of the store the model must live in."""
    b = bits / 8
    if b <= 8 * 1024: return 10.0
    if b <= 32 * 1024: return 20.0
    if b <= 1024 * 1024: return 100.0
    return 1300.0


_lib.co_predict.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6
_lib.co_epoch.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p] * 7 + [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_double, ctypes.c_int]


class MycelCo(MycelD):
    """Shared colony: one pool of P nuclei; every yes/no contest reads all of them through a
    signed 8-bit weight. Only nuclei that fire send anything downstream (event-driven)."""
    def __init__(self, K, P=100, **kw):
        super().__init__(K, C=P, **kw)
        self.P = P
        self.T = kw.get("T") or 4 * P // 10 + 4

    def _init(self, X):
        self.enc = Encoder(self.B, self.R, self.seed).fit(X)
        L, W, P = self.enc.L, self.enc.W, self.P
        self.L, self.W = L, W
        self.M = L if (self.M0 <= 0 or self.M0 >= L) else self.M0
        if self.M == L:
            self.lit = np.tile(np.arange(L, dtype=np.uint16), (P, 1))
        else:
            self.lit = np.stack([self.rng.choice(L, self.M, replace=False) for _ in range(P)]).astype(np.uint16)
        self.cnt = np.full((P, self.M), 127, np.uint8)
        if self.q < 100:      # quorum nuclei start with a random half of their tubes open
            self.cnt = (127 + (self.rng.random((P, self.M)) < self.open0)).astype(np.uint8)
        self.frz = np.zeros((P, self.M), np.uint8)
        self.alive = np.ones(P, np.uint8)
        self.w8 = self.rng.choice(np.array([-1, 1], np.int8), (self.K, P))
        self.inc = np.zeros((P, W), np.uint64); self.has = np.zeros((P, W), np.uint64)
        self.rebuild()
        _lib.d_seed(self.seed + 1)

    def rebuild(self):
        _lib.d_rebuild(1, self.P, self.M, self.W, P(self.lit), P(self.cnt), P(self.inc), P(self.has))

    def fit(self, X, y, epochs=None):
        if not hasattr(self, "enc"):
            self._init(X)
        Xb, Xp = self._prep(X)
        self.pon = Xb.mean(0)
        y = np.ascontiguousarray(y, np.int32)
        _lib.d_set_quorum(int(self.q))
        for _ in range(epochs or self.epochs):
            order = self.rng.permutation(len(y)).astype(np.int32)
            _lib.co_epoch(len(y), self.K, self.P, self.M, self.W, self.L, P(self.lit), P(self.cnt), P(self.inc),
                          P(self.has), P(self.alive), P(self.frz), P(self.w8), P(Xp), P(Xb), P(y), P(order),
                          int(self.T), float(self.s), int(self.grow))
        return self

    def votes(self, X):
        Xb, Xp = self._prep(X); self._lastXb = Xb
        out = np.zeros((len(X), self.K), np.int32); nf = np.zeros(len(X), np.int32)
        _lib.d_set_quorum(int(self.q))
        _lib.co_predict(len(X), self.K, self.P, self.W, P(self.inc), P(self.alive), P(self.w8), P(Xp), P(out), P(nf))
        self._nf = float(nf.mean())
        return out

    def _nucleus_bits(self):
        k = np.array([int(sum(bin(int(w)).count("1") for w in self.inc[j])) for j in range(self.P)])
        idx = 8 + k * int(np.ceil(np.log2(max(self.L, 2))))
        b = np.minimum(self.L, idx)
        return np.where(self.alive > 0, b, 0), np.where(self.alive > 0, k, 0)

    def model_bits(self):
        b, _ = self._nucleus_bits()
        return int(b.sum()) + int((self.alive > 0).sum()) * self.K * 8

    def energy(self, X=None):
        """Read every live nucleus (early exit as in words_touched); then, for each nucleus that
        fired, read its K 8-bit weights and add them (event-driven)."""
        Xb = self.enc.bits(X) if X is not None else self._lastXb
        live = self.alive > 0
        w, t = words_touched(self.inc[live], self.L, Xb, self.pon)
        tier = cache_read_pJ(self.model_bits())
        nf = self._nf
        read = tier * (w.sum() + nf * np.ceil(self.K * 8 / 64)) + 10.0 * self.W + self._port_read()   # whole word per fired nucleus
        ops = 0.1 * t.sum() * self._test_ops(t) + self.enc.compares() * 0.1 + nf * self.K * 0.1 + self.K * 0.1
        return float(read + ops)

    def kill(self, frac, seed):
        r = np.random.default_rng(seed)
        self.alive[r.random(self.P) < frac] = 0

    def consolidate(self, X, y, top=0.5):
        Xb, Xp = self._prep(X)
        use = np.zeros(self.P)
        for j in range(self.P):
            if not self.inc[j].any():
                continue
            f = ~np.any(self.inc[j][None, :] & ~Xp, axis=1)
            sgn = np.sign(self.w8[y, j]).astype(int)
            use[j] = np.sum(f * sgn)
        keep = use >= max(np.quantile(use, 1 - top), 1)
        self.frz[keep] = (self.cnt[keep] >= 128).astype(np.uint8)
        self.alive[keep & (self.alive > 0)] = 2


def words_touched(inc_rows, L, Xb, pon):
    """Words of memory read per answer, per nucleus, and bit tests. Each nucleus uses the cheaper of:
    mask form   L bits, nuclei packed back to back: L/64 words, AND-NOT + OR per word;
    index form  its own 64-bit-aligned slot: 8-bit header + indices sorted so the tube most likely
                to be blocked comes first; the first word's indices are tested without branching,
                and the rest is fetched only if all of them passed (early exit)."""
    ib = int(np.ceil(np.log2(max(L, 2)))); per_first = (64 - 8) // ib
    Wm = (L + 63) // 64
    words, tests = [], []
    for row in inc_rows:
        lits = np.flatnonzero(np.unpackbits(row.view(np.uint8), bitorder="little")[:L])
        k = len(lits)
        mask_w, mask_t = L / 64.0, 2.0 * Wm
        if k == 0:
            words.append(min(mask_w, 1.0)); tests.append(0.0); continue
        lits = lits[np.argsort(pon[lits])]
        nw = int(np.ceil((8 + k * ib) / 64))
        if nw == 1:
            iw, it = 1.0, float(k)
        else:
            passed = Xb[:, lits[:per_first]].min(1) == 1
            iw, it = 1.0 + passed.mean() * (nw - 1), per_first + passed.mean() * (k - per_first)
        if mask_w <= iw:
            words.append(mask_w); tests.append(mask_t)
        else:
            words.append(iw); tests.append(it)
    return np.array(words), np.array(tests)


def grow_nuclei(m, n_new):
    """Neurogenesis for a shared colony: add n_new fresh nuclei (all tubes closed, weights +-1)."""
    P0 = m.P; L, W, M = m.L, m.W, m.M
    m.P = P0 + n_new; m.C = m.P
    if M == L:
        lit = np.tile(np.arange(L, dtype=np.uint16), (n_new, 1))
    else:
        lit = np.stack([m.rng.choice(L, M, replace=False) for _ in range(n_new)]).astype(np.uint16)
    m.lit = np.concatenate([m.lit, lit]); m.cnt = np.concatenate([m.cnt, np.full((n_new, M), 127, np.uint8)])
    m.frz = np.concatenate([m.frz, np.zeros((n_new, M), np.uint8)])
    m.alive = np.concatenate([m.alive, np.ones(n_new, np.uint8)])
    m.w8 = np.ascontiguousarray(np.concatenate([m.w8, m.rng.choice(np.array([-1, 1], np.int8), (m.K, n_new))], 1))
    m.inc = np.zeros((m.P, W), np.uint64); m.has = np.zeros((m.P, W), np.uint64)
    m.rebuild()
    m.T = 4 * m.P // 10 + 4 if m.T == 4 * P0 // 10 + 4 else m.T
