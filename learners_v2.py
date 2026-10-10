"""Energy model v2 (after the round-five audit). Every learner is priced by the same rules:
  * per store: each memory read/write is priced at the tier of the store it lives in
    (cache_read_pJ: 10/20/100/1300 pJ per 64-bit word for <=8 KB / 32 KB / 1 MB / more);
    what answering reads and what only learning touches are separate stores.
  * latch for everyone: learning straight after answering the same x reuses what the answer computed
    (Mycel: fired bitmap + empty-nucleus bitmap; networks: activations; trees: the path).
  * bytes are bytes: an 8-bit counter costs 8/64 of a word, a float 32/64.
  * logic: 32-bit int op 0.1 pJ, fp32 add 0.9, fp32 multiply 3.7, random draw 0.6.
No setting is tuned per task in the market scan; each family runs a fixed list of variants."""
import ctypes, numpy as np
import dmycel
from dmycel import MycelCo, Encoder, _lib, cache_read_pJ, P as PTR

_lib.d_counters.argtypes = [ctypes.c_void_p]
_lib.d_counters2.argtypes = [ctypes.c_void_p]
_lib.d_reset_counters.argtypes = []
FADD, FMUL = 0.9, 3.7
FMAC = FADD + FMUL
RNG = 0.6
W64 = 64.0


def words(bits):
    return np.ceil(bits / W64)


# ------------------------------------------------------------------ binary-aware thresholds
class Encoder2(Encoder):
    """As Encoder, but a feature whose warm-up values are all 0/1 gets the single threshold 0.5,
    so a token never seen during warm-up is still visible later (the plain quantile rule would
    give it the useless threshold 0 and Mycel could never use it)."""
    def fit(self, X):
        self.d = X.shape[1]
        if self.R:
            self.proj = np.random.default_rng(self.seed + 999).choice([-1.0, 1.0], (self.d, self.R))
        F = self._feat(X)
        qs = np.linspace(0, 1, self.B + 2)[1:-1]
        self.th = []
        for j in range(F.shape[1]):
            col = F[:, j]
            if j < self.d and np.all((col == 0) | (col == 1)):
                self.th.append(np.array([0.5]))
            else:
                t = np.unique(np.quantile(col, qs))
                self.th.append(t)
        self.L = 2 * sum(len(t) for t in self.th)
        self.W = (self.L + 63) // 64
        return self


def _scaler(Xw):
    """Standardise on the warm-up; 0/1 features are left as they are, so sparse inputs stay sparse."""
    mu = Xw.mean(0); sd = Xw.std(0) + 1e-6
    b = np.all((Xw == 0) | (Xw == 1), axis=0)
    mu[b] = 0.0; sd[b] = 1.0
    return mu, sd


class Base:
    answer_pJ = 0.0; update_pJ = 0.0; n_ans = 0; n_upd = 0
    def boundary(self): pass


# ------------------------------------------------------------------ Mycel
class Mycel2(Base):
    def __init__(self, K, P=512, B=24, s=3.0, T=None, M=16, R=0, gate=0.0, ga=0, seed=0, price_every=500):
        self.cfg = dict(P=P, B=B, s=s, M=M, R=R, gate=gate, ga=ga)
        self.m = MycelCo(K, P=P, B=B, s=s, T=T or P, epochs=1, seed=seed, M=M, grow=1, R=R)
        self.K, self.gate, self.ga, self.price_every = K, gate, ga, price_every
        self.e_fast = self.e_slow = 0.5
        self.recent = []; self._ans = None; self.answer_pJ = 0.0; self.update_pJ = 0.0
        self.c1 = np.zeros(3, np.uint64); self.c2 = np.zeros(6, np.uint64)

    def start(self, Xw, K):
        m = self.m
        dmycel.Encoder = Encoder2                      # MycelCo._init looks the name up at call time
        try:
            m._init(Xw)
        finally:
            dmycel.Encoder = Encoder
        m.pon = m.enc.bits(Xw).mean(0)
        self.y1 = np.zeros(1, np.int32); self.order = np.zeros(1, np.int32)
        self._tiers()

    def _tiers(self):
        m = self.m
        self.tier_m = cache_read_pJ(m.model_bits())                       # what answering reads (masks/indices + weights)
        self.state_bits = m.lit.size * 16 + m.cnt.size * 8 + m.frz.size * 8 + m.has.size * 64
        self.tier_s = cache_read_pJ(self.state_bits)                      # what only learning touches

    def answer(self, x):
        m = self.m
        Xb, Xp = m._prep(x[None, :]); self._xb, self._xp = Xb, Xp
        out = np.zeros((1, self.K), np.int32); nf = np.zeros(1, np.int32)
        _lib.co_predict(1, m.K, m.P, m.W, PTR(m.inc), PTR(m.alive), PTR(m.w8), PTR(Xp), PTR(out), PTR(nf))
        self.recent.append(x)
        if self._ans is None or len(self.recent) >= self.price_every:
            m.votes(np.array(self.recent[-200:])); self._ans = m.energy(); self._tiers(); self.recent = []
        self.answer_pJ += self._ans; self.n_ans += 1
        self._v = out[0].copy()
        return int(np.flatnonzero(self._v == self._v.max())[0])

    def learn(self, x, y):
        m = self.m; self.y1[0] = y; v = self._v; self.n_upd += 1
        if self.ga:
            err = float(np.flatnonzero(v == v.max())[0] != y)
            self.e_fast += (err - self.e_fast) / 100.0; self.e_slow += (err - self.e_slow) / 2000.0
            self.update_pJ += 4 * FADD
        if self.gate and not (self.ga and self.e_fast > self.e_slow):
            mg = int(v[y]) - (int(np.delete(v, y).max()) if self.K > 1 else 0)
            self.update_pJ += 0.1 * self.K
            if mg >= self.gate * m.T or (self.ga == 2 and -mg >= self.gate * m.T):
                return
        _lib.d_reset_counters()
        _lib.co_epoch(1, m.K, m.P, m.M, m.W, m.L, PTR(m.lit), PTR(m.cnt), PTR(m.inc), PTR(m.has), PTR(m.alive),
                      PTR(m.frz), PTR(m.w8), PTR(self._xp), PTR(self._xb), PTR(self.y1), PTR(self.order),
                      int(m.T), float(m.s), int(m.grow))
        _lib.d_counters(PTR(self.c1)); _lib.d_counters2(PTR(self.c2))
        ev, visit, bump = (float(a) for a in self.c1)
        rng, sel, wupd, flip, regrow, scan = (float(a) for a in self.c2)
        tm, ts = self.tier_m, self.tier_s
        e = 2 * 10.0 * words(m.P)                                         # latched fired + empty bitmaps
        e += visit * (ts * 32 / W64 + 10.0 * 8 / W64 + 0.3)               # slot: literal(2B) counter(1B) freeze(1B) + input bit
        e += bump * (ts * 8 / W64 + 0.1)                                  # counter write
        e += flip * 2 * tm                                                # mask word read-modify-write in the answer store
        e += sel * (tm * 8 / W64 + 0.1) + wupd * (tm * 8 / W64 + 0.1)     # weight byte read / write
        e += regrow * (ts * 16 / W64 + 2 * 2 * ts)                        # new slot literal + two 'has' bit RMWs
        e += scan / W64 * 2 * (ts + 10.0) + scan / W64 * 0.3              # scan input & 'has' bitmasks, word at a time
        e += rng * RNG
        self.update_pJ += e

    def mem_bits(self):
        return int(self.m.model_bits() + self.state_bits)


# ------------------------------------------------------------------ networks and linear models
class MLP2(Base):
    """One hidden layer, fp32. opt='adam' or 'sgd'. Weights are the answer store; Adam moments a second store."""
    def __init__(self, K, h=64, lr=0.003, opt="adam", seed=0):
        self.K, self.h, self.lr, self.opt = K, h, lr, opt; self.r = np.random.default_rng(seed)
        self.cfg = dict(h=h, lr=lr, opt=opt); self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        d = Xw.shape[1]; self.d = d; r = self.r
        self.mu, self.sd = _scaler(Xw)
        self.ps = [r.normal(0, np.sqrt(2 / d), (d, self.h)).astype(np.float32), np.zeros(self.h, np.float32),
                   r.normal(0, np.sqrt(1 / self.h), (self.h, K)).astype(np.float32), np.zeros(K, np.float32)]
        self.mm = [np.zeros_like(p) for p in self.ps]; self.vv = [np.zeros_like(p) for p in self.ps]; self.t = 0
        self.n = sum(p.size for p in self.ps); self.fwd = d * self.h + self.h * K
        self.tw = cache_read_pJ(self.n * 32); self.tmom = cache_read_pJ(self.n * 64)
        self.top = self.h * K + self.h + K                                # second layer + biases

    def answer(self, x):
        x = ((x - self.mu) / self.sd).astype(np.float32); nz = int(np.count_nonzero(x))
        # sparse-aware: only rows of W1 for non-zero inputs are read and multiplied
        self.answer_pJ += self.tw * (words(nz * self.h * 32) + words(self.top * 32)) + (nz * self.h + self.h * self.K) * FMAC \
            + self.h * 0.9 + 10 * words(self.d * 32) + nz * (FADD + FMUL); self.n_ans += 1
        a = x @ self.ps[0] + self.ps[1]; hd = np.maximum(a, 0); z = hd @ self.ps[2] + self.ps[3]
        self._c = (x, a, hd, z, nz); return int(np.argmax(z))

    def learn(self, x, y):
        x, a, hd, z, nz = self._c; self.n_upd += 1
        z = z - z.max(); p = np.exp(z); p /= p.sum(); dz = p.copy(); dz[y] -= 1
        g = [None] * 4; g[2] = np.outer(hd, dz); g[3] = dz; dh = (self.ps[2] @ dz) * (a > 0); g[0] = np.outer(x, dh); g[1] = dh
        e = (2 * self.h * self.K + nz * self.h) * FMAC + self.K * 10          # backward (forward latched), softmax
        if self.opt == "adam":
            self.t += 1
            for i, prm in enumerate(self.ps):
                self.mm[i] = 0.9 * self.mm[i] + 0.1 * g[i]; self.vv[i] = 0.999 * self.vv[i] + 0.001 * g[i] ** 2
                prm -= self.lr * (self.mm[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.vv[i] / (1 - 0.999 ** self.t)) + 1e-8)
            e += self.n * (10 * FADD + 2 * FMUL)                              # moment updates, sqrt, divide (approx.)
            e += self.tw * words(self.n * 32) * 2 + self.tmom * words(self.n * 64) * 2   # weights RMW + moments RMW
        else:                                                             # SGD touches only W1 rows of non-zero inputs
            for i, prm in enumerate(self.ps):
                prm -= self.lr * g[i]
            e += (nz * self.h + self.top) * FMAC + self.tw * (words(nz * self.h * 32) + words(self.top * 32)) * 2
        self.update_pJ += e

    def mem_bits(self):
        return int(self.n * (96 if self.opt == "adam" else 32))


class Logistic2(Base):
    """Online softmax regression, SGD, fp32, standardised on warm-up."""
    def __init__(self, K, lr=0.05, seed=0):
        self.K, self.lr = K, lr; self.cfg = dict(lr=lr); self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        d = Xw.shape[1]; self.d = d; self.mu, self.sd = _scaler(Xw)
        self.Wt = np.zeros((d, K), np.float32); self.b = np.zeros(K, np.float32)
        self.n = d * K + K; self.tw = cache_read_pJ(self.n * 32)

    def answer(self, x):
        x = ((x - self.mu) / self.sd).astype(np.float32); nz = int(np.count_nonzero(x))
        self.answer_pJ += self.tw * words((nz + 1) * self.K * 32) + (nz + 1) * self.K * FMAC + 10 * words(self.d * 32) + nz * (FADD + FMUL)
        self.n_ans += 1; z = x @ self.Wt + self.b; self._c = (x, z, nz)
        return int(np.argmax(z))

    def learn(self, x, y):
        x, z, nz = self._c; self.n_upd += 1
        z = z - z.max(); p = np.exp(z); p /= p.sum(); p[y] -= 1
        self.Wt -= self.lr * np.outer(x, p); self.b -= self.lr * p
        self.update_pJ += self.K * 10 + (nz + 1) * self.K * FMAC + self.tw * words((nz + 1) * self.K * 32) * 2

    def mem_bits(self):
        return int(self.n * 32)


class GaussNB2(Base):
    """Online Gaussian naive Bayes (Welford statistics per class and feature)."""
    def __init__(self, K, seed=0):
        self.K = K; self.cfg = {}; self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        d = self.d = Xw.shape[1]
        self.n = np.zeros(K); self.mu = np.zeros((K, d)); self.m2 = np.ones((K, d))
        self.tw = cache_read_pJ(K * d * 3 * 32)
        self.ans = self.tw * words(K * d * 2 * 32) + K * d * (2 * FADD + 2 * FMUL) + 10 * words(d * 32)

    def answer(self, x):
        self.answer_pJ += self.ans; self.n_ans += 1
        var = self.m2 / np.maximum(self.n[:, None], 1) + 1e-3
        ll = -0.5 * (((x - self.mu) ** 2) / var + np.log(var)).sum(1) + np.log((self.n + 1) / (self.n.sum() + self.K))
        return int(np.argmax(ll))

    def learn(self, x, y):
        self.n[y] += 1; dl = x - self.mu[y]; self.mu[y] += dl / self.n[y]; self.m2[y] += dl * (x - self.mu[y])
        self.update_pJ += self.tw * words(self.d * 3 * 32) * 2 + self.d * (3 * FADD + 2 * FMUL + 3.7); self.n_upd += 1

    def mem_bits(self):
        return int(self.K * self.d * 3 * 32)


class KNN2(Base):
    def __init__(self, K, window=1000, k=9, seed=0):
        self.K, self.window, self.k = K, window, k; self.cfg = dict(window=window, k=k)
        self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        self.d = Xw.shape[1]; self.lo = Xw.min(0); self.sc = np.maximum(Xw.max(0) - self.lo, 1e-9)
        self.X = np.zeros((self.window, self.d), np.float32); self.y = np.zeros(self.window, int); self.n = 0
        self.tier = cache_read_pJ(self.window * (self.d * 8 + 8))

    def answer(self, x):
        n = min(self.n, self.window); self.n_ans += 1
        self.answer_pJ += n * self.d * 0.33 + self.tier * n * words(self.d * 8) + n * 0.1 * np.log2(self.k + 1) + 10 * words(self.d * 8)
        x = np.clip((x - self.lo) / self.sc, -0.5, 1.5); self._x = x
        if n == 0: return 0
        D = np.sum((self.X[:n] - x) ** 2, 1); k = min(self.k, n)
        nn = np.argpartition(D, k - 1)[:k]
        return int(np.bincount(self.y[:n][nn], minlength=self.K).argmax())

    def learn(self, x, y):
        i = self.n % self.window; self.X[i] = self._x; self.y[i] = y; self.n += 1; self.n_upd += 1
        self.update_pJ += self.tier * words(self.d * 8 + 8)

    def mem_bits(self):
        return int(self.window * (self.d * 8 + 8))


# ------------------------------------------------------------------ trees
class HTree2(Base):
    """VFDT with Gaussian leaf statistics and a DDM reset (as round five), priced per store:
    node store (64 bits per node), leaf class counts (K x 32 bits per leaf), leaf statistics
    (K x d x 3 x 32 bits per leaf, touched only by learning and split attempts)."""
    def __init__(self, K, grace=50, delta=1e-3, tau=0.05, seed=0, max_depth=20):
        from learners_s import HTree
        self.t = HTree(K, grace=grace, delta=delta, tau=tau, seed=seed, max_depth=max_depth)
        self.K = K; self.cfg = dict(grace=grace, delta=delta); self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        self.t.start(Xw, K); self.d = Xw.shape[1]; self._tiers()

    def _count(self, nd):
        if nd["leaf"]:
            return 1, 1
        n1, l1 = self._count(nd["l"]); n2, l2 = self._count(nd["r"])
        return 1 + n1 + n2, l1 + l2

    def _tiers(self):
        nodes, leaves = self._count(self.t.root)
        self.tn = cache_read_pJ(nodes * 64); self.tc = cache_read_pJ(leaves * self.K * 32)
        self.tsx = cache_read_pJ(leaves * self.K * self.d * 3 * 32); self.leaves = leaves; self.nodes = nodes

    def answer(self, x):
        t = self.t; leaf = t._find(t.root, x); self.n_ans += 1
        if self.n_ans % 500 == 0: self._tiers()
        self.answer_pJ += leaf["depth"] * (self.tn + 0.1) + self.tc * words(self.K * 32) + 10 * words(self.d * 8) + self.K * 0.1
        t._last = int(leaf["n"].argmax()) if leaf["n"].sum() else 0
        return t._last

    def learn(self, x, y):
        t = self.t; self.n_upd += 1
        leaves_attempt = self._splits_due(x)
        t.learn(x, y)                                          # its own update_pJ is ignored; priced here instead
        e = self.tc * words(self.K * 32) * 2 + self.tsx * words(self.d * 3 * 32) * 2 + self.d * (3 * FADD + 2 * FMUL)
        if t.bg is not None:
            e *= 2
        e += leaves_attempt * (self.tsx * words(self.K * self.d * 3 * 32) + self.d * 10 * self.K * 20 * 0.1)
        self.update_pJ += e

    def _splits_due(self, x):
        leaf = self.t._find(self.t.root, x)
        return int(leaf["since"] + 1 >= self.t.grace)

    def mem_bits(self):
        return int(self.nodes * 64 + self.leaves * self.K * 32 + self.leaves * self.K * self.d * 3 * 32)


class River2(Base):
    """river HAT / ARF, priced per store like HTree2. Both use naive-Bayes-adaptive leaves by default,
    so an answer also reads the leaf statistics (K x d x 2 values) and does K x d Gaussian terms, per tree.
    ARF also runs two ADWIN detectors per tree on every update (~20 int ops each)."""
    def __init__(self, K, model="arf", seed=0, **kw):
        self.K, self.model, self.kw, self.seed = K, model, kw, seed; self.cfg = dict(model=model, **kw)
        self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        import patches  # noqa: river submodule import fix
        from learners_s import _river
        rv = _river(); self.d = Xw.shape[1]
        if self.model == "hat":
            self.m = rv.tree.HoeffdingAdaptiveTreeClassifier(seed=self.seed, **self.kw); self.trees = 1
        else:
            self.m = rv.forest.ARFClassifier(seed=self.seed, **self.kw); self.trees = self.kw.get("n_models", 10)
        self.grace = self.kw.get("grace_period", 50 if self.model == "arf" else 200)
        self.depth, self.leaves, self.nodes = 1.0, float(self.trees), float(self.trees); self._tiers()

    def _tiers(self):
        try:
            ms = self.m.models if hasattr(self.m, "models") else [self.m]
            self.depth = float(np.mean([t.height for t in ms]))
            self.leaves = float(np.sum([t.n_leaves for t in ms])); self.nodes = float(np.sum([t.n_nodes for t in ms]))
        except Exception:
            pass
        self.tn = cache_read_pJ(self.nodes * 64)
        self.tsx = cache_read_pJ(self.leaves * self.K * self.d * 3 * 32)

    def _x(self, x):
        return {i: float(v) for i, v in enumerate(x)}

    def answer(self, x):
        p = self.m.predict_one(self._x(x)); self.n_ans += 1
        per_tree = self.depth * (self.tn + 0.1) + self.tsx * words(self.K * self.d * 2 * 32) + self.K * self.d * (2 * FADD + 2 * FMUL)
        self.answer_pJ += self.trees * per_tree + 10 * words(self.d * 8)
        return int(p) if p is not None else 0

    def learn(self, x, y):
        self.m.learn_one(self._x(x), int(y)); self.n_upd += 1
        if self.n_upd % 500 == 0 or self.n_upd == 50: self._tiers()
        per_tree = self.tsx * words(self.d * 3 * 32) * 2 + self.d * (3 * FADD + 2 * FMUL)
        per_tree += (self.tsx * words(self.K * self.d * 3 * 32) + self.d * 10 * self.K * 20 * 0.1) / self.grace
        if self.model == "arf": per_tree += 2 * 20 * 0.1
        self.update_pJ += self.trees * per_tree

    def mem_bits(self):
        return int(self.nodes * 64 + self.leaves * self.K * self.d * 3 * 32)


class Majority2(Base):
    def __init__(self, K, seed=0):
        self.K = K; self.cfg = {}; self.answer_pJ = 0.0; self.update_pJ = 0.0
    def start(self, Xw, K): self.c = np.zeros(K)
    def answer(self, x): self.n_ans += 1; self.answer_pJ += 10.1; return int(self.c.argmax())
    def learn(self, x, y): self.c[y] += 1; self.update_pJ += 20.2; self.n_upd += 1
    def mem_bits(self): return self.K * 32


# fixed variant lists (no per-task tuning). Mycel A is the round-five frozen design.
VARIANTS = {
    "mycelT": (Mycel2, dict(P=128, B=8, s=3.0, M=8)),
    "mycelA": (Mycel2, dict(P=512, B=24, s=3.0, M=16, gate=0.75, ga=2)),
    "mycelS": (Mycel2, dict(P=256, B=16, s=3.0, M=8)),
    "mycelL": (Mycel2, dict(P=1024, B=24, s=3.0, M=16, gate=0.75, ga=2)),
    "mlp_adam128": (MLP2, dict(h=128, lr=0.003, opt="adam")),
    "mlp_sgd32": (MLP2, dict(h=32, lr=0.02, opt="sgd")),
    "logistic": (Logistic2, dict(lr=0.05)),
    "gnb": (GaussNB2, {}),
    "knn": (KNN2, dict(window=1000, k=9)),
    "htree": (HTree2, dict(grace=50, delta=1e-3)),
    "majority": (Majority2, {}),
    "hat": (River2, dict(model="hat")),
    "arf5": (River2, dict(model="arf", n_models=5)),
    "arf20": (River2, dict(model="arf", n_models=20)),
}
