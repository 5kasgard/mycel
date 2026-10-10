"""Online learners for round five, each with the same interface:
    start(Xwarm, K)        fit scalers/encoders on the warm-up examples only
    answer(x) -> class     (counts answer energy)
    learn(x, y)            (counts update energy)
    boundary()             optional: a known task boundary (task-sequence streams only)
    answer_pJ, update_pJ   running totals; mem_bits() memory held
Energy uses the round-three cost table (45 nm): 64-bit read 10/20/100/1300 pJ by store size,
32-bit add 0.1, fp32 multiply 3.7 and add 0.9, int8 multiply 0.2."""
import ctypes, numpy as np
from dmycel import MycelD, MycelCo, Encoder, _lib, cache_read_pJ, words_touched, P as PTR

_lib.d_counters.argtypes = [ctypes.c_void_p]
_lib.d_reset_counters.argtypes = []
FMAC = 3.7 + 0.9


class Base:
    answer_pJ = 0.0; update_pJ = 0.0; n_ans = 0; n_upd = 0
    def boundary(self): pass


# ------------------------------------------------------------------ Mycel
class OnlineMycel(Base):
    def __init__(self, K, tag="shared", P=64, C=8, B=4, s=4.0, T=None, w=0, consolidate=0.0, grow=0, seed=0,
                 price_every=500, bmode=0, M=0, reuse=0, gate=0.0, R=0, ga=0):
        self.cfg = dict(tag=tag, P=P, C=C, B=B, s=s, T=T, w=w, consolidate=consolidate, grow=grow, bmode=bmode, M=M)
        if reuse: self.cfg["reuse"] = reuse
        if gate: self.cfg["gate"] = gate
        if R: self.cfg["R"] = R
        if ga: self.cfg["ga"] = ga
        self.reuse, self.gate, self.ga = reuse, gate, ga
        self.e_fast, self.e_slow = 0.5, 0.5                              # running error rates (surprise detector)
        self.bx, self.by = [], []
        if tag == "shared":
            self.m = MycelCo(K, P=P, B=B, s=s, T=T, epochs=1, seed=seed, M=M, grow=grow, R=R)
        else:
            self.m = MycelD(K, C=C, B=B, s=s, T=T, weighted=w, epochs=1, seed=seed, M=M, grow=grow, R=R)
        self.K, self.price_every, self.seed = K, price_every, seed
        self.recent = []; self._ans_cost = None; self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        self.m._init(Xw)
        self.m.pon = self.m.enc.bits(Xw).mean(0)
        self.y1 = np.zeros(1, np.int32); self.order = np.zeros(1, np.int32)
        self.cnt_buf = np.zeros(3, np.uint64)

    def answer(self, x):
        Xb, Xp = self.m._prep(x[None, :])
        self._xb, self._xp = Xb, Xp
        m = self.m
        out = np.zeros((1, self.K), np.int32)
        if isinstance(m, MycelCo):
            nf = np.zeros(1, np.int32)
            _lib.co_predict(1, m.K, m.P, m.W, PTR(m.inc), PTR(m.alive), PTR(m.w8), PTR(Xp), PTR(out), PTR(nf))
        else:
            _lib.d_predict(1, m.K, m.C, m.W, PTR(m.inc), PTR(m.alive), PTR(m.wgt), PTR(Xp), PTR(out))
        self.recent.append(x)
        if self._ans_cost is None or len(self.recent) >= self.price_every:
            R = np.array(self.recent[-200:]); m.votes(R); self._ans_cost = m.energy(); self.recent = []
        self.answer_pJ += self._ans_cost; self.n_ans += 1
        v = out[0]; self._v = v.copy()
        return int(np.flatnonzero(v == v.max())[0])

    def learn(self, x, y):
        m = self.m; self.y1[0] = y
        if self.ga:                                                      # neuromodulation: surprise (recent error above
            v = self._v; err = float(np.flatnonzero(v == v.max())[0] != y)  # the long-run error) switches satiation off
            self.e_fast += (err - self.e_fast) / 100.0; self.e_slow += (err - self.e_slow) / 2000.0
            self.update_pJ += 0.4
        surprised = self.ga and self.e_fast > self.e_slow
        if self.gate and not surprised:                                  # satiation: a confident right answer teaches nothing
            v = self._v; T = m.T if getattr(m, "T", None) else 1
            mo = np.delete(v, y).max() if len(v) > 1 else 0
            self.update_pJ += 0.1 * self.K
            mg = v[y] - mo
            if mg >= self.gate * T or (self.ga == 2 and -mg >= self.gate * T):   # ga=2: when calm, a confident
                self.n_upd += 1; return                                  # contradiction is treated as noise too
        _lib.d_reset_counters()
        if isinstance(m, MycelCo):
            _lib.co_epoch(1, m.K, m.P, m.M, m.W, m.L, PTR(m.lit), PTR(m.cnt), PTR(m.inc), PTR(m.has), PTR(m.alive),
                          PTR(m.frz), PTR(m.w8), PTR(self._xp), PTR(self._xb), PTR(self.y1), PTR(self.order),
                          int(m.T), float(m.s), int(m.grow))
        else:
            _lib.d_epoch(1, m.K, m.C, m.M, m.W, m.L, PTR(m.lit), PTR(m.cnt), PTR(m.inc), PTR(m.has), PTR(m.alive),
                         PTR(m.frz), PTR(m.wgt), int(m.weighted), PTR(self._xp), PTR(self._xb), PTR(self.y1),
                         PTR(self.order), int(m.T), float(m.s), int(m.grow))
        _lib.d_counters(PTR(self.cnt_buf))
        ev, vi, bu = (float(v) for v in self.cnt_buf)
        tier_m = cache_read_pJ(m.inc.size * 64)                         # masks read while learning
        tier_s = cache_read_pJ(m.cnt.size * 8 * 4)                      # training state: slot, counter, freeze bytes
        if self.reuse:                                                   # fired bitmap latched from the answer (same model, same x)
            nP = m.P if isinstance(m, MycelCo) else m.K * m.C
            self.update_pJ += 10.0 * np.ceil(nP / 64) + vi * (tier_s * 5 / 64 + 0.3) + bu * (tier_s / 64 + 0.1)
        else:
            self.update_pJ += ev * (tier_m * m.W + 0.2 * m.W) + vi * (tier_s * 5 / 64 + 0.3) + bu * (tier_s / 64 + 0.1)
        self.n_upd += 1
        if self.cfg.get("bmode"):
            self.bx.append(x); self.by.append(y); self.bx = self.bx[-200:]; self.by = self.by[-200:]
        if self.cfg["consolidate"] > 0 and self.n_upd % 1000 == 0:
            self._slow_consolidate()

    def boundary(self):
        """Known task boundary (task-sequence streams only, as for EWC): consolidate what served the
        last task, double its votes, grow fresh nuclei for the next (round-four recipe)."""
        if not self.cfg.get("bmode") or not isinstance(self.m, MycelCo) or len(self.bx) < 20: return
        from dmycel import grow_nuclei
        m = self.m; X = np.array(self.bx); Y = np.array(self.by)
        m.consolidate(X, Y, 1.0); k = m.alive == 2
        m.w8[:, k] = np.clip(m.w8[:, k].astype(int) * 2, -127, 127).astype(np.int8)
        grow_nuclei(m, self.cfg["P"] // 2)
        self.update_pJ += len(Y) * m.P * 2.0                             # one pass of evaluations over the recent examples

    def _slow_consolidate(self):
        """Online consolidation without task boundaries: tubes saturated near the top of their range
        are frozen with a small probability each round (late-phase strengthening)."""
        m = self.m
        sat = (m.cnt >= 250) & (m.frz == 0)
        pick = sat & (m.rng.random(m.cnt.shape) < self.cfg["consolidate"])
        m.frz[pick] = 1

    def mem_bits(self):
        return int(self.m.cnt.size * 8 * 4 + self.m.inc.size * 64)


# ------------------------------------------------------------------ networks
class OnlineMLP(Base):
    """One hidden layer, float32, Adam, one example per step; optional replay buffer and EWC at boundaries."""
    def __init__(self, K, h=32, lr=0.01, replay=0, k=1, ewc=0.0, seed=0, reuse=0, gate=0.0, ga=0):
        self.K, self.h, self.lr, self.replay, self.k, self.ewc_lam = K, h, lr, replay, k, ewc
        self.r = np.random.default_rng(seed); self.cfg = dict(h=h, lr=lr, replay=replay, k=k, ewc=ewc)
        if reuse: self.cfg["reuse"] = reuse
        if gate: self.cfg["gate"] = gate
        if ga: self.cfg["ga"] = ga
        self.reuse, self.gate, self.ga = reuse, gate, ga
        self.e_fast, self.e_slow = 0.5, 0.5
        self.buf_x, self.buf_y, self.seen = [], [], 0
        self.answer_pJ = 0.0; self.update_pJ = 0.0; self.anchor = None; self.recentx, self.recenty = [], []

    def start(self, Xw, K):
        d = Xw.shape[1]; self.d = d; r = self.r
        self.W1 = r.normal(0, np.sqrt(2 / d), (d, self.h)).astype(np.float32); self.b1 = np.zeros(self.h, np.float32)
        self.W2 = r.normal(0, np.sqrt(1 / self.h), (self.h, K)).astype(np.float32); self.b2 = np.zeros(K, np.float32)
        self.ps = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.ps]; self.v = [np.zeros_like(p) for p in self.ps]; self.t = 0
        self.nparam = sum(p.size for p in self.ps)
        self.tier = cache_read_pJ(self.nparam * 32 * 3)                  # weights + two Adam moments
        self.fwd = self.d * self.h + self.h * K
        self.ans_cost = cache_read_pJ(self.nparam * 32) * self.nparam * 32 / 64 + self.fwd * FMAC + self.h * 1.8 + 10 * np.ceil(d * 32 / 64)

    def _fwd(self, x):
        a = x @ self.W1 + self.b1; hd = np.maximum(a, 0); z = hd @ self.W2 + self.b2
        return a, hd, z

    def answer(self, x):
        self.answer_pJ += self.ans_cost; self.n_ans += 1
        z = self._fwd(x.astype(np.float32))[2]; self._z = z
        return int(np.argmax(z))

    def _step(self, X, Y):
        a, hd, z = self._fwd(X); z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        T = np.eye(self.K, dtype=np.float32)[Y]; dz = (p - T) / len(Y)
        g = [None] * 4
        g[2] = hd.T @ dz; g[3] = dz.sum(0); dh = (dz @ self.W2.T) * (a > 0); g[0] = X.T @ dh; g[1] = dh.sum(0)
        if self.anchor is not None:
            lam, F, ref = self.anchor
            for i in range(4): g[i] = g[i] + lam * F[i] * (self.ps[i] - ref[i])
        self.t += 1
        for i, prm in enumerate(self.ps):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i]; self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] ** 2
            prm -= self.lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)
        n = len(Y)
        # forward again for the batch, backward (2x MACs), Adam: ~10 flops and 3 arrays read+written per parameter
        return n * (3 * self.fwd * FMAC) + self.nparam * 10 * 0.9 + self.tier * self.nparam * 32 * 6 / 64

    def learn(self, x, y):
        surprised = False
        if self.ga:                                                      # same surprise detector as Mycel's
            err = float(np.argmax(self._z) != y); self.update_pJ += 0.4
            self.e_fast += (err - self.e_fast) / 100.0; self.e_slow += (err - self.e_slow) / 2000.0
            surprised = self.e_fast > self.e_slow
        if self.gate and not surprised:                                  # same satiation rule: skip if p(y) >= gate
            z = self._z - self._z.max(); p = np.exp(z); p /= p.sum(); self.update_pJ += self.K * 2.0
            if p[y] >= self.gate or (self.ga == 2 and p[y] <= 1 - self.gate):
                self.n_upd += 1; return
        X = [x]; Y = [y]
        if self.replay and self.buf_x:
            idx = self.r.integers(0, len(self.buf_x), min(self.k, len(self.buf_x)))
            X += [self.buf_x[i] for i in idx]; Y += [self.buf_y[i] for i in idx]
            self.update_pJ += len(idx) * cache_read_pJ(self.replay * self.d * 8) * np.ceil(self.d * 8 / 64)
        self.update_pJ += self._step(np.array(X, np.float32), np.array(Y)); self.n_upd += 1
        if self.reuse: self.update_pJ -= self.fwd * FMAC                 # activations for x latched from the answer
        if self.replay:                                                  # reservoir sampling keeps an even sample of the past
            self.seen += 1
            if len(self.buf_x) < self.replay:
                self.buf_x.append(x); self.buf_y.append(y)
            else:
                j = self.r.integers(0, self.seen)
                if j < self.replay: self.buf_x[j] = x; self.buf_y[j] = y
            self.update_pJ += cache_read_pJ(self.replay * self.d * 8) * np.ceil(self.d * 8 / 64)
        if self.ewc_lam:
            self.recentx.append(x); self.recenty.append(y); self.recentx = self.recentx[-200:]; self.recenty = self.recenty[-200:]

    def boundary(self):
        if not self.ewc_lam or not self.recentx: return
        X = np.array(self.recentx, np.float32); Y = np.array(self.recenty); F = [np.zeros_like(p) for p in self.ps]
        for i in range(len(Y)):
            a, hd, z = self._fwd(X[i:i+1]); z = z - z.max(); p = np.exp(z); p /= p.sum(); dz = p.copy(); dz[0, Y[i]] -= 1
            dh = (dz @ self.W2.T) * (a > 0)
            for k, gg in enumerate((X[i:i+1].T @ dh, dh.sum(0), hd.T @ dz, dz.sum(0))): F[k] += gg ** 2 / len(Y)
        if self.anchor is not None:
            F = [f + f0 for f, f0 in zip(F, self.anchor[1])]
        self.anchor = (self.ewc_lam, F, [p.copy() for p in self.ps])
        self.update_pJ += len(Y) * 3 * self.fwd * FMAC

    def mem_bits(self):
        return int(self.nparam * 32 * 3 + self.replay * self.d * 8)


# ------------------------------------------------------------------ nearest neighbours
class WindowKNN(Base):
    def __init__(self, K, window=1000, k=5, seed=0):
        self.K, self.window, self.k = K, window, k; self.cfg = dict(window=window, k=k)
        self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        self.d = Xw.shape[1]; self.X = np.zeros((self.window, self.d)); self.y = np.zeros(self.window, int); self.n = 0

    def answer(self, x):
        n = min(self.n, self.window); self.n_ans += 1
        tier = cache_read_pJ(self.window * self.d * 8)
        self.answer_pJ += n * self.d * 0.33 + tier * n * np.ceil(self.d * 8 / 64) + n * 0.1 + 10 * np.ceil(self.d * 8 / 64)
        if n == 0: return 0
        D = np.sum((self.X[:n] - x) ** 2, 1)
        nn = np.argpartition(D, min(self.k, n - 1))[: self.k]
        return int(np.bincount(self.y[:n][nn], minlength=self.K).argmax())

    def learn(self, x, y):
        i = self.n % self.window; self.X[i] = x; self.y[i] = y; self.n += 1
        self.update_pJ += cache_read_pJ(self.window * self.d * 8) * np.ceil(self.d * 8 / 64); self.n_upd += 1

    def mem_bits(self):
        return int(self.window * (self.d * 8 + 8))


# ------------------------------------------------------------------ Hoeffding tree with drift detector
class HTree(Base):
    """VFDT with Gaussian class-conditional estimators per feature; a DDM drift detector resets the tree
    (keeping a background tree trained since the warning level, as in common practice)."""
    def __init__(self, K, grace=200, delta=1e-5, tau=0.05, seed=0, max_depth=20, reuse=0, tier=0):
        self.K, self.grace, self.delta, self.tau, self.max_depth = K, grace, delta, tau, max_depth
        self.cfg = dict(grace=grace, delta=delta); self.reuse = reuse; self.tierfix = tier; self.tw = 10.0
        if reuse: self.cfg["reuse"] = reuse
        if tier: self.cfg["tier"] = tier
        self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        self.d = Xw.shape[1]; self.root = self._leaf(0); self.bg = None
        self.ddm_n = 0; self.ddm_p = 1.0; self.ddm_s = 0.0; self.ddm_min = np.inf; self.err = 0

    def _leaf(self, depth):
        return dict(leaf=True, depth=depth, n=np.zeros(self.K), mu=np.zeros((self.K, self.d)), m2=np.zeros((self.K, self.d)),
                    lo=np.full(self.d, np.inf), hi=np.full(self.d, -np.inf), since=0)

    def _find(self, node, x):
        while not node["leaf"]:
            node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
        return node

    def answer(self, x):
        leaf = self._find(self.root, x); self.n_ans += 1
        if self.tierfix and self.n_ans % 500 == 0: self.tw = cache_read_pJ(self.mem_bits())
        self.answer_pJ += (leaf["depth"] + 1) * (self.tw + 0.1) + 10 * np.ceil(self.d * 8 / 64) + self.K * 0.1
        self._last = int(leaf["n"].argmax()) if leaf["n"].sum() else 0
        return self._last

    def _update(self, root, x, y):
        leaf = self._find(root, x)
        n = leaf["n"]; n[y] += 1; dlt = x - leaf["mu"][y]; leaf["mu"][y] += dlt / n[y]; leaf["m2"][y] += dlt * (x - leaf["mu"][y])
        leaf["lo"] = np.minimum(leaf["lo"], x); leaf["hi"] = np.maximum(leaf["hi"], x); leaf["since"] += 1
        cost = (leaf["depth"] + 1) * (self.tw + 0.1) + self.d * 3 * 2 * self.tw / 8 + self.d * 6 * 0.9
        if leaf["since"] >= self.grace and leaf["depth"] < self.max_depth and (n > 0).sum() > 1:
            leaf["since"] = 0; cost += self._try_split(leaf)
        return cost

    def _try_split(self, leaf):
        from scipy.special import erf
        n = leaf["n"]; N = n.sum(); p = n / N
        H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        sd = np.sqrt(np.maximum(leaf["m2"] / np.maximum(n[:, None] - 1, 1), 1e-12))
        best = []
        for f in range(self.d):
            if leaf["hi"][f] <= leaf["lo"][f]: best.append((0, f, 0)); continue
            ts = np.linspace(leaf["lo"][f], leaf["hi"][f], 12)[1:-1]
            z = (ts[:, None] - leaf["mu"][None, :, f]) / (sd[None, :, f] * np.sqrt(2))
            left = n[None, :] * 0.5 * (1 + erf(z)); right = n[None, :] - left
            def ent(c):
                s_ = c.sum(1, keepdims=True); q = c / np.maximum(s_, 1e-12)
                return -np.sum(np.where(q > 0, q * np.log2(np.maximum(q, 1e-12)), 0), 1), s_[:, 0]
            el, sl = ent(left); er, sr = ent(right)
            gains = H - (sl / N) * el - (sr / N) * er
            i = int(np.argmax(gains)); best.append((float(gains[i]), f, float(ts[i])))
        best.sort(reverse=True)
        R = np.log2(self.K); eps = np.sqrt(R * R * np.log(1 / self.delta) / (2 * N))
        cost = self.d * 10 * self.K * 20 * 0.1
        if best[0][0] - best[1][0] > eps or eps < self.tau:
            g, f, t = best[0]
            d0 = leaf["depth"]
            leaf.clear(); leaf.update(leaf=False, f=f, t=t, depth=d0)
            leaf["l"] = self._leaf(d0 + 1); leaf["r"] = self._leaf(d0 + 1)
        return cost

    def _depth_of(self, node):
        return node.get("depth", 0)

    def learn(self, x, y):
        # DDM on the error stream
        e = int(self._last != y); self.ddm_n += 1
        self.ddm_p += (e - self.ddm_p) / self.ddm_n; self.ddm_s = np.sqrt(self.ddm_p * (1 - self.ddm_p) / self.ddm_n)
        d0 = self._find(self.root, x)["depth"] if self.reuse else 0
        cost = self._update(self.root, x, y) - d0 * (self.tw + 0.1)                    # reuse: path latched from the answer
        if self.ddm_n > 30:
            if self.ddm_p + self.ddm_s < self.ddm_min: self.ddm_min = self.ddm_p + self.ddm_s; self.ddm_smin = self.ddm_s; self.ddm_pmin = self.ddm_p
            lvl = self.ddm_p + self.ddm_s
            if lvl > self.ddm_pmin + 3 * self.ddm_smin:          # drift: switch to the background tree
                self.root = self.bg if self.bg is not None else self._leaf(0); self.bg = None
                self.ddm_n = 0; self.ddm_p = 1.0; self.ddm_min = np.inf
            elif lvl > self.ddm_pmin + 2 * self.ddm_smin:        # warning: grow a background tree
                if self.bg is None: self.bg = self._leaf(0)
            else:
                self.bg = None
        if self.bg is not None: cost += self._update(self.bg, x, y)
        self.update_pJ += cost; self.n_upd += 1

    def mem_bits(self):
        def count(nd):
            return 1 if nd["leaf"] else 1 + count(nd["l"]) + count(nd["r"])
        return int(count(self.root) * (self.K * self.d * 3 * 32))


class Majority(Base):
    def __init__(self, K, seed=0):
        self.K = K; self.cfg = {}; self.answer_pJ = 0.0; self.update_pJ = 0.0
    def start(self, Xw, K): self.c = np.zeros(K)
    def answer(self, x): self.n_ans += 1; self.answer_pJ += 10.1; return int(self.c.argmax())
    def learn(self, x, y): self.c[y] += 1; self.update_pJ += 10.1; self.n_upd += 1
    def mem_bits(self): return self.K * 32


# ------------------------------------------------------------------ river baselines (installed on GitHub if possible)
def _river():
    try:
        import river  # noqa
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "river"], check=False)
    import river
    return river


class RiverLearner(Base):
    """river's Hoeffding adaptive tree ('hat') or adaptive random forest ('arf'). Energy is priced like our
    Hoeffding tree: per answer, one word read per node on the path, per tree; per update, the path plus
    the leaf statistics, per tree (an estimate; river runs in Python).
    Sweep 3 (tier=1): reads are priced at the memory tier the model's size puts it in (as for every other
    learner), split attempts are priced (as for our tree), and reuse=1 latches the answer's path."""
    def __init__(self, K, model="hat", seed=0, tier=0, reuse=0, **kw):
        self.K, self.model, self.kw = K, model, kw; self.cfg = dict(model=model, **kw)
        if tier: self.cfg["tier"] = tier
        if reuse: self.cfg["reuse"] = reuse
        self.tierfix, self.reuse = tier, reuse
        self.answer_pJ = 0.0; self.update_pJ = 0.0; self.seed = seed

    def start(self, Xw, K):
        rv = _river(); self.d = Xw.shape[1]
        if self.model == "hat":
            self.m = rv.tree.HoeffdingAdaptiveTreeClassifier(seed=self.seed, **self.kw); self.trees = 1
        else:
            self.m = rv.forest.ARFClassifier(seed=self.seed, **self.kw); self.trees = self.kw.get("n_models", 10)
        self.depth = 4; self.leaves = 1; self.nodes = 1; self.tier = 10.0
        self.grace = self.kw.get("grace_period", 50 if self.model == "arf" else 200)

    def _x(self, x):
        return {i: float(v) for i, v in enumerate(x)}

    def _measure(self):
        try:
            ms = self.m.models if hasattr(self.m, "models") else [self.m]
            self.depth = float(np.mean([t.height for t in ms]))
            self.leaves = float(np.sum([t.n_leaves for t in ms])); self.nodes = float(np.sum([t.n_nodes for t in ms]))
        except Exception:
            pass
        if self.tierfix:
            self.tier = cache_read_pJ(self.mem_bits())

    def answer(self, x):
        p = self.m.predict_one(self._x(x)); self.n_ans += 1
        self.answer_pJ += self.trees * (self.depth + 1) * (self.tier + 0.1) + 10 * np.ceil(self.d * 8 / 64)
        return int(p) if p is not None else 0

    def learn(self, x, y):
        self.m.learn_one(self._x(x), int(y)); self.n_upd += 1
        if self.n_upd % 500 == 0 or self.n_upd == 50:
            self._measure()
        path = 0 if self.reuse else (self.depth + 1) * (self.tier + 0.1)
        stats = self.d * 3 * 2 * self.tier / 8 + self.d * 6 * 0.9           # class-y Gaussian stats of each feature, read + write
        split = (self.d * 10 * self.K * 20 * 0.1) / self.grace if self.tierfix else 0.0
        self.update_pJ += self.trees * (path + stats + split)

    def mem_bits(self):
        if not self.tierfix:
            return 0
        return int(self.leaves * self.K * self.d * 3 * 32 + self.nodes * 64)
