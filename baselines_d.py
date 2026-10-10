"""Digital baselines with full control: one-hidden-layer MLP (float32 and int8),
optionally magnitude-pruned, and logistic regression, all in numpy.
Same interface as MycelD: fit, predict, energy, kill, flip, retrain."""
import numpy as np
from dmycel import cache_read_pJ


class MLPD:
    def __init__(self, n_in, K, h=32, seed=0, epochs=300, lr=0.01, bits=32, prune=0.0, l2=1e-4):
        self.n_in, self.K, self.h, self.epochs, self.lr, self.bits, self.prune, self.l2 = n_in, K, h, epochs, lr, bits, prune, l2
        r = np.random.default_rng(seed); self.rng = r
        self.W1 = (r.normal(0, np.sqrt(2 / n_in), (n_in, h))).astype(np.float32); self.b1 = np.zeros(h, np.float32)
        self.W2 = (r.normal(0, np.sqrt(1 / h), (h, K))).astype(np.float32); self.b2 = np.zeros(K, np.float32)
        self.m1 = np.ones_like(self.W1); self.m2 = np.ones_like(self.W2)   # pruning masks
        self.alive = np.ones(h, np.float32)
        self.opt = None
        self.q = None

    # ------------------------------------------------------------- train (Adam, minibatch)
    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def fit(self, X, y, epochs=None, batch=32):
        X = X.astype(np.float32); Y = np.eye(self.K, dtype=np.float32)[y]
        if self.opt is None:
            self.opt = [[np.zeros_like(p), np.zeros_like(p)] for p in self._params()]; self.t = 0
        for ep in range(epochs or self.epochs):
            perm = self.rng.permutation(len(y))
            for s in range(0, len(y), batch):
                idx = perm[s:s + batch]
                x, t = X[idx], Y[idx]
                a = x @ (self.W1 * self.m1) + self.b1; hdn = np.maximum(a, 0) * self.alive
                z = hdn @ (self.W2 * self.m2) + self.b2; z -= z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True)
                dz = (p - t) / len(idx)
                g2 = hdn.T @ dz + self.l2 * self.W2; gb2 = dz.sum(0)
                dh = (dz @ (self.W2 * self.m2).T) * (a > 0) * self.alive
                g1 = x.T @ dh + self.l2 * self.W1; gb1 = dh.sum(0)
                if getattr(self, "ewc", None) is not None:          # elastic weight consolidation
                    lam, F, ref = self.ewc
                    g1 = g1 + lam * F[0] * (self.W1 - ref[0]); gb1 = gb1 + lam * F[1] * (self.b1 - ref[1])
                    g2 = g2 + lam * F[2] * (self.W2 - ref[2]); gb2 = gb2 + lam * F[3] * (self.b2 - ref[3])
                self.t += 1
                for prm, g, st in zip(self._params(), (g1 * self.m1, gb1, g2 * self.m2, gb2), self.opt):
                    st[0] = 0.9 * st[0] + 0.1 * g; st[1] = 0.999 * st[1] + 0.001 * g * g
                    mh = st[0] / (1 - 0.9 ** self.t); vh = st[1] / (1 - 0.999 ** self.t)
                    prm -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            if self.prune > 0 and ep == int(0.6 * (epochs or self.epochs)):
                self._prune()
        self.q = None
        if self.bits == 8:
            self.quantize()
        return self

    def _prune(self):
        for W, m in ((self.W1, self.m1), (self.W2, self.m2)):
            thr = np.quantile(np.abs(W * m)[m > 0], self.prune)
            m[np.abs(W) < thr] = 0

    # ------------------------------------------------------------- int8 storage
    def quantize(self):
        self.q = []
        for W in (self.W1 * self.m1, self.W2 * self.m2):
            sc = np.abs(W).max() / 127 + 1e-12
            self.q.append([np.clip(np.round(W / sc), -127, 127).astype(np.int8), np.float32(sc)])

    def weights(self):
        if self.q is not None:
            return [self.q[0][0].astype(np.float32) * self.q[0][1], self.q[1][0].astype(np.float32) * self.q[1][1]]
        return [self.W1 * self.m1, self.W2 * self.m2]

    def predict(self, X):
        W1, W2 = self.weights()
        with np.errstate(all="ignore"):
            hdn = np.maximum(X.astype(np.float32) @ W1 + self.b1, 0) * self.alive
            z = hdn @ W2 + self.b2
        z = np.where(np.isfinite(z), z, -np.inf)
        bad = ~np.isfinite(z).any(1)
        out = np.argmax(z, 1)
        out[bad] = -1                          # broken arithmetic counts as wrong
        return out

    # ------------------------------------------------------------- cost
    def n_weights(self):
        return int(self.m1[:, self.alive > 0].sum() + self.m2[self.alive > 0].sum())

    def model_bits(self):
        nz = self.n_weights()
        idx = 0 if self.prune == 0 else nz * int(np.ceil(np.log2(max(self.n_in, self.h, 2))))   # sparse indices
        return nz * self.bits + idx + (self.h + self.K) * 32

    def energy(self):
        nz = self.n_weights()
        read = cache_read_pJ(self.model_bits()) * self.model_bits() / 64
        mac = (3.7 + 0.9) if self.bits == 32 else (0.2 + 0.1)        # multiply + accumulate
        act = int(self.alive.sum()) * (0.9 if self.bits == 32 else 0.1) * 2   # bias add + ReLU compare
        if self.bits == 8:
            act += int(self.alive.sum()) * 3.1   # audit fix: rescale each hidden activation back to int8
        inp = 10.0 * np.ceil(self.n_in * self.bits / 64)              # load the input once (L1)
        return read + nz * mac + act + self.K * 0.1 + inp

    # ------------------------------------------------------------- damage
    def kill(self, frac, seed):
        r = np.random.default_rng(seed)
        self.alive[r.random(self.h) < frac] = 0

    def flip(self, rate, seed):
        r = np.random.default_rng(seed)
        if self.q is not None:
            for k in range(2):
                a = self.q[k][0].view(np.uint8)
                bits = np.unpackbits(a); bits ^= (r.random(bits.shape) < rate).astype(np.uint8)
                self.q[k][0] = np.packbits(bits).view(np.int8).reshape(self.q[k][0].shape)
        else:
            for W in (self.W1, self.W2):
                a = W.view(np.uint32)
                for b in range(32):
                    m = r.random(a.shape) < rate
                    a[m] ^= np.uint32(1 << b)

    def retrain(self, X, y, passes):
        self.fit(X, y, passes)


class LogRegD(MLPD):
    """Softmax regression = no hidden layer."""
    def __init__(self, n_in, K, seed=0, epochs=300, lr=0.01, bits=32):
        self.n_in, self.K, self.epochs, self.lr, self.bits = n_in, K, epochs, lr, bits
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((n_in, K), np.float32); self.b = np.zeros(K, np.float32); self.opt = None; self.q = None

    def fit(self, X, y, epochs=None, batch=32):
        X = X.astype(np.float32); Y = np.eye(self.K, dtype=np.float32)[y]
        if self.opt is None:
            self.opt = [[np.zeros_like(p), np.zeros_like(p)] for p in (self.W, self.b)]; self.t = 0
        for _ in range(epochs or self.epochs):
            perm = self.rng.permutation(len(y))
            for s in range(0, len(y), batch):
                idx = perm[s:s + batch]
                z = X[idx] @ self.W + self.b; z -= z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True); dz = (p - Y[idx]) / len(idx)
                self.t += 1
                for prm, g, st in zip((self.W, self.b), (X[idx].T @ dz, dz.sum(0)), self.opt):
                    st[0] = 0.9 * st[0] + 0.1 * g; st[1] = 0.999 * st[1] + 0.001 * g * g
                    prm -= self.lr * (st[0] / (1 - 0.9 ** self.t)) / (np.sqrt(st[1] / (1 - 0.999 ** self.t)) + 1e-8)
        return self

    def predict(self, X):
        with np.errstate(all="ignore"):
            z = X.astype(np.float32) @ self.W + self.b
        z = np.where(np.isfinite(z), z, -np.inf)
        out = np.argmax(z, 1); out[~np.isfinite(z).any(1)] = -1
        return out

    def model_bits(self):
        return self.W.size * 32 + self.K * 32

    def energy(self):
        return cache_read_pJ(self.model_bits()) * self.model_bits() / 64 + self.W.size * 4.6 + self.K * 0.1 + 10.0 * np.ceil(self.n_in * 32 / 64)


def _lr_flip(self, rate, seed):
    r = np.random.default_rng(seed)
    a = self.W.view(np.uint32)
    for b in range(32):
        m = r.random(a.shape) < rate
        a[m] ^= np.uint32(1 << b)
LogRegD.flip = _lr_flip


def ewc_anchor(m, X, y, lam):
    """Diagonal Fisher information on task A (per-example squared gradients), then anchor."""
    X = X.astype(np.float32); F = [np.zeros_like(p) for p in m._params()]
    for i in range(len(y)):
        x = X[i:i + 1]
        a = x @ (m.W1 * m.m1) + m.b1; h = np.maximum(a, 0) * m.alive
        z = h @ (m.W2 * m.m2) + m.b2; z -= z.max(); p = np.exp(z); p /= p.sum()
        dz = p.copy(); dz[0, y[i]] -= 1
        g2 = h.T @ dz; dh = (dz @ (m.W2 * m.m2).T) * (a > 0); g1 = x.T @ dh
        for k, g in enumerate((g1, dh.sum(0), g2, dz.sum(0))):
            F[k] += g * g / len(y)
    m.ewc = (lam, F, [p.copy() for p in m._params()])
