"""Grow-on-demand Mycel ('MycelG'), built from what the market scan showed sells:
  1. sensing layer: Mycel's thermometer ports (Encoder2 quantile thresholds, both rails), always on;
  2. readout: an integer perceptron over the positive-rail port bits plus the outputs of grown nuclei;
  3. nuclei grow only where errors persist: on a mistake, with probability g, a new AND-nucleus is grown
     from k port literals that are active on the misclassified example (a hypha extending toward unmet
     demand). When the budget is full, the weakest nucleus (smallest total readout weight) is recycled.
Nuclei are fixed once grown (no Tsetlin feedback); only the readout learns. Priced with energy model v2:
per-store tiers, bytes as bytes, bit tests 0.1 pJ, int16 adds 0.05 pJ, random draws 0.6 pJ."""
import numpy as np
import learners_v2 as V


class MycelG(V.Base):
    def __init__(self, K, B=8, k=2, Pmax=256, g=0.5, chem=0, seed=0):
        self.K, self.B, self.k, self.Pmax, self.g, self.chem = K, B, k, Pmax, g, chem
        self.r = np.random.default_rng(seed); self.cfg = dict(B=B, k=k, Pmax=Pmax, g=g, chem=chem)
        self.answer_pJ = 0.0; self.update_pJ = 0.0; self.grown = 0

    def start(self, Xw, K):
        self.enc = V.Encoder2(self.B).fit(Xw); self.L = self.enc.L; self.h = self.L // 2; self.d = Xw.shape[1]
        self.lits = np.zeros((self.Pmax, self.k), np.int32); self.alive = np.zeros(self.Pmax, bool)
        self.nf = self.h + self.Pmax + 1                                # readout inputs: ports, nuclei, bias
        self.W = np.zeros((self.nf, K), np.int32)
        self.S = np.zeros((self.L, K), np.int32)                         # chemotaxis: error trail per literal and class
        self._tiers()

    def _tiers(self):
        self.t_read = V.cache_read_pJ(self.nf * self.K * 16)            # readout weights
        self.t_nuc = V.cache_read_pJ(self.Pmax * self.k * 16)           # nucleus literal indices

    def answer(self, x):
        b = self.enc.bits(x[None, :])[0].astype(bool)
        na = int(self.alive.sum())
        fired = np.zeros(self.Pmax, bool)
        if na:
            idx = np.flatnonzero(self.alive); fired[idx] = b[self.lits[idx]].all(1)
        f = np.concatenate([b[:self.h], fired, [True]]); nz = np.flatnonzero(f)
        s = self.W[nz].sum(0)
        e = self.h * 0.1 + 10 * V.words(self.h * 8) + 10 * V.words(self.d * 8)          # ports: compares + thresholds + input
        e += na * self.k * 0.1 + self.t_nuc * V.words(na * self.k * 16)                 # nuclei: AND over k latched bits
        e += self.t_read * V.words(len(nz) * self.K * 16) + len(nz) * self.K * 0.05      # readout: binary inputs, adds only
        self.answer_pJ += e; self.n_ans += 1
        self._c = (b, nz); self._p = int(np.argmax(s)); return self._p

    def learn(self, x, y):
        self.n_upd += 1; self.update_pJ += self.K * 0.05
        if self._p == y:
            return
        b, nz = self._c
        self.W[nz, y] += 1; self.W[nz, self._p] -= 1
        self.update_pJ += 2 * (self.t_read * V.words(len(nz) * 16) * 2 + len(nz) * 0.05)
        self.update_pJ += 0.6
        if self.chem:                                                    # lay an error trail on the active literals
            act = np.flatnonzero(b); self.S[act, y] += 1; self.S[act, self._p] -= 1
            self.update_pJ += 2 * (V.cache_read_pJ(self.L * self.K * 16) * V.words(len(act) * 16) * 2 + len(act) * 0.05)
        if self.r.random() < self.g:                                     # grow toward the error
            act = np.flatnonzero(b)
            if len(act) < self.k:
                return
            free = np.flatnonzero(~self.alive)
            if len(free):
                j = int(free[0])
            else:                                                        # recycle the weakest nucleus
                strength = np.abs(self.W[self.h:self.h + self.Pmax]).sum(1)
                j = int(np.argmin(strength))
                self.update_pJ += self.t_read * V.words(self.Pmax * self.K * 16) + self.Pmax * self.K * 0.05
                self.W[self.h + j] = 0
                self.update_pJ += self.t_read * V.words(self.K * 16)
            if self.chem:                                                # follow the trail: sample literals by error pull
                w = np.maximum(self.S[act, y] - self.S[act, self._p], 0).astype(float) + 1.0
                self.lits[j] = self.r.choice(act, self.k, replace=False, p=w / w.sum())
                self.update_pJ += V.cache_read_pJ(self.L * self.K * 16) * V.words(len(act) * 2 * 16) + len(act) * 0.3
            else:
                self.lits[j] = self.r.choice(act, self.k, replace=False)
            self.alive[j] = True; self.grown += 1
            self.update_pJ += self.k * 0.6 + self.t_nuc * V.words(self.k * 16) + 10 * V.words(self.L)

    def mem_bits(self):
        return int(self.nf * self.K * 16 + self.Pmax * self.k * 16 + self.h * 8 + (self.L * self.K * 16 if self.chem else 0))


VARIANTS = {"mycelG2": (MycelG, dict(k=2, Pmax=256)), "mycelG3": (MycelG, dict(k=3, Pmax=512)),
            "mycelGc2": (MycelG, dict(k=2, Pmax=256, g=1.0, chem=1)), "mycelGc3": (MycelG, dict(k=3, Pmax=512, g=1.0, chem=1))}
