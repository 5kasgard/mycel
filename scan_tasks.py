"""Market scan task pool. Each loader(seed) returns dict(X, y, K, changes, recur, tasks) like streams.py.
TAGS (registered before any scan result was seen):
  tag 'decomp' = the input is a composite artefact broken down into parts or tokens, and the label is
  about which recurring sub-structures are present (the digital analogue of decomposition):
  log lines -> templates, documents -> topics, DNA -> splice motifs, memory traces -> dead objects,
  aggregate power -> appliance, web pages / e-mails -> indicator tokens.  Everything else 'other'.
temporal = keep the natural order (a real stream or a drifting generator); otherwise shuffled per seed.
Real data loads on the GitHub runner (Loghub via git, OpenML and 20 Newsgroups via scikit-learn)."""
import hashlib, os, re, subprocess, sys
import numpy as np
import streams

NMAX = 10000
CACHE = os.environ.get("SCAN_CACHE", os.path.expanduser("~/scan_cache"))


# ------------------------------------------------------------------ helpers
def _h(tok, mod):
    return int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % mod


def _pack(X, y, K, temporal, seed, changes=(), recur=()):
    X = np.asarray(X, float); y = np.asarray(y, int)
    if not temporal:
        idx = np.random.default_rng(seed).permutation(len(y)); X, y = X[idx], y[idx]
    X, y = X[:NMAX], y[:NMAX]
    return dict(X=X, y=y, K=int(K), changes=[c for c in changes if c < len(y)], recur=list(recur), tasks=None)


def _codes(labels):
    u, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return inv, len(u)


# ------------------------------------------------------------------ new synthetic generators
def led(seed, n=20000, seg=5000, noise=0.1):
    r = np.random.default_rng(seed)
    seg7 = np.array([[1,1,1,0,1,1,1],[0,0,1,0,0,1,0],[1,0,1,1,1,0,1],[1,0,1,1,0,1,1],[0,1,1,1,0,1,0],
                     [1,1,0,1,0,1,1],[1,1,0,1,1,1,1],[1,0,1,0,0,1,0],[1,1,1,1,1,1,1],[1,1,1,1,0,1,1]])
    y = r.integers(0, 10, n); X = np.zeros((n, 24))
    for k in range(n // seg):                                  # drift: the 7 segment wires move among the 24 inputs
        pos = r.choice(24, 7, replace=False); sl = slice(k * seg, (k + 1) * seg)
        blk = (r.random((seg, 24)) < 0.5).astype(float); blk[:, pos] = seg7[y[sl]]; X[sl] = blk
    X = np.where(r.random(X.shape) < noise, 1 - X, X)
    return dict(X=X, y=y, K=10, changes=[seg * i for i in range(1, n // seg)], recur=[], tasks=None)


def dnf(seed, n=20000, seg=5000, d=64, terms=4, width=3, noise=0.02):
    r = np.random.default_rng(seed)
    X = (r.random((n, d)) < 0.3).astype(float); y = np.zeros(n, int); rules = []
    for k in range(n // seg):
        rule = rules[0] if k == 3 else [(r.choice(d, width, replace=False), r.integers(0, 2, width)) for _ in range(terms)]
        rules.append(rule); sl = slice(k * seg, (k + 1) * seg)
        hit = np.zeros(seg, bool)
        for idx, pol in rule:
            hit |= np.all(X[sl][:, idx] == pol, axis=1)
        y[sl] = hit
    y = streams._flip(y, noise, r)
    return dict(X=X, y=y, K=2, changes=[seg, 2 * seg, 3 * seg], recur=[(0, 3 * seg)], tasks=None)


def parity(seed, n=10000, d=16, k=3, noise=0.05):
    r = np.random.default_rng(seed)
    X = (r.random((n, d)) < 0.5).astype(float); idx = r.choice(d, k, replace=False)
    y = streams._flip((X[:, idx].sum(1) % 2).astype(int), noise, r)
    return dict(X=X, y=y, K=2, changes=[], recur=[], tasks=None)


def cachereuse(seed, n=20000, seg=5000, npc=16, horizon=1024):
    """Dead-object detection: will the line touched now be touched again within `horizon` accesses?
    A synthetic program of 16 instructions (PCs), each with an access habit: streaming (never reused),
    small loop (reused soon), large random (rarely), pointer chase over a medium set. Habits reshuffle
    every `seg` accesses (phase change). Features: PC one-hot, last two PCs one-hot, 8 low address bits."""
    r = np.random.default_rng(seed)
    habits = ["stream", "loop", "rand", "chase"]
    pcs, addrs = np.zeros(n, int), np.zeros(n, np.int64)
    ptr = {p: 0 for p in range(npc)}; base = {p: (p + 1) << 20 for p in range(npc)}
    for k in range(n // seg):
        hab = r.choice(habits, npc)
        size = {p: int(r.choice([8, 32, 128])) for p in range(npc)}
        for i in range(k * seg, (k + 1) * seg):
            p = int(r.integers(0, npc)) if r.random() < 0.3 else (pcs[i - 1] + 1) % npc if i else 0
            pcs[i] = p; h = hab[p]
            if h == "stream": ptr[p] += 1; a = base[p] + ptr[p] + (k << 30)
            elif h == "loop": ptr[p] = (ptr[p] + 1) % size[p]; a = base[p] + ptr[p]
            elif h == "rand": a = base[p] + int(r.integers(0, 1 << 12))
            else: ptr[p] = (ptr[p] * 5 + 1) % 64; a = base[p] + ptr[p]
            addrs[i] = a
    nxt = {}; y = np.zeros(n, int)
    for i in range(n - 1, -1, -1):
        j = nxt.get(addrs[i]); y[i] = int(j is not None and j - i <= horizon); nxt[addrs[i]] = i
    X = np.zeros((n, 3 * npc + 8))
    X[np.arange(n), pcs] = 1
    X[np.arange(1, n), npc + pcs[:-1]] = 1; X[np.arange(2, n), 2 * npc + pcs[:-2]] = 1
    X[:, 3 * npc:] = (addrs[:, None] >> np.arange(8)) & 1
    return dict(X=X, y=y, K=2, changes=[seg * i for i in range(1, n // seg)], recur=[], tasks=None)


def nilm(seed, n=20000, napp=6, win=8):
    """Aggregate power -> is the fridge (appliance 0, a cycling load) on? Six appliances with on/off
    habits and noisy draw; features are the last `win` aggregate readings and their differences."""
    r = np.random.default_rng(seed)
    draw = np.array([150, 2000, 800, 60, 1200, 300.0]); on = np.zeros(napp, bool); agg = np.zeros(n); y = np.zeros(n, int)
    p_on = np.array([0.05, 0.003, 0.01, 0.02, 0.004, 0.008]); p_off = np.array([0.05, 0.05, 0.03, 0.01, 0.08, 0.02])
    for i in range(n):
        flip = np.where(on, r.random(napp) < p_off, r.random(napp) < p_on); on ^= flip
        agg[i] = (draw * on * r.normal(1, 0.05, napp)).sum() + r.normal(0, 20); y[i] = on[0]
    X = np.zeros((n, 2 * win))
    for j in range(win):
        X[j:, j] = agg[: n - j] if j else agg
    X[:, win:] = X[:, :win] - np.roll(X[:, :win], -1, axis=1)
    X = X / 4000.0
    return dict(X=X, y=y, K=2, changes=[], recur=[], tasks=None)


# ------------------------------------------------------------------ real data
LOGS = ["HDFS", "BGL", "Thunderbird", "Apache", "OpenSSH", "Linux", "Zookeeper", "Spark", "Hadoop", "HPC",
        "Windows", "Mac", "Android", "HealthApp", "Proxifier", "OpenStack"]


def _loghub():
    path = os.path.join(CACHE, "loghub")
    if not os.path.isdir(path):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/logpai/loghub", path], check=True)
    return path


def logs(name):
    def load(seed):
        import csv
        f = os.path.join(_loghub(), name, f"{name}_2k.log_structured.csv")
        rows = list(csv.DictReader(open(f, encoding="utf-8", errors="replace")))
        content = [r.get("Content", "") for r in rows]; ev = np.array([r.get("EventId", "") for r in rows])
        u, c = np.unique(ev, return_counts=True); rare = set(u[c < 5]); ev = np.array(["rare" if e in rare else e for e in ev])
        y, K = _codes(ev)
        X = np.zeros((len(rows), 640))
        for i, line in enumerate(content):
            toks = ["<*>" if re.search(r"\d", t) else t for t in re.split(r"[\s=:,\[\]\(\)\"']+", line) if t]
            for t in toks: X[i, _h(t, 512)] = 1
            for p, t in enumerate(toks[:3]): X[i, 512 + _h(f"{p}:{t}", 128)] = 1
        return _pack(X, y, K, True, seed)
    return load


def news(seed):
    from sklearn.datasets import fetch_20newsgroups
    cats = ["comp.graphics", "rec.autos", "sci.med", "talk.politics.misc", "soc.religion.christian"]
    d = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"), data_home=CACHE)
    X = np.zeros((len(d.data), 1024))
    for i, doc in enumerate(d.data):
        for t in set(re.findall(r"[a-z]{3,}", doc.lower())): X[i, _h(t, 1024)] = 1
    return _pack(X, d.target, len(cats), False, seed)


def openml(did):
    def load(seed):
        try:
            import pandas  # noqa
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas"], check=False)
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=did, as_frame=True, parser="auto", data_home=CACHE)
        df = d.data; cols = []
        for c in df.columns:
            s = df[c]
            if str(s.dtype) in ("category", "object", "bool", "string"):
                s = s.astype(str); lv = s.value_counts().index[:20]
                for v in lv: cols.append((s == v).to_numpy(float))
            else:
                v = s.to_numpy(float); med = np.nanmedian(v[:200]) if np.isfinite(v[:200]).any() else 0.0
                v = np.where(np.isfinite(v), v, med); cols.append(v)
        X = np.column_stack(cols); y, K = _codes(d.target)
        temporal = did in (150, 151)
        S = _pack(X, y, K, temporal, seed)
        X = S["X"]; lo, hi = X[:200].min(0), X[:200].max(0)
        S["X"] = np.clip((X - lo) / np.maximum(hi - lo, 1e-12), -0.5, 1.5)
        return S
    return load


OPENML = {"kr-vs-kp": 3, "letter": 6, "mushroom": 24, "optdigits": 28, "credit-g": 31, "pendigits": 32,
          "segment": 36, "diabetes": 37, "spambase": 44, "splice": 46, "tic-tac-toe": 50, "vehicle": 54,
          "waveform": 60, "covertype": 150, "electricity": 151, "satimage": 182, "monks1": 333, "monks2": 334,
          "monks3": 335, "kc1": 1067, "magic": 1120, "bank": 1461, "eeg": 1471, "phoneme": 1489, "adult": 1590,
          "phishing": 4534, "higgs": 23512}

TASKS, TAG, TEMPORAL = {}, {}, {}


def _reg(name, fn, tag, temporal):
    TASKS[name] = fn; TAG[name] = tag; TEMPORAL[name] = temporal


for nm in ["sea", "hyperplane", "stagger", "rbfdrift", "agrawal", "seanoisy", "splitdigits", "permdigits"]:
    _reg("syn_" + nm, (lambda f: (lambda seed: f(seed)))(getattr(streams, nm)), "other", True)
_reg("syn_led", led, "other", True); _reg("syn_dnf", dnf, "other", True); _reg("syn_parity", parity, "other", False)
_reg("syn_cachereuse", cachereuse, "decomp", True); _reg("syn_nilm", nilm, "decomp", True)
for nm in LOGS:
    _reg("log_" + nm, logs(nm), "decomp", True)
_reg("news5", news, "decomp", False)
for nm, did in OPENML.items():
    _reg("oml_" + nm, openml(did), "decomp" if nm in ("spambase", "splice", "phishing") else "other", did in (150, 151))


def half(name):
    """Pre-registered split: within each tag, tasks sorted by md5(name); alternate discovery/confirm."""
    same = sorted([t for t in TASKS if TAG[t] == TAG[name]], key=lambda t: hashlib.md5(t.encode()).hexdigest())
    return "discovery" if same.index(name) % 2 == 0 else "confirm"


def get(name, seed):
    S = TASKS[name](seed)
    if len(S["y"]) > NMAX:
        S["X"], S["y"] = S["X"][:NMAX], S["y"][:NMAX]
        S["changes"] = [c for c in S["changes"] if c < NMAX]; S["recur"] = [(a, b) for a, b in S["recur"] if b < NMAX]
    if S.get("tasks") is not None:
        S["tasks"] = None                                      # retention is not part of the market scan
    return S
