"""Round five development grids, generated on the runner (so job lists stay small).
spec 'grid:<family>:<stream>:<i>:<n>' -> the i-th of n chunks of that family's grid for that stream."""
import itertools
import patches  # noqa: runtime fixes (see patches.py)


def grid(family, stream):
    if family == "mycel":
        # M = tube slots per nucleus (0 = every input): learning cost scales with it, so the budget
        # (with regrowth of dead tubes toward active inputs) is a real lever here
        out = []
        for B, s in itertools.product((4, 8), (2.0, 8.0)):
            for P, Tm, cons, M in itertools.product((16, 64, 256), (0.5, 1.0), (0.0, 0.05), (0, 16, 32)):
                base = dict(tag="shared", P=P, B=B, s=s, T=max(2, int(Tm * P)), consolidate=cons, M=M, grow=int(M > 0))
                out.append(base)
                if stream == "splitdigits":
                    out.append(dict(base, bmode=1))
            for C, w, cons, M in itertools.product((8, 32), (0, 1), (0.0, 0.05), (0, 16)):
                out.append(dict(tag="perclass", C=C, B=B, s=s, w=w, T=(2 * C if w else None), consolidate=cons, M=M, grow=int(M > 0)))
        return out
    if family == "mycel2":
        # sweep 2: beyond every edge the best sweep-1 setting touched
        out = []
        for P, B, s, Tm, M in itertools.product((256, 512, 1024), (8, 12, 16), (1.5, 2.0, 3.0), (1.0, 2.0), (0, 8, 16)):
            out.append(dict(tag="shared", P=P, B=B, s=s, T=int(Tm * P), consolidate=0.0, M=M, grow=int(M > 0)))
        return out
    # sweep 2 for baselines too: beyond the edges their best sweep-1 settings touched
    if family == "mlp2":
        out = []
        for h, lr in itertools.product((64, 128, 256), (0.0003, 0.001, 0.003)):
            out += [dict(h=h, lr=lr), dict(h=h, lr=lr, replay=1000, k=4)]
        return out
    if family == "knn2":
        return [dict(window=w, k=k) for w in (1000, 2000, 4000) for k in (9, 15, 25)]
    if family == "htree2":
        return [dict(grace=g, delta=d) for g in (20, 50) for d in (1e-2, 1e-3)]
    if family == "mlp":
        out = []
        for h, lr in itertools.product((16, 64), (0.003, 0.01, 0.03)):
            out.append(dict(h=h, lr=lr))
            for rp, k in itertools.product((200, 1000), (1, 4)):
                out.append(dict(h=h, lr=lr, replay=rp, k=k))
            if stream == "splitdigits":
                out += [dict(h=h, lr=lr, ewc=e) for e in (10, 100, 1000)]
        return out
    if family == "knn":
        return [dict(window=w, k=k) for w in (200, 500, 1000, 2000) for k in (3, 9)]
    if family == "htree":
        return [dict(grace=g, delta=d) for g in (50, 200, 500) for d in (1e-3, 1e-7)]
    if family == "river":
        return [dict(model="hat", grace_period=g) for g in (50, 200)] + [dict(model="arf", n_models=n) for n in (5, 10)]
    return [{}]


def chunk(spec):
    _, fam, stream, i, n = spec.split(":")
    g = grid(fam, stream); i, n = int(i), int(n); k = (len(g) + n - 1) // n
    return g[i * k:(i + 1) * k]
