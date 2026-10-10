/* Mycel-D core: nuclei as bit-masks, tubes as 8-bit saturating counters.

   Layout (all arrays owned by Python/numpy, passed as pointers):
     lit  uint16 [K][C][M]   which literal each tube slot listens to
     cnt  uint8  [K][C][M]   tube counter; tube is open when cnt >= 128
     inc  uint64 [K][C][W]   bitmask of open tubes (what answering reads)
     has  uint64 [K][C][W]   bitmask of literals that own a slot (no duplicates)
     alive uint8 [K][C]      0 = dead nucleus (never fires)
     frz  uint8  [K][C][M]   1 = tube consolidated (learning cannot move it)
   Nucleus j of a contest votes YES if j < C/2, NO otherwise.
   A nucleus fires when every open tube sees a 1 (coincidence detection).
*/
#include <stdint.h>
#include <string.h>

static uint64_t rs = 88172645463325252ULL;
static inline uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint64_t g_rng = 0;
static inline uint32_t r32(void) { g_rng++; return (uint32_t)(xr() >> 32); }
void d_seed(uint64_t s) { rs = s * 2654435761ULL + 0x9E3779B97F4A7C15ULL; if (!rs) rs = 1; for (int i = 0; i < 10; i++) xr(); }

/* fire(): does nucleus fire on x?  empty_val: what an empty nucleus returns */
/* quorum (percent): 100 = every open tube must see a 1 (AND, a coincidence detector);
   below 100 = fire when at least that share of open tubes see a 1 (quorum sensing).
   With dual-rail thermometer inputs a quorum nucleus draws a slanted boundary. */
static int g_q = 100;
/* work counters, so learning can be priced as well as answering */
static uint64_t g_eval = 0, g_visit = 0, g_bump = 0;
/* energy model v2: weight reads/writes in feedback, mask-bit flips, slot regrowth, regrowth scans */
static uint64_t g_sel = 0, g_wupd = 0, g_flip = 0, g_regrow = 0, g_scan = 0;
void d_counters(uint64_t *out) { out[0] = g_eval; out[1] = g_visit; out[2] = g_bump; }
void d_counters2(uint64_t *out) { out[0] = g_rng; out[1] = g_sel; out[2] = g_wupd; out[3] = g_flip; out[4] = g_regrow; out[5] = g_scan; }
void d_reset_counters(void) { g_eval = g_visit = g_bump = 0; g_rng = g_sel = g_wupd = g_flip = g_regrow = g_scan = 0; }
void d_set_quorum(int q) { g_q = q; }
static inline int fire(const uint64_t *inc, const uint64_t *x, int W, int empty_val) {
    g_eval++;
    if (g_q >= 100) {
        uint64_t any = 0, viol = 0;
        for (int w = 0; w < W; w++) { any |= inc[w]; viol |= inc[w] & ~x[w]; }
        if (!any) return empty_val;
        return viol == 0;
    }
    int a = 0, n = 0;
    for (int w = 0; w < W; w++) { n += __builtin_popcountll(inc[w]); a += __builtin_popcountll(inc[w] & x[w]); }
    if (!n) return empty_val;
    return a * 100 >= g_q * n;
}

/* wgt: per-nucleus vote weight (output tube diameter); all ones = plain counting */
static inline int vote(int c, int K, int C, int W, const uint64_t *inc, const uint8_t *alive,
                       const uint8_t *wgt, const uint64_t *x, int empty_val) {
    int v = 0;
    for (int j = 0; j < C; j++) {
        if (!alive[c * C + j]) continue;
        int f = fire(inc + ((size_t)c * C + j) * W, x, W, empty_val) * wgt[c * C + j];
        v += (j < C / 2) ? f : -f;
    }
    return v;
}

void d_predict(int n, int K, int C, int W, const uint64_t *inc, const uint8_t *alive, const uint8_t *wgt,
               const uint64_t *X, int32_t *votes_out) {
    for (int i = 0; i < n; i++)
        for (int c = 0; c < K; c++)
            votes_out[i * K + c] = vote(c, K, C, W, inc, alive, wgt, X + (size_t)i * W, 0);
}

static inline void set_bit(uint64_t *m, int b, int on) {
    if (on) m[b >> 6] |= (1ULL << (b & 63)); else m[b >> 6] &= ~(1ULL << (b & 63));
}
static inline int get_bit(const uint64_t *m, int b) { return (m[b >> 6] >> (b & 63)) & 1; }

/* bump a tube counter by d (+1/-1), keep mask in step */
static inline void bump(uint8_t *cnt, uint64_t *inc, int lit, int d) {
    int old = *cnt, nw = old + d;
    g_bump++;
    if (nw < 0) nw = 0; if (nw > 255) nw = 255;
    *cnt = (uint8_t)nw;
    if ((old >= 128) != (nw >= 128)) { set_bit(inc, lit, nw >= 128); g_flip++; }
}

/* One pass over examples in the given order.
   T: vote clip (bounded feedback); s: forgetting rate (1/s per step)
   L: number of literals; M: tube slots per nucleus
   grow: 1 = recycle dead tubes toward active literals (budgeted growth) */
void d_epoch(int n, int K, int C, int M, int W, int L,
             uint16_t *lit, uint8_t *cnt, uint64_t *inc, uint64_t *has, const uint8_t *alive,
             const uint8_t *frz, uint8_t *wgt, int weighted, const uint64_t *X, const uint8_t *Xb, const int32_t *y,
             const int32_t *order, int T, double s, int grow) {
    uint32_t p_forget = (uint32_t)(4294967295.0 / s);
    int act[4096];
    for (int t = 0; t < n; t++) {
        int i = order[t];
        const uint64_t *x = X + (size_t)i * W;
        const uint8_t *xb = Xb + (size_t)i * L;
        int yi = y[i];
        int other = (int)(r32() % (uint32_t)(K - 1)); if (other >= yi) other++;
        for (int pass = 0; pass < 2; pass++) {
            int c = pass == 0 ? yi : other;
            int v = vote(c, K, C, W, inc, alive, wgt, x, 1);
            if (v > T) v = T; if (v < -T) v = -T;
            /* probability this contest's nuclei get feedback: bounded by the margin */
            double pf = pass == 0 ? (double)(T - v) / (2.0 * T) : (double)(T + v) / (2.0 * T);
            uint32_t P = (uint32_t)(pf * 4294967295.0);
            for (int j = 0; j < C; j++) {
                size_t nj = (size_t)c * C + j;
                if (!alive[nj]) continue;
                if (r32() > P) continue;
                int yes = j < C / 2;
                int typeI = (pass == 0) ? yes : !yes;   /* reinforce pattern vs block false alarm */
                uint64_t *in = inc + nj * W, *hs = has + nj * W;
                uint16_t *lt = lit + nj * M; uint8_t *ct = cnt + nj * M;
                const uint8_t *fz = frz + nj * M;
                int f = fire(in, x, W, 1);
                if (weighted && f && alive[nj] != 2) {
                    if (typeI) { if (wgt[nj] < 255) wgt[nj]++; }
                    else if (wgt[nj] > 1) wgt[nj]--;
                }
                if (typeI) {
                    if (f) {
                        int nact = -1;   /* lazily gathered active literals not owned yet */
                        for (int m = 0; m < M; m++) {
                            g_visit++;
                            if (fz[m]) continue;
                            int l = lt[m];
                            if (l != 0xFFFF) {
                                if (xb[l]) bump(&ct[m], in, l, +1);
                                else if (r32() < p_forget) bump(&ct[m], in, l, -1);
                            }
                            if (grow && (l == 0xFFFF || ct[m] == 0)) {
                                /* dead tube: regrow toward a literal active now (where the drive is) */
                                if (nact < 0) {
                                    nact = 0; g_visit += L;   /* scanning the input for regrowth (priced harshly: one slot visit per literal) */
                                    for (int q = 0; q < L && nact < 4096; q++)
                                        if (xb[q] && !get_bit(hs, q)) act[nact++] = q;
                                }
                                if (nact > 0) {
                                    int k = (int)(r32() % (uint32_t)nact);
                                    int nl = act[k]; act[k] = act[--nact];
                                    if (l != 0xFFFF) set_bit(hs, l, 0); set_bit(hs, nl, 1);
                                    lt[m] = (uint16_t)nl; ct[m] = 127;
                                }
                            }
                        }
                    } else {
                        for (int m = 0; m < M; m++)
                            if (++g_visit && !fz[m] && lt[m] != 0xFFFF && r32() < p_forget) bump(&ct[m], in, lt[m], -1);
                    }
                } else if (f) {
                    /* type II: open a tube on a literal that is 0 now, so this false alarm stops */
                    for (int m = 0; m < M; m++) {
                        g_visit++;
                        if (fz[m]) continue;
                        int l = lt[m];
                        if (l != 0xFFFF && !xb[l] && ct[m] < 128) bump(&ct[m], in, l, +1);
                    }
                }
            }
        }
    }
}

/* rebuild masks from slots (after damage or edits from Python) */
void d_rebuild(int K, int C, int M, int W, const uint16_t *lit, const uint8_t *cnt, uint64_t *inc, uint64_t *has) {
    memset(inc, 0, sizeof(uint64_t) * (size_t)K * C * W);
    memset(has, 0, sizeof(uint64_t) * (size_t)K * C * W);
    for (size_t nj = 0; nj < (size_t)K * C; nj++)
        for (int m = 0; m < M; m++) {
            int l = lit[nj * M + m];
            if (l == 0xFFFF) continue;     /* removed tube */
            set_bit(has + nj * W, l, 1);
            if (cnt[nj * M + m] >= 128) set_bit(inc + nj * W, l, 1);
        }
}

/* ---------------- shared colony: one pool of P nuclei, every contest reads it with signed weights
   wgt8 int8 [K][P]. A nucleus that fires sends its weight to each contest (event-driven:
   silent nuclei cost nothing downstream). */
static inline int vote_co(int c, int P_, int W, const uint64_t *inc, const uint8_t *alive,
                          const int8_t *w8, const uint8_t *fired) {
    int v = 0;
    for (int j = 0; j < P_; j++) if (fired[j]) v += w8[c * P_ + j];
    return v;
}

void co_fire(int P_, int W, const uint64_t *inc, const uint8_t *alive, const uint64_t *x, int empty_val, uint8_t *fired) {
    for (int j = 0; j < P_; j++) fired[j] = alive[j] ? fire(inc + (size_t)j * W, x, W, empty_val) : 0;
}

void co_predict(int n, int K, int P_, int W, const uint64_t *inc, const uint8_t *alive, const int8_t *w8,
                const uint64_t *X, int32_t *votes_out, int32_t *nfired) {
    uint8_t fired[65536];
    for (int i = 0; i < n; i++) {
        co_fire(P_, W, inc, alive, X + (size_t)i * W, 0, fired);
        int nf = 0; for (int j = 0; j < P_; j++) nf += fired[j];
        nfired[i] = nf;
        for (int c = 0; c < K; c++) votes_out[i * K + c] = vote_co(c, P_, W, inc, alive, w8, fired);
    }
}

/* Learning for quorum nuclei (binary synapses moved by counters, like a perceptron with
   two-state synapses).  d = what the nucleus should have done, f = what it did. */
static inline void quorum_update(uint64_t *in, uint16_t *lt, uint8_t *ct, const uint8_t *fz, int M,
                                 const uint8_t *xb, int f, int d, uint32_t p_forget) {
    for (int m = 0; m < M; m++) {
        if (fz[m]) continue;
        int l = lt[m]; if (l == 0xFFFF) continue;
        int on = xb[l];
        if (d && !f) bump(&ct[m], in, l, on ? +1 : -1);
        else if (!d && f) bump(&ct[m], in, l, on ? -1 : +1);
        else if (r32() < p_forget) bump(&ct[m], in, l, (d == on) ? +1 : -1);   /* slow consolidation of a right answer */
    }
}

void co_epoch(int n, int K, int P_, int M, int W, int L,
              uint16_t *lit, uint8_t *cnt, uint64_t *inc, uint64_t *has, const uint8_t *alive,
              const uint8_t *frz, int8_t *w8, const uint64_t *X, const uint8_t *Xb, const int32_t *y,
              const int32_t *order, int T, double s, int grow) {
    uint32_t p_forget = (uint32_t)(4294967295.0 / s);
    uint8_t fired[65536];
    int act[4096];
    for (int t = 0; t < n; t++) {
        int i = order[t];
        const uint64_t *x = X + (size_t)i * W;
        const uint8_t *xb = Xb + (size_t)i * L;
        int yi = y[i];
        int other = (int)(r32() % (uint32_t)(K - 1)); if (other >= yi) other++;
        co_fire(P_, W, inc, alive, x, 1, fired);
        for (int pass = 0; pass < 2; pass++) {
            int c = pass == 0 ? yi : other;
            int v = vote_co(c, P_, W, inc, alive, w8, fired);
            if (v > T) v = T; if (v < -T) v = -T;
            double pf = pass == 0 ? (double)(T - v) / (2.0 * T) : (double)(T + v) / (2.0 * T);
            uint32_t Pp = (uint32_t)(pf * 4294967295.0);
            for (int j = 0; j < P_; j++) {
                if (!alive[j]) continue;
                if (r32() > Pp) continue;
                int8_t *wj = &w8[c * P_ + j];
                g_sel++;
                int pos = *wj >= 0;
                int typeI = pass == 0 ? pos : !pos;
                int f = fired[j];
                if (f && alive[j] != 2) {           /* weight moves toward the right answer */
                    if (pass == 0) { if (*wj < 127) { (*wj)++; g_wupd++; } }
                    else { if (*wj > -127) { (*wj)--; g_wupd++; } }
                }
                uint64_t *in = inc + (size_t)j * W, *hs = has + (size_t)j * W;
                uint16_t *lt = lit + (size_t)j * M; uint8_t *ct = cnt + (size_t)j * M;
                const uint8_t *fz = frz + (size_t)j * M;
                if (g_q < 100) {
                    int want = pass == 0 ? pos : !pos;
                    quorum_update(in, lt, ct, fz, M, xb, f, want, p_forget);
                    continue;
                }
                if (typeI) {
                    if (f) {
                        int nact = -1;
                        for (int m = 0; m < M; m++) {
                            g_visit++;
                            if (fz[m]) continue;
                            int l = lt[m];
                            if (l != 0xFFFF) {
                                if (xb[l]) bump(&ct[m], in, l, +1);
                                else if (r32() < p_forget) bump(&ct[m], in, l, -1);
                            }
                            if (grow && (l == 0xFFFF || ct[m] == 0)) {
                                if (nact < 0) { nact = 0; g_scan += L; for (int q = 0; q < L && nact < 4096; q++) if (xb[q] && !get_bit(hs, q)) act[nact++] = q; }
                                if (nact > 0) {
                                    int k = (int)(r32() % (uint32_t)nact); int nl = act[k]; act[k] = act[--nact];
                                    if (l != 0xFFFF) set_bit(hs, l, 0); set_bit(hs, nl, 1);
                                    lt[m] = (uint16_t)nl; ct[m] = 127; g_regrow++;
                                }
                            }
                        }
                    } else {
                        for (int m = 0; m < M; m++)
                            if (++g_visit && !fz[m] && lt[m] != 0xFFFF && r32() < p_forget) bump(&ct[m], in, lt[m], -1);
                    }
                } else if (f) {
                    for (int m = 0; m < M; m++) {
                        g_visit++;
                        if (fz[m]) continue;
                        int l = lt[m];
                        if (l != 0xFFFF && !xb[l] && ct[m] < 128) bump(&ct[m], in, l, +1);
                    }
                }
            }
        }
    }
}
