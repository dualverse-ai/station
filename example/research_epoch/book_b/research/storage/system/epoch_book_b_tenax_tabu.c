
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAXK 128
#define MAXVARS 256
#define MAX_MOTIFS 20
#define MAX_VIOLATIONS 256

typedef struct {
    int found;
    int n;
    int k;
    int best_score;
    int best_linear_score;
    int best_deg_u;
    int best_deg_v;
    int best_max_edge_u;
    int best_max_edge_v;
    int best_max_edge_uv;
    int best_max_excess_u;
    int best_max_excess_v;
    int best_max_excess_uv;
    int best_abs_bound_u;
    int best_abs_bound_v;
    int best_abs_bound_uv;
    long long iterations;
    long long kicks;
    double elapsed;
    double ips;
    int len_su;
    int len_sv;
    int len_suv;
    int su[MAXK];
    int sv[MAXK];
    int suv[MAXK];
} CResult;

typedef struct {
    int channel; /* 0=U, 1=V, 2=UV */
    int kind;    /* 0=red edge, 1=blue non-edge */
    int d;
    int count;
    int bound;
    int excess;
    int present;
} Violation;

typedef struct {
    int used;
    long long iteration;
    long long kick_index;
    int score;
    int linear_score;
    int deg_u;
    int deg_v;
    int violation_count;
    int hamming_to_previous;
    int vars[MAXVARS];
    Violation violations[MAX_VIOLATIONS];
} Motif;

typedef struct {
    int count;
    int total_traps_seen;
    int best_overall_score;
    Motif motifs[MAX_MOTIFS];
} MotifLog;

typedef struct {
    int score;
    int linear_score;
    int deg_u;
    int deg_v;
    int max_edge_u;
    int max_edge_v;
    int max_edge_uv;
    int max_excess_u;
    int max_excess_v;
    int max_excess_uv;
    int abs_bound_u;
    int abs_bound_v;
    int abs_bound_uv;
} Diag;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static double rng_unit(uint64_t *state) {
    return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static int rng_int(uint64_t *state, int limit) {
    return (int)(rng_next(state) % (uint64_t)limit);
}

static int imod(int x, int k) {
    x %= k;
    return x < 0 ? x + k : x;
}

static int is_changed(int q, const int *changed, int m) {
    for (int i = 0; i < m; ++i) {
        if (changed[i] == q) return i;
    }
    return -1;
}

static void zero_ints(int *a, int n) {
    memset(a, 0, (size_t)n * sizeof(int));
}

static void recompute_all(
    int k,
    const unsigned char *xU,
    const unsigned char *xV,
    const unsigned char *xUV,
    int *CU,
    int *CV,
    int *CUV
) {
    zero_ints(CU, k);
    zero_ints(CV, k);
    zero_ints(CUV, k);
    for (int d = 0; d < k; ++d) {
        int cu = 0, cv = 0, cuv = 0;
        for (int w = 0; w < k; ++w) {
            cu += (int)xU[w] * (int)xU[imod(w - d, k)];
            cu += (int)xUV[w] * (int)xUV[imod(w - d, k)];
            cv += (int)xV[w] * (int)xV[imod(w - d, k)];
            cv += (int)xUV[w] * (int)xUV[imod(w + d, k)];
            cuv += (int)xU[w] * (int)xUV[imod(d - w, k)];
            cuv += (int)xUV[w] * (int)xV[imod(w - d, k)];
        }
        CU[d] = cu;
        CV[d] = cv;
        CUV[d] = cuv;
    }
}

static void update_autocorr(
    int k,
    const unsigned char *x,
    int *C,
    const int *changed,
    const unsigned char *oldv,
    const unsigned char *newv,
    int m,
    int plus_orientation
) {
    for (int pi = 0; pi < m; ++pi) {
        int p = changed[pi];
        int oldp = oldv[pi];
        int newp = newv[pi];
        for (int q = 0; q < k; ++q) {
            int qi = is_changed(q, changed, m);
            int oldq = qi >= 0 ? oldv[qi] : x[q];
            int newq = qi >= 0 ? newv[qi] : x[q];
            int delta = (newp && newq) - (oldp && oldq);
            int d1 = plus_orientation ? imod(q - p, k) : imod(p - q, k);
            C[d1] += delta;
            if (qi < 0) {
                int delta2 = (newq && newp) - (oldq && oldp);
                int d2 = plus_orientation ? imod(p - q, k) : imod(q - p, k);
                C[d2] += delta2;
            }
        }
    }
}

static void score_diag(
    int n,
    int k,
    const unsigned char *xU,
    const unsigned char *xV,
    const unsigned char *xUV,
    const int *CU,
    const int *CV,
    const int *CUV,
    Diag *out
) {
    int sum_u = 0, sum_v = 0, sum_uv = 0;
    for (int i = 0; i < k; ++i) {
        sum_u += xU[i];
        sum_v += xV[i];
        sum_uv += xUV[i];
    }
    int deg_u = sum_u + sum_uv;
    int deg_v = sum_v + sum_uv;
    int edge_bound = n - 2;
    int abs_u = 2 * deg_u - 3 * n + 3;
    int abs_v = 2 * deg_v - 3 * n + 3;
    int abs_uv = deg_u + deg_v - 3 * n + 3;
    int score = 0;
    int linear = 0;
    int max_edge_u = 0, max_edge_v = 0, max_edge_uv = 0;
    int max_ex_u = 0, max_ex_v = 0, max_ex_uv = 0;

    for (int d = 1; d < k; ++d) {
        int bound_u = xU[d] ? edge_bound : abs_u;
        int ex_u = CU[d] - bound_u;
        if (ex_u < 0) ex_u = 0;
        linear += ex_u;
        score += ex_u * ex_u;
        if (ex_u > max_ex_u) max_ex_u = ex_u;
        if (xU[d] && CU[d] > max_edge_u) max_edge_u = CU[d];

        int bound_v = xV[d] ? edge_bound : abs_v;
        int ex_v = CV[d] - bound_v;
        if (ex_v < 0) ex_v = 0;
        linear += ex_v;
        score += ex_v * ex_v;
        if (ex_v > max_ex_v) max_ex_v = ex_v;
        if (xV[d] && CV[d] > max_edge_v) max_edge_v = CV[d];
    }
    for (int d = 0; d < k; ++d) {
        int bound_uv = xUV[d] ? edge_bound : abs_uv;
        int ex_uv = CUV[d] - bound_uv;
        if (ex_uv < 0) ex_uv = 0;
        linear += ex_uv;
        score += ex_uv * ex_uv;
        if (ex_uv > max_ex_uv) max_ex_uv = ex_uv;
        if (xUV[d] && CUV[d] > max_edge_uv) max_edge_uv = CUV[d];
    }

    out->score = score;
    out->linear_score = linear;
    out->deg_u = deg_u;
    out->deg_v = deg_v;
    out->max_edge_u = max_edge_u;
    out->max_edge_v = max_edge_v;
    out->max_edge_uv = max_edge_uv;
    out->max_excess_u = max_ex_u;
    out->max_excess_v = max_ex_v;
    out->max_excess_uv = max_ex_uv;
    out->abs_bound_u = abs_u;
    out->abs_bound_v = abs_v;
    out->abs_bound_uv = abs_uv;
}

static void random_state(int n, int k, uint64_t *rng, unsigned char *xU, unsigned char *xV, unsigned char *xUV) {
    int half = (k - 1) / 2;
    double p = 0.43 + 0.14 * rng_unit(rng);
    memset(xU, 0, (size_t)k);
    memset(xV, 0, (size_t)k);
    memset(xUV, 0, (size_t)k);
    (void)n;
    for (int d = 1; d <= half; ++d) {
        unsigned char bu = rng_unit(rng) < p;
        unsigned char bv = rng_unit(rng) < p;
        xU[d] = xU[k - d] = bu;
        xV[d] = xV[k - d] = bv;
    }
    for (int d = 0; d < k; ++d) {
        xUV[d] = rng_unit(rng) < p;
    }
}

static void pack_vars(int k, const unsigned char *xU, const unsigned char *xV, const unsigned char *xUV, int *vars) {
    int half = (k - 1) / 2;
    int idx = 0;
    for (int d = 1; d <= half; ++d) vars[idx++] = xU[d] ? 1 : 0;
    for (int d = 1; d <= half; ++d) vars[idx++] = xV[d] ? 1 : 0;
    for (int d = 0; d < k; ++d) vars[idx++] = xUV[d] ? 1 : 0;
}

static int hamming_vars(const int *a, const int *b, int n) {
    int dist = 0;
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) dist++;
    }
    return dist;
}

static void append_violation(Motif *motif, int channel, int kind, int d, int count, int bound, int present) {
    if (motif->violation_count >= MAX_VIOLATIONS) return;
    int idx = motif->violation_count++;
    motif->violations[idx].channel = channel;
    motif->violations[idx].kind = kind;
    motif->violations[idx].d = d;
    motif->violations[idx].count = count;
    motif->violations[idx].bound = bound;
    motif->violations[idx].excess = count - bound;
    motif->violations[idx].present = present;
}

static void collect_violations(
    int n,
    int k,
    const unsigned char *xU,
    const unsigned char *xV,
    const unsigned char *xUV,
    const int *CU,
    const int *CV,
    const int *CUV,
    const Diag *diag,
    Motif *motif
) {
    int edge_bound = n - 2;
    motif->violation_count = 0;
    for (int d = 1; d < k; ++d) {
        if (xU[d]) {
            if (CU[d] > edge_bound) append_violation(motif, 0, 0, d, CU[d], edge_bound, 1);
        } else {
            if (CU[d] > diag->abs_bound_u) append_violation(motif, 0, 1, d, CU[d], diag->abs_bound_u, 0);
        }
        if (xV[d]) {
            if (CV[d] > edge_bound) append_violation(motif, 1, 0, d, CV[d], edge_bound, 1);
        } else {
            if (CV[d] > diag->abs_bound_v) append_violation(motif, 1, 1, d, CV[d], diag->abs_bound_v, 0);
        }
    }
    for (int d = 0; d < k; ++d) {
        if (xUV[d]) {
            if (CUV[d] > edge_bound) append_violation(motif, 2, 0, d, CUV[d], edge_bound, 1);
        } else {
            if (CUV[d] > diag->abs_bound_uv) append_violation(motif, 2, 1, d, CUV[d], diag->abs_bound_uv, 0);
        }
    }
}

static void maybe_save_score4_motif(
    int n,
    int k,
    const unsigned char *xU,
    const unsigned char *xV,
    const unsigned char *xUV,
    const int *CU,
    const int *CV,
    const int *CUV,
    const Diag *diag,
    long long iterations,
    long long kicks,
    MotifLog *log
) {
    int half = (k - 1) / 2;
    int var_count = 2 * half + k;
    if (log == NULL || diag->score != 4) return;
    log->total_traps_seen++;

    int packed[MAXVARS];
    pack_vars(k, xU, xV, xUV, packed);
    int prev_index = log->count - 1;
    int hprev = prev_index >= 0 ? hamming_vars(packed, log->motifs[prev_index].vars, var_count) : -1;
    int diverse = (prev_index < 0 || hprev > 0);
    if (log->count >= MAX_MOTIFS || !diverse) {
        return;
    }

    Motif *motif = &log->motifs[log->count++];
    memset(motif, 0, sizeof(Motif));
    motif->used = 1;
    motif->iteration = iterations;
    motif->kick_index = kicks + 1;
    motif->score = diag->score;
    motif->linear_score = diag->linear_score;
    motif->deg_u = diag->deg_u;
    motif->deg_v = diag->deg_v;
    motif->hamming_to_previous = hprev;
    memcpy(motif->vars, packed, (size_t)var_count * sizeof(int));
    collect_violations(n, k, xU, xV, xUV, CU, CV, CUV, diag, motif);
    printf("score4_motif saved=%d traps_seen=%d iter=%lld kick=%lld hprev=%d violations=%d degU=%d degV=%d\n",
        log->count, log->total_traps_seen, iterations, kicks + 1, hprev, motif->violation_count, diag->deg_u, diag->deg_v);
    fflush(stdout);
}

static void flip_u(
    int k,
    unsigned char *xU,
    const unsigned char *xUV,
    int d,
    int *CU,
    int *CUV
) {
    int changed[2] = {d, k - d};
    unsigned char oldv[2] = {xU[d], xU[k - d]};
    unsigned char nv = (unsigned char)(!xU[d]);
    unsigned char newv[2] = {nv, nv};
    update_autocorr(k, xU, CU, changed, oldv, newv, 2, 0);
    for (int pi = 0; pi < 2; ++pi) {
        int p = changed[pi];
        int delta = (int)newv[pi] - (int)oldv[pi];
        if (delta != 0) {
            for (int dd = 0; dd < k; ++dd) {
                CUV[dd] += delta * (int)xUV[imod(dd - p, k)];
            }
        }
    }
    xU[d] = nv;
    xU[k - d] = nv;
}

static void flip_v(
    int k,
    unsigned char *xV,
    const unsigned char *xUV,
    int d,
    int *CV,
    int *CUV
) {
    int changed[2] = {d, k - d};
    unsigned char oldv[2] = {xV[d], xV[k - d]};
    unsigned char nv = (unsigned char)(!xV[d]);
    unsigned char newv[2] = {nv, nv};
    update_autocorr(k, xV, CV, changed, oldv, newv, 2, 0);
    for (int pi = 0; pi < 2; ++pi) {
        int p = changed[pi];
        int delta = (int)newv[pi] - (int)oldv[pi];
        if (delta != 0) {
            for (int w = 0; w < k; ++w) {
                CUV[imod(w - p, k)] += delta * (int)xUV[w];
            }
        }
    }
    xV[d] = nv;
    xV[k - d] = nv;
}

static void flip_uv(
    int k,
    const unsigned char *xU,
    const unsigned char *xV,
    unsigned char *xUV,
    int d,
    int *CU,
    int *CV,
    int *CUV
) {
    int changed[1] = {d};
    unsigned char oldv[1] = {xUV[d]};
    unsigned char newv[1] = {(unsigned char)(!xUV[d])};
    int delta = (int)newv[0] - (int)oldv[0];
    update_autocorr(k, xUV, CU, changed, oldv, newv, 1, 0);
    update_autocorr(k, xUV, CV, changed, oldv, newv, 1, 1);
    if (delta != 0) {
        for (int w = 0; w < k; ++w) {
            CUV[imod(d + w, k)] += delta * (int)xU[w];
            CUV[w] += delta * (int)xV[imod(d - w, k)];
        }
    }
    xUV[d] = newv[0];
}

static void apply_flip(
    int k,
    int var_id,
    unsigned char *xU,
    unsigned char *xV,
    unsigned char *xUV,
    int *CU,
    int *CV,
    int *CUV
) {
    int half = (k - 1) / 2;
    if (var_id < half) {
        flip_u(k, xU, xUV, var_id + 1, CU, CUV);
    } else if (var_id < 2 * half) {
        flip_v(k, xV, xUV, var_id - half + 1, CV, CUV);
    } else {
        flip_uv(k, xU, xV, xUV, var_id - 2 * half, CU, CV, CUV);
    }
}

static void store_best(
    int n,
    int k,
    const unsigned char *xU,
    const unsigned char *xV,
    const unsigned char *xUV,
    const Diag *diag,
    long long iterations,
    long long kicks,
    double elapsed,
    CResult *res
) {
    res->n = n;
    res->k = k;
    res->best_score = diag->score;
    res->best_linear_score = diag->linear_score;
    res->best_deg_u = diag->deg_u;
    res->best_deg_v = diag->deg_v;
    res->best_max_edge_u = diag->max_edge_u;
    res->best_max_edge_v = diag->max_edge_v;
    res->best_max_edge_uv = diag->max_edge_uv;
    res->best_max_excess_u = diag->max_excess_u;
    res->best_max_excess_v = diag->max_excess_v;
    res->best_max_excess_uv = diag->max_excess_uv;
    res->best_abs_bound_u = diag->abs_bound_u;
    res->best_abs_bound_v = diag->abs_bound_v;
    res->best_abs_bound_uv = diag->abs_bound_uv;
    res->iterations = iterations;
    res->kicks = kicks;
    res->elapsed = elapsed;
    res->ips = iterations / (elapsed > 1e-9 ? elapsed : 1e-9);
    res->len_su = 0;
    res->len_sv = 0;
    res->len_suv = 0;
    for (int i = 1; i < k; ++i) {
        if (xU[i]) res->su[res->len_su++] = i;
        if (xV[i]) res->sv[res->len_sv++] = i;
    }
    for (int i = 0; i < k; ++i) {
        if (xUV[i]) res->suv[res->len_suv++] = i;
    }
}

int self_check(int n, unsigned long long seed, int steps) {
    int k = 2 * n - 1;
    int half = (k - 1) / 2;
    int var_count = 2 * half + k;
    uint64_t rng = seed ? (uint64_t)seed : 1ULL;
    unsigned char xU[MAXK], xV[MAXK], xUV[MAXK];
    int CU[MAXK], CV[MAXK], CUV[MAXK];
    int RU[MAXK], RV[MAXK], RUV[MAXK];
    random_state(n, k, &rng, xU, xV, xUV);
    recompute_all(k, xU, xV, xUV, CU, CV, CUV);
    for (int s = 0; s < steps; ++s) {
        int var_id = rng_int(&rng, var_count);
        apply_flip(k, var_id, xU, xV, xUV, CU, CV, CUV);
        recompute_all(k, xU, xV, xUV, RU, RV, RUV);
        for (int d = 0; d < k; ++d) {
            if (CU[d] != RU[d] || CV[d] != RV[d] || CUV[d] != RUV[d]) {
                fprintf(stderr, "self_check mismatch n=%d step=%d d=%d CU=%d/%d CV=%d/%d CUV=%d/%d\n",
                    n, s, d, CU[d], RU[d], CV[d], RV[d], CUV[d], RUV[d]);
                return 1;
            }
        }
    }
    return 0;
}

static void print_best(const char *prefix, const CResult *res) {
    printf("%s n=%d iter=%lld elapsed=%.3f score_sq=%d score_linear=%d degU=%d degV=%d maxEdge=(%d,%d,%d) maxEx=(%d,%d,%d) abs=(%d,%d,%d) kicks=%lld\n",
        prefix, res->n, res->iterations, res->elapsed, res->best_score, res->best_linear_score,
        res->best_deg_u, res->best_deg_v,
        res->best_max_edge_u, res->best_max_edge_v, res->best_max_edge_uv,
        res->best_max_excess_u, res->best_max_excess_v, res->best_max_excess_uv,
        res->best_abs_bound_u, res->best_abs_bound_v, res->best_abs_bound_uv,
        res->kicks);
    fflush(stdout);
}

int solve_tabu(int n, double seconds, unsigned long long seed, long long max_iterations, CResult *res, MotifLog *motif_log) {
    int k = 2 * n - 1;
    if (k >= MAXK || k <= 1 || (k % 2) == 0) return 2;

    memset(res, 0, sizeof(CResult));
    if (motif_log != NULL) {
        memset(motif_log, 0, sizeof(MotifLog));
        motif_log->best_overall_score = 2147483647;
    }
    res->n = n;
    res->k = k;
    res->best_score = 2147483647;
    res->best_linear_score = 2147483647;

    int half = (k - 1) / 2;
    int var_count = 2 * half + k;
    if (var_count >= MAXVARS) return 3;
    uint64_t rng = seed ? (uint64_t)seed : 1ULL;
    unsigned char xU[MAXK], xV[MAXK], xUV[MAXK];
    unsigned char pending_xU[MAXK], pending_xV[MAXK], pending_xUV[MAXK];
    int CU[MAXK], CV[MAXK], CUV[MAXK];
    int pending_CU[MAXK], pending_CV[MAXK], pending_CUV[MAXK];
    int tabu_until[MAXVARS];
    memset(tabu_until, 0, sizeof(tabu_until));

    Diag diag;
    Diag pending_diag;
    int pending_score4_valid = 0;
    long long pending_score4_iter = 0;
    long long iterations = 0;
    long long kicks = 0;
    long long since_improve = 0;
    double start = now_seconds();
    double deadline = start + seconds;
    double last_print = start;

    random_state(n, k, &rng, xU, xV, xUV);
    recompute_all(k, xU, xV, xUV, CU, CV, CUV);
    score_diag(n, k, xU, xV, xUV, CU, CV, CUV, &diag);
    if (motif_log != NULL && diag.score < motif_log->best_overall_score) {
        motif_log->best_overall_score = diag.score;
    }
    if (diag.score == 4) {
        memcpy(pending_xU, xU, (size_t)k);
        memcpy(pending_xV, xV, (size_t)k);
        memcpy(pending_xUV, xUV, (size_t)k);
        memcpy(pending_CU, CU, (size_t)k * sizeof(int));
        memcpy(pending_CV, CV, (size_t)k * sizeof(int));
        memcpy(pending_CUV, CUV, (size_t)k * sizeof(int));
        pending_diag = diag;
        pending_score4_valid = 1;
        pending_score4_iter = iterations;
    }
    store_best(n, k, xU, xV, xUV, &diag, iterations, kicks, 0.0, res);
    if (res->best_score == 0) {
        res->found = 1;
        printf("C tabu start n=%d k=%d variables=%d already_score_zero seed=%llu\n", n, k, var_count, seed);
        print_best("initial_best", res);
        return 0;
    }

    printf("C tabu start n=%d k=%d variables=%d time_limit=%.3f max_iterations=%lld seed=%llu\n",
        n, k, var_count, seconds, max_iterations, seed);
    print_best("initial_best", res);

    int current_score = diag.score;
    while (iterations < max_iterations) {
        if ((iterations & 255LL) == 0) {
            if (now_seconds() >= deadline) break;
        }

        int best_var = -1;
        int best_neighbor_score = 2147483647;
        int best_neighbor_linear = 2147483647;
        int best_neighbor_delta = 2147483647;

        for (int var_id = 0; var_id < var_count; ++var_id) {
            apply_flip(k, var_id, xU, xV, xUV, CU, CV, CUV);
            score_diag(n, k, xU, xV, xUV, CU, CV, CUV, &diag);
            int is_tabu = tabu_until[var_id] > iterations;
            int aspirates = diag.score < res->best_score;
            if (!is_tabu || aspirates) {
                int delta = diag.score - current_score;
                if (
                    diag.score < best_neighbor_score ||
                    (diag.score == best_neighbor_score && diag.linear_score < best_neighbor_linear) ||
                    (diag.score == best_neighbor_score && diag.linear_score == best_neighbor_linear && delta < best_neighbor_delta) ||
                    (diag.score == best_neighbor_score && diag.linear_score == best_neighbor_linear && delta == best_neighbor_delta && rng_int(&rng, 2) == 0)
                ) {
                    best_neighbor_score = diag.score;
                    best_neighbor_linear = diag.linear_score;
                    best_neighbor_delta = delta;
                    best_var = var_id;
                }
            }
            apply_flip(k, var_id, xU, xV, xUV, CU, CV, CUV);
        }

        if (best_var < 0) {
            memset(tabu_until, 0, sizeof(tabu_until));
            continue;
        }

        apply_flip(k, best_var, xU, xV, xUV, CU, CV, CUV);
        score_diag(n, k, xU, xV, xUV, CU, CV, CUV, &diag);
        current_score = diag.score;
        if (motif_log != NULL && diag.score < motif_log->best_overall_score) {
            motif_log->best_overall_score = diag.score;
        }
        if (diag.score == 4) {
            memcpy(pending_xU, xU, (size_t)k);
            memcpy(pending_xV, xV, (size_t)k);
            memcpy(pending_xUV, xUV, (size_t)k);
            memcpy(pending_CU, CU, (size_t)k * sizeof(int));
            memcpy(pending_CV, CV, (size_t)k * sizeof(int));
            memcpy(pending_CUV, CUV, (size_t)k * sizeof(int));
            pending_diag = diag;
            pending_score4_valid = 1;
            pending_score4_iter = iterations;
        }
        int tenure = 10 + rng_int(&rng, 11);
        tabu_until[best_var] = (int)(iterations + tenure);
        iterations++;
        since_improve++;

        if (diag.score < res->best_score || (diag.score == res->best_score && diag.linear_score < res->best_linear_score)) {
            double elapsed = now_seconds() - start;
            store_best(n, k, xU, xV, xUV, &diag, iterations, kicks, elapsed, res);
            print_best("new_best", res);
            since_improve = 0;
            if (res->best_score == 0) {
                res->found = 1;
                break;
            }
        }

        if (since_improve >= 10000) {
            if (pending_score4_valid) {
                maybe_save_score4_motif(n, k, pending_xU, pending_xV, pending_xUV,
                    pending_CU, pending_CV, pending_CUV, &pending_diag, pending_score4_iter, kicks, motif_log);
            }
            for (int i = 0; i < 10; ++i) {
                int var_id = rng_int(&rng, var_count);
                apply_flip(k, var_id, xU, xV, xUV, CU, CV, CUV);
            }
            memset(tabu_until, 0, sizeof(tabu_until));
            score_diag(n, k, xU, xV, xUV, CU, CV, CUV, &diag);
            current_score = diag.score;
            kicks++;
            since_improve = 0;
            pending_score4_valid = 0;
        }

        double now = now_seconds();
        if (now - last_print >= 10.0) {
            double elapsed = now - start;
            printf("progress n=%d elapsed=%.1f iter=%lld ips=%.1f current_sq=%d current_linear=%d best_sq=%d best_linear=%d kicks=%lld last_best_gap=%lld\n",
                n, elapsed, iterations, iterations / (elapsed > 1e-9 ? elapsed : 1e-9),
                current_score, diag.linear_score, res->best_score, res->best_linear_score,
                kicks, since_improve);
            fflush(stdout);
            last_print = now;
        }
    }

    {
        double elapsed = now_seconds() - start;
        res->iterations = iterations;
        res->kicks = kicks;
        res->elapsed = elapsed;
        res->ips = iterations / (elapsed > 1e-9 ? elapsed : 1e-9);
        printf("C tabu done n=%d found=%d best_sq=%d best_linear=%d iter=%lld elapsed=%.3f ips=%.1f kicks=%lld\n",
            n, res->found, res->best_score, res->best_linear_score, iterations, elapsed, res->ips, kicks);
        fflush(stdout);
    }
    return 0;
}
