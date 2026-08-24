
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M 71
#define REPS 35

typedef struct {
    int max_excess;
    int total_excess;
    int positive_count;
    int sumsq_excess;
    int target_l1;
    int target_l2;
    int min_slack;
    int worst_d;
    long long energy;
} Metrics;

typedef struct {
    int max_excess;
    int total_excess;
    int positive_count;
    int sumsq_excess;
    int target_l1;
    int target_l2;
    int min_slack;
    int worst_d;
    uint64_t c_lo;
    uint64_t c_hi;
    long long evaluated_moves;
    long long accepted_moves;
    long long improving_moves;
    long long restarts;
    long long best_updates;
    int threads_used;
    double elapsed_sec;
} SearchResult;

static const __uint128_t MASK71 = (((__uint128_t)1) << 71) - 1;

static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static inline double rng_unit(uint64_t *state) {
    return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int bit_get(__uint128_t x, int idx) {
    return (int)((x >> idx) & 1);
}

static inline int popcount128(__uint128_t x) {
    return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64));
}

static inline __uint128_t rotate_mod71(__uint128_t x, int shift) {
    return ((x >> shift) | (x << (71 - shift))) & MASK71;
}

static Metrics evaluate_c(__uint128_t c_bits, const int *cap, const int *target) {
    Metrics m;
    memset(&m, 0, sizeof(m));
    m.min_slack = 1000000;
    for (int d = 1; d <= REPS; d++) {
        int nc = popcount128(c_bits & rotate_mod71(c_bits, d));
        int slack = cap[d] - nc;
        int excess = slack < 0 ? -slack : 0;
        int delta = nc - target[d];
        if (delta < 0) delta = -delta;
        if (excess > m.max_excess) {
            m.max_excess = excess;
            m.worst_d = d;
        }
        m.total_excess += 2 * excess;
        m.positive_count += excess > 0 ? 2 : 0;
        m.sumsq_excess += 2 * excess * excess;
        m.target_l1 += delta;
        m.target_l2 += delta * delta;
        if (slack < m.min_slack) m.min_slack = slack;
    }
    m.energy =
        (long long)m.max_excess * 10000000LL
        + (long long)m.total_excess * 100000LL
        + (long long)m.positive_count * 1000LL
        + (long long)m.sumsq_excess * 20LL
        + (long long)m.target_l1;
    return m;
}

static int lex_better(const Metrics *a, const Metrics *b) {
    if (a->max_excess != b->max_excess) return a->max_excess < b->max_excess;
    if (a->total_excess != b->total_excess) return a->total_excess < b->total_excess;
    if (a->positive_count != b->positive_count) return a->positive_count < b->positive_count;
    if (a->sumsq_excess != b->sumsq_excess) return a->sumsq_excess < b->sumsq_excess;
    if (a->target_l1 != b->target_l1) return a->target_l1 < b->target_l1;
    if (a->target_l2 != b->target_l2) return a->target_l2 < b->target_l2;
    return a->min_slack > b->min_slack;
}

static void random_c(uint64_t *rng, __uint128_t *c_bits) {
    __uint128_t c = 1;
    int count = 1;
    while (count < 35) {
        int r = 1 + (int)(rng_next(rng) % 70ULL);
        if (!bit_get(c, r)) {
            c |= ((__uint128_t)1) << r;
            count++;
        }
    }
    *c_bits = c;
}

static void list_selected(__uint128_t c, int *sel, int *unsel, int *ns, int *nu) {
    *ns = 0;
    *nu = 0;
    for (int r = 1; r < M; r++) {
        if (bit_get(c, r)) {
            sel[(*ns)++] = r;
        } else {
            unsel[(*nu)++] = r;
        }
    }
}

static __uint128_t propose_swap(__uint128_t c, uint64_t *rng) {
    int sel[70];
    int unsel[70];
    int ns = 0;
    int nu = 0;
    list_selected(c, sel, unsel, &ns, &nu);
    if (ns <= 0 || nu <= 0) return c;
    int rem = sel[(int)(rng_next(rng) % (uint64_t)ns)];
    int add = unsel[(int)(rng_next(rng) % (uint64_t)nu)];
    c &= ~(((__uint128_t)1) << rem);
    c |= ((__uint128_t)1) << add;
    return c;
}

static __uint128_t best_of_random_swaps(__uint128_t c, uint64_t *rng, const int *cap, const int *target, int samples) {
    Metrics best_m = evaluate_c(c, cap, target);
    __uint128_t best = c;
    for (int i = 0; i < samples; i++) {
        __uint128_t cand = propose_swap(c, rng);
        Metrics cm = evaluate_c(cand, cap, target);
        if (lex_better(&cm, &best_m)) {
            best_m = cm;
            best = cand;
        }
    }
    return best;
}

static __uint128_t greedy_one_swap_descent(__uint128_t c, const int *cap, const int *target, double deadline) {
    int improved = 1;
    while (improved && omp_get_wtime() < deadline) {
        improved = 0;
        Metrics best_m = evaluate_c(c, cap, target);
        __uint128_t best = c;
        int sel[70];
        int unsel[70];
        int ns = 0;
        int nu = 0;
        list_selected(c, sel, unsel, &ns, &nu);
        for (int i = 0; i < ns; i++) {
            for (int j = 0; j < nu; j++) {
                __uint128_t cand = c;
                cand &= ~(((__uint128_t)1) << sel[i]);
                cand |= ((__uint128_t)1) << unsel[j];
                Metrics cm = evaluate_c(cand, cap, target);
                if (lex_better(&cm, &best_m)) {
                    best_m = cm;
                    best = cand;
                    improved = 1;
                }
            }
        }
        c = best;
    }
    return c;
}

static __uint128_t greedy_one_swap_descent_passes(__uint128_t c, const int *cap, const int *target, int max_passes) {
    for (int pass = 0; pass < max_passes; pass++) {
        int improved = 0;
        Metrics best_m = evaluate_c(c, cap, target);
        __uint128_t best = c;
        int sel[70];
        int unsel[70];
        int ns = 0;
        int nu = 0;
        list_selected(c, sel, unsel, &ns, &nu);
        for (int i = 0; i < ns; i++) {
            for (int j = 0; j < nu; j++) {
                __uint128_t cand = c;
                cand &= ~(((__uint128_t)1) << sel[i]);
                cand |= ((__uint128_t)1) << unsel[j];
                Metrics cm = evaluate_c(cand, cap, target);
                if (lex_better(&cm, &best_m)) {
                    best_m = cm;
                    best = cand;
                    improved = 1;
                }
            }
        }
        c = best;
        if (!improved) break;
    }
    return c;
}

int run_c_search_fixed_steps(
    const int *cap,
    const int *target,
    long long max_steps,
    uint64_t seed,
    SearchResult *out
) {
    memset(out, 0, sizeof(*out));
    double start = omp_get_wtime();
    if (max_steps < 1) max_steps = 1;

    __uint128_t global_best_c = 1;
    random_c(&seed, &global_best_c);
    Metrics global_best_m = evaluate_c(global_best_c, cap, target);

    uint64_t rng = seed ^ 0x9e3779b97f4a7c15ULL;
    __uint128_t c;
    random_c(&rng, &c);
    Metrics cur = evaluate_c(c, cap, target);
    Metrics local_best_m = cur;
    __uint128_t local_best_c = c;

    for (long long steps = 0; steps < max_steps; steps++) {
        if ((steps % 25000LL) == 0) {
            if (steps > 0) {
                c = local_best_c;
                c = greedy_one_swap_descent_passes(c, cap, target, 1);
                cur = evaluate_c(c, cap, target);
                if (lex_better(&cur, &local_best_m)) {
                    local_best_m = cur;
                    local_best_c = c;
                }
            }
            if ((steps % 100000LL) == 0) {
                random_c(&rng, &c);
                cur = evaluate_c(c, cap, target);
                out->restarts++;
            }
        }

        int samples = (rng_next(&rng) & 7ULL) == 0ULL ? 6 : 1;
        __uint128_t cand = samples == 1 ? propose_swap(c, &rng) : best_of_random_swaps(c, &rng, cap, target, samples);
        Metrics cm = evaluate_c(cand, cap, target);
        out->evaluated_moves += samples;
        int accept = 0;
        if (lex_better(&cm, &cur)) {
            accept = 1;
            out->improving_moves++;
        } else {
            double frac = (double)steps / (double)max_steps;
            if (frac > 1.0) frac = 1.0;
            double temp = 1800000.0 * (1.0 - frac) + 25.0;
            double delta = (double)(cm.energy - cur.energy);
            if (delta < 0.0 || rng_unit(&rng) < exp(-delta / temp)) {
                accept = 1;
            }
        }
        if (accept) {
            c = cand;
            cur = cm;
            out->accepted_moves++;
            if (lex_better(&cur, &local_best_m)) {
                local_best_m = cur;
                local_best_c = c;
                out->best_updates++;
            }
        }
        if (lex_better(&local_best_m, &global_best_m)) {
            global_best_m = local_best_m;
            global_best_c = local_best_c;
        }
        if (global_best_m.max_excess == 0) break;
    }

    global_best_c = greedy_one_swap_descent_passes(global_best_c, cap, target, 20);
    global_best_m = evaluate_c(global_best_c, cap, target);
    out->max_excess = global_best_m.max_excess;
    out->total_excess = global_best_m.total_excess;
    out->positive_count = global_best_m.positive_count;
    out->sumsq_excess = global_best_m.sumsq_excess;
    out->target_l1 = global_best_m.target_l1;
    out->target_l2 = global_best_m.target_l2;
    out->min_slack = global_best_m.min_slack;
    out->worst_d = global_best_m.worst_d;
    out->c_lo = (uint64_t)global_best_c;
    out->c_hi = (uint64_t)(global_best_c >> 64);
    out->threads_used = 1;
    out->elapsed_sec = omp_get_wtime() - start;
    return global_best_m.max_excess == 0 ? 1 : 0;
}

int run_c_search(
    const int *cap,
    const int *target,
    double seconds,
    int requested_threads,
    uint64_t seed,
    SearchResult *out
) {
    memset(out, 0, sizeof(*out));
    if (requested_threads < 1) requested_threads = 1;
    omp_set_num_threads(requested_threads);

    double start = omp_get_wtime();
    double deadline = start + seconds;
    __uint128_t global_best_c = 1;
    random_c(&seed, &global_best_c);
    Metrics global_best_m = evaluate_c(global_best_c, cap, target);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t rng = seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1));
        long long local_evaluated = 0;
        long long local_accepted = 0;
        long long local_improving = 0;
        long long local_restarts = 0;
        long long local_updates = 0;
        __uint128_t c;
        random_c(&rng, &c);
        Metrics cur = evaluate_c(c, cap, target);
        Metrics local_best_m = cur;
        __uint128_t local_best_c = c;
        long long steps = 0;

        while (omp_get_wtime() < deadline) {
            if ((steps % 25000LL) == 0) {
                if (steps > 0) {
                    c = local_best_c;
                    c = greedy_one_swap_descent(c, cap, target, omp_get_wtime() + 0.03);
                    cur = evaluate_c(c, cap, target);
                    if (lex_better(&cur, &local_best_m)) {
                        local_best_m = cur;
                        local_best_c = c;
                    }
                }
                if ((steps % 100000LL) == 0) {
                    random_c(&rng, &c);
                    cur = evaluate_c(c, cap, target);
                    local_restarts++;
                }
            }

            int samples = (rng_next(&rng) & 7ULL) == 0ULL ? 6 : 1;
            __uint128_t cand = samples == 1 ? propose_swap(c, &rng) : best_of_random_swaps(c, &rng, cap, target, samples);
            Metrics cm = evaluate_c(cand, cap, target);
            local_evaluated += samples;
            int accept = 0;
            if (lex_better(&cm, &cur)) {
                accept = 1;
                local_improving++;
            } else {
                double elapsed = omp_get_wtime() - start;
                double frac = seconds > 0.0 ? elapsed / seconds : 1.0;
                if (frac > 1.0) frac = 1.0;
                double temp = 1800000.0 * (1.0 - frac) + 25.0;
                double delta = (double)(cm.energy - cur.energy);
                if (delta < 0.0 || rng_unit(&rng) < exp(-delta / temp)) {
                    accept = 1;
                }
            }
            if (accept) {
                c = cand;
                cur = cm;
                local_accepted++;
                if (lex_better(&cur, &local_best_m)) {
                    local_best_m = cur;
                    local_best_c = c;
                    local_updates++;
                }
            }
            if (lex_better(&local_best_m, &global_best_m)) {
                #pragma omp critical
                {
                    if (lex_better(&local_best_m, &global_best_m)) {
                        global_best_m = local_best_m;
                        global_best_c = local_best_c;
                    }
                }
            }
            if (global_best_m.max_excess == 0) break;
            steps++;
        }
        #pragma omp atomic
        out->evaluated_moves += local_evaluated;
        #pragma omp atomic
        out->accepted_moves += local_accepted;
        #pragma omp atomic
        out->improving_moves += local_improving;
        #pragma omp atomic
        out->restarts += local_restarts;
        #pragma omp atomic
        out->best_updates += local_updates;
    }

    global_best_c = greedy_one_swap_descent(global_best_c, cap, target, omp_get_wtime() + 1.5);
    global_best_m = evaluate_c(global_best_c, cap, target);
    out->max_excess = global_best_m.max_excess;
    out->total_excess = global_best_m.total_excess;
    out->positive_count = global_best_m.positive_count;
    out->sumsq_excess = global_best_m.sumsq_excess;
    out->target_l1 = global_best_m.target_l1;
    out->target_l2 = global_best_m.target_l2;
    out->min_slack = global_best_m.min_slack;
    out->worst_d = global_best_m.worst_d;
    out->c_lo = (uint64_t)global_best_c;
    out->c_hi = (uint64_t)(global_best_c >> 64);
    out->threads_used = requested_threads;
    out->elapsed_sec = omp_get_wtime() - start;
    return global_best_m.max_excess == 0 ? 1 : 0;
}
