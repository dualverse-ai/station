#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef unsigned __int128 u128;

enum {
    M = 77,
    PAIR_COUNT = 38,
    A_PAIR_SELECTED = 19,
    C_NONZERO_SELECTED = 37,
    IN_A_THRESHOLD = 37,
    NOT_A_THRESHOLD = 38
};

typedef struct {
    int max_excess;
    int total_excess;
    int positive_count;
    int sumsq;
} Cost;

typedef struct {
    uint8_t A_pair[PAIR_COUNT];
    uint8_t A_full[M];
    uint8_t C_full[M];
    int NA[M];
    int NC[M];
    Cost cost;
} State;

typedef struct {
    u128 A_bits;
    u128 C_bits;
    Cost cost;
    int restart;
    long long step;
} BestRecord;

static int add_table[M][M];
static int neg_table[M];
static int pair_rep[PAIR_COUNT];
static int pair_other[PAIR_COUNT];

static inline u128 bit_mask(int idx) {
    return ((u128)1) << idx;
}

static double now_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static inline int rng_bounded(uint64_t *state, int bound) {
    return (int)(rng_next(state) % (uint64_t)bound);
}

static inline double rng_double(uint64_t *state) {
    return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static int index_from_coords(int a, int b) {
    int aa = a % 7;
    int bb = b % 11;
    if (aa < 0) aa += 7;
    if (bb < 0) bb += 11;
    return aa + 7 * bb;
}

static void coords_from_index(int idx, int *a, int *b) {
    *a = idx % 7;
    *b = (idx / 7) % 11;
}

static void init_group_tables(void) {
    for (int i = 0; i < M; ++i) {
        int a, b;
        coords_from_index(i, &a, &b);
        neg_table[i] = index_from_coords(-a, -b);
    }
    for (int i = 0; i < M; ++i) {
        int ai, bi;
        coords_from_index(i, &ai, &bi);
        for (int j = 0; j < M; ++j) {
            int aj, bj;
            coords_from_index(j, &aj, &bj);
            add_table[i][j] = index_from_coords(ai + aj, bi + bj);
        }
    }

    int seen[M];
    memset(seen, 0, sizeof(seen));
    seen[0] = 1;
    int count = 0;
    for (int i = 1; i < M; ++i) {
        if (seen[i]) {
            continue;
        }
        int j = neg_table[i];
        int r = i < j ? i : j;
        int o = i < j ? j : i;
        pair_rep[count] = r;
        pair_other[count] = o;
        seen[r] = 1;
        seen[o] = 1;
        count += 1;
    }
    if (count != PAIR_COUNT) {
        fprintf(stderr, "bad inverse-pair count %d\n", count);
    }
}

static int legendre7(int x) {
    int v = x % 7;
    if (v < 0) v += 7;
    if (v == 0) return 0;
    return (v == 1 || v == 2 || v == 4) ? 1 : -1;
}

static int legendre11(int x) {
    int v = x % 11;
    if (v < 0) v += 11;
    if (v == 0) return 0;
    return (v == 1 || v == 3 || v == 4 || v == 5 || v == 9) ? 1 : -1;
}

static int structured_score_elem(int idx, int mode) {
    int a, b;
    coords_from_index(idx, &a, &b);
    int sa = mode % 7;
    int sb = (mode / 7) % 11;
    int mix = (mode / 77) % 11;
    int score = 0;
    score += 11 * legendre7(a + sa);
    score += 9 * legendre11(b + sb);
    score += 7 * legendre11(a + b + mix);
    score += 5 * legendre11(2 * a + b + sb);
    score += 3 * legendre7(a + 3 * b + sa);
    score += ((idx * 19 + mode * 37) % 17) - 8;
    return score;
}

static int structured_score_pair(int pair_id, int mode) {
    return structured_score_elem(pair_rep[pair_id], mode)
        + structured_score_elem(pair_other[pair_id], mode + 23);
}

static void shuffle_ints(int *values, int count, uint64_t *rng) {
    for (int i = count - 1; i > 0; --i) {
        int j = rng_bounded(rng, i + 1);
        int tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
}

static void clear_state(State *state) {
    memset(state, 0, sizeof(*state));
}

static void build_a_full(State *state) {
    memset(state->A_full, 0, sizeof(state->A_full));
    for (int p = 0; p < PAIR_COUNT; ++p) {
        if (state->A_pair[p]) {
            state->A_full[pair_rep[p]] = 1;
            state->A_full[pair_other[p]] = 1;
        }
    }
}

static void initialize_corrs(State *state) {
    for (int d = 0; d < M; ++d) {
        int na = 0;
        int nc = 0;
        for (int x = 0; x < M; ++x) {
            na += (int)state->A_full[x] * (int)state->A_full[add_table[x][d]];
            nc += (int)state->C_full[x] * (int)state->C_full[add_table[x][d]];
        }
        state->NA[d] = na;
        state->NC[d] = nc;
    }
}

static Cost cost_from_corrs(const State *state) {
    Cost cost;
    cost.max_excess = 0;
    cost.total_excess = 0;
    cost.positive_count = 0;
    cost.sumsq = 0;
    for (int d = 1; d < M; ++d) {
        int threshold = state->A_full[d] ? IN_A_THRESHOLD : NOT_A_THRESHOLD;
        int excess = state->NA[d] + state->NC[d] - threshold;
        if (excess > 0) {
            if (excess > cost.max_excess) {
                cost.max_excess = excess;
            }
            cost.total_excess += excess;
            cost.positive_count += 1;
            cost.sumsq += excess * excess;
        }
    }
    return cost;
}

static int tuple_less(Cost a, Cost b) {
    if (a.max_excess != b.max_excess) return a.max_excess < b.max_excess;
    if (a.total_excess != b.total_excess) return a.total_excess < b.total_excess;
    if (a.positive_count != b.positive_count) return a.positive_count < b.positive_count;
    return a.sumsq < b.sumsq;
}

static double energy(Cost c) {
    return 1200.0 * (double)c.max_excess
        + 14.0 * (double)c.total_excess
        + 1.7 * (double)c.positive_count
        + 0.06 * (double)c.sumsq;
}

static u128 bits_from_full(const uint8_t *full) {
    u128 bits = 0;
    for (int i = 0; i < M; ++i) {
        if (full[i]) {
            bits |= bit_mask(i);
        }
    }
    return bits;
}

static void init_random_state(State *state, uint64_t *rng) {
    clear_state(state);
    int pairs[PAIR_COUNT];
    for (int i = 0; i < PAIR_COUNT; ++i) {
        pairs[i] = i;
    }
    shuffle_ints(pairs, PAIR_COUNT, rng);
    for (int i = 0; i < A_PAIR_SELECTED; ++i) {
        state->A_pair[pairs[i]] = 1;
    }

    int residues[M - 1];
    for (int i = 0; i < M - 1; ++i) {
        residues[i] = i + 1;
    }
    shuffle_ints(residues, M - 1, rng);
    state->C_full[0] = 1;
    for (int i = 0; i < C_NONZERO_SELECTED; ++i) {
        state->C_full[residues[i]] = 1;
    }
    build_a_full(state);
    initialize_corrs(state);
    state->cost = cost_from_corrs(state);
}

static void init_structured_state(State *state, int mode, uint64_t *rng) {
    clear_state(state);
    int used_pairs[PAIR_COUNT];
    memset(used_pairs, 0, sizeof(used_pairs));
    for (int k = 0; k < A_PAIR_SELECTED; ++k) {
        int best_id = -1;
        int best_score = -1000000000;
        for (int p = 0; p < PAIR_COUNT; ++p) {
            if (used_pairs[p]) {
                continue;
            }
            int score = structured_score_pair(p, mode) + (int)(rng_next(rng) % 23);
            if (score > best_score) {
                best_score = score;
                best_id = p;
            }
        }
        used_pairs[best_id] = 1;
        state->A_pair[best_id] = 1;
    }

    int used_res[M];
    memset(used_res, 0, sizeof(used_res));
    state->C_full[0] = 1;
    used_res[0] = 1;
    for (int k = 0; k < C_NONZERO_SELECTED; ++k) {
        int best_idx = -1;
        int best_score = -1000000000;
        for (int x = 1; x < M; ++x) {
            if (used_res[x]) {
                continue;
            }
            int score = structured_score_elem(x, mode + 41) + (int)(rng_next(rng) % 29);
            if (score > best_score) {
                best_score = score;
                best_idx = x;
            }
        }
        used_res[best_idx] = 1;
        state->C_full[best_idx] = 1;
    }

    build_a_full(state);
    initialize_corrs(state);
    state->cost = cost_from_corrs(state);
}

static int choose_active_pair(const State *state, uint64_t *rng) {
    for (;;) {
        int p = rng_bounded(rng, PAIR_COUNT);
        if (state->A_pair[p]) {
            return p;
        }
    }
}

static int choose_inactive_pair(const State *state, uint64_t *rng) {
    for (;;) {
        int p = rng_bounded(rng, PAIR_COUNT);
        if (!state->A_pair[p]) {
            return p;
        }
    }
}

static int choose_active_c(const State *state, uint64_t *rng) {
    for (;;) {
        int r = 1 + rng_bounded(rng, M - 1);
        if (state->C_full[r]) {
            return r;
        }
    }
}

static int choose_inactive_c(const State *state, uint64_t *rng) {
    for (;;) {
        int r = 1 + rng_bounded(rng, M - 1);
        if (!state->C_full[r]) {
            return r;
        }
    }
}

static int delta_for_pos(int pos, const int *flip_pos, const int *flip_delta, int flip_count) {
    for (int i = 0; i < flip_count; ++i) {
        if (flip_pos[i] == pos) {
            return flip_delta[i];
        }
    }
    return 0;
}

static int delta_corr_for_shift(
    const uint8_t *full,
    const int *flip_pos,
    const int *flip_delta,
    int flip_count,
    int d
) {
    int delta = 0;
    int neg_d = neg_table[d];
    for (int i = 0; i < flip_count; ++i) {
        int p = flip_pos[i];
        int q_minus = add_table[p][neg_d];
        int q_plus = add_table[p][d];
        delta += flip_delta[i] * ((int)full[q_minus] + (int)full[q_plus]);
    }
    for (int i = 0; i < flip_count; ++i) {
        int target = add_table[flip_pos[i]][d];
        for (int j = 0; j < flip_count; ++j) {
            if (flip_pos[j] == target) {
                delta += flip_delta[i] * flip_delta[j];
                break;
            }
        }
    }
    return delta;
}

static Cost proposal_cost(
    const State *state,
    const int *a_pos,
    const int *a_delta,
    int a_count,
    const int *c_pos,
    const int *c_delta,
    int c_count,
    int *delta_na,
    int *delta_nc
) {
    Cost cost;
    cost.max_excess = 0;
    cost.total_excess = 0;
    cost.positive_count = 0;
    cost.sumsq = 0;
    delta_na[0] = 0;
    delta_nc[0] = 0;
    for (int d = 1; d < M; ++d) {
        int dna = a_count ? delta_corr_for_shift(state->A_full, a_pos, a_delta, a_count, d) : 0;
        int dnc = c_count ? delta_corr_for_shift(state->C_full, c_pos, c_delta, c_count, d) : 0;
        delta_na[d] = dna;
        delta_nc[d] = dnc;
        int in_a_after = (int)state->A_full[d] + (a_count ? delta_for_pos(d, a_pos, a_delta, a_count) : 0);
        int threshold = in_a_after ? IN_A_THRESHOLD : NOT_A_THRESHOLD;
        int excess = state->NA[d] + dna + state->NC[d] + dnc - threshold;
        if (excess > 0) {
            if (excess > cost.max_excess) {
                cost.max_excess = excess;
            }
            cost.total_excess += excess;
            cost.positive_count += 1;
            cost.sumsq += excess * excess;
        }
    }
    return cost;
}

static void apply_proposal(
    State *state,
    int remove_pair,
    int add_pair,
    int remove_c,
    int add_c,
    const int *delta_na,
    const int *delta_nc
) {
    for (int d = 1; d < M; ++d) {
        state->NA[d] += delta_na[d];
        state->NC[d] += delta_nc[d];
    }
    if (remove_pair >= 0) {
        state->A_pair[remove_pair] = 0;
        state->A_pair[add_pair] = 1;
        state->A_full[pair_rep[remove_pair]] = 0;
        state->A_full[pair_other[remove_pair]] = 0;
        state->A_full[pair_rep[add_pair]] = 1;
        state->A_full[pair_other[add_pair]] = 1;
    }
    if (remove_c >= 0) {
        state->C_full[remove_c] = 0;
        state->C_full[add_c] = 1;
    }
}

static void copy_to_best(BestRecord *best, const State *state, int restart, long long step) {
    best->A_bits = bits_from_full(state->A_full);
    best->C_bits = bits_from_full(state->C_full);
    best->cost = state->cost;
    best->restart = restart;
    best->step = step;
}

static void print_bits_hex(const char *label, u128 bits) {
    unsigned long long hi = (unsigned long long)(bits >> 64);
    unsigned long long lo = (unsigned long long)bits;
    printf("%s=0x%llx%016llx", label, hi, lo);
}

static int best_record_update(
    BestRecord *global_best,
    const State *state,
    int restart,
    long long step,
    double started,
    int *found_zero
) {
    int updated = 0;
#ifdef _OPENMP
#pragma omp critical(eval212_best)
#endif
    {
        if (tuple_less(state->cost, global_best->cost)) {
            copy_to_best(global_best, state, restart, step);
            updated = 1;
            double elapsed = now_seconds() - started;
            printf(
                "eval212 C-search improvement elapsed=%.2fs restart=%d step=%lld "
                "objective=(%d,%d,%d,%d) ",
                elapsed,
                restart,
                step,
                state->cost.max_excess,
                state->cost.total_excess,
                state->cost.positive_count,
                state->cost.sumsq
            );
            print_bits_hex("A_bits", global_best->A_bits);
            printf(" ");
            print_bits_hex("C_bits", global_best->C_bits);
            printf("\n");
            fflush(stdout);
            if (state->cost.max_excess == 0) {
                *found_zero = 1;
            }
        }
    }
    return updated;
}

int run_search(
    int restarts,
    long long steps_per_restart,
    int threads,
    unsigned long long base_seed,
    double time_limit_sec,
    unsigned long long *out_A_lo,
    unsigned long long *out_A_hi,
    unsigned long long *out_C_lo,
    unsigned long long *out_C_hi,
    int *out_best_restart,
    long long *out_best_step,
    int *out_max_excess,
    int *out_total_excess,
    int *out_positive_count,
    int *out_sumsq,
    int *out_completed_restarts,
    long long *out_total_steps,
    long long *out_accepted_moves
) {
    init_group_tables();
    BestRecord global_best;
    memset(&global_best, 0, sizeof(global_best));
    global_best.cost.max_excess = 1000000;
    global_best.cost.total_excess = 1000000;
    global_best.cost.positive_count = 1000000;
    global_best.cost.sumsq = 1000000;

    int completed_restarts = 0;
    long long total_steps = 0;
    long long accepted_moves = 0;
    int found_zero = 0;
    double started = now_seconds();
    double last_progress = started;

    printf(
        "eval212 C-search setup group=Z7xZ11 index=a+7*b pair_count=%d "
        "|A_pairs|=%d |C_nonzero|=%d thresholds=(inA:%d,notA:%d) "
        "restarts=%d steps_per_restart=%lld threads=%d seed=%llu time_limit=%.1fs\n",
        PAIR_COUNT,
        A_PAIR_SELECTED,
        C_NONZERO_SELECTED,
        IN_A_THRESHOLD,
        NOT_A_THRESHOLD,
        restarts,
        steps_per_restart,
        threads,
        base_seed,
        time_limit_sec
    );
    fflush(stdout);

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic)
#endif
    for (int restart = 0; restart < restarts; ++restart) {
        int local_found;
#ifdef _OPENMP
#pragma omp atomic read
#endif
        local_found = found_zero;
        if (local_found) {
            continue;
        }
        if (now_seconds() - started >= time_limit_sec) {
            continue;
        }

        uint64_t seed_state = (uint64_t)base_seed
            ^ (UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)(restart + 1));
        uint64_t rng = splitmix64(&seed_state);
        if (rng == 0) {
            rng = UINT64_C(0xD1B54A32D192ED03);
        }

        State state;
        if (restart < 350) {
            init_structured_state(&state, restart, &rng);
        } else {
            init_random_state(&state, &rng);
        }
        best_record_update(&global_best, &state, restart, 0, started, &found_zero);

        double t0 = 280.0 * pow(0.84, (double)(restart % 18)) + 10.0 * rng_double(&rng);
        double t1 = 0.025 + 0.075 * rng_double(&rng);
        long long accepted_here = 0;
        long long steps_done = 0;

        for (long long step = 1; step <= steps_per_restart; ++step) {
            steps_done = step;
            if ((step & 4095LL) == 0) {
                int should_stop;
#ifdef _OPENMP
#pragma omp atomic read
#endif
                should_stop = found_zero;
                if (should_stop || now_seconds() - started >= time_limit_sec) {
                    break;
                }
                double now = now_seconds();
#ifdef _OPENMP
#pragma omp critical(eval212_progress)
#endif
                {
                    if (now - last_progress >= 30.0) {
                        last_progress = now;
                        printf(
                            "eval212 C-search progress elapsed=%.1fs completed_restarts=%d "
                            "total_steps=%lld accepted=%lld best=(%d,%d,%d,%d)\n",
                            now - started,
                            completed_restarts,
                            total_steps,
                            accepted_moves,
                            global_best.cost.max_excess,
                            global_best.cost.total_excess,
                            global_best.cost.positive_count,
                            global_best.cost.sumsq
                        );
                        fflush(stdout);
                    }
                }
            }

            int move_roll = rng_bounded(&rng, 100);
            int remove_pair = -1;
            int add_pair = -1;
            int remove_c = -1;
            int add_c = -1;
            int a_pos[4];
            int a_delta[4];
            int c_pos[2];
            int c_delta[2];
            int a_count = 0;
            int c_count = 0;

            if (move_roll < 38 || move_roll >= 82) {
                remove_pair = choose_active_pair(&state, &rng);
                add_pair = choose_inactive_pair(&state, &rng);
                a_pos[0] = pair_rep[remove_pair];
                a_delta[0] = -1;
                a_pos[1] = pair_other[remove_pair];
                a_delta[1] = -1;
                a_pos[2] = pair_rep[add_pair];
                a_delta[2] = 1;
                a_pos[3] = pair_other[add_pair];
                a_delta[3] = 1;
                a_count = 4;
            }
            if (move_roll >= 38) {
                remove_c = choose_active_c(&state, &rng);
                add_c = choose_inactive_c(&state, &rng);
                c_pos[0] = remove_c;
                c_delta[0] = -1;
                c_pos[1] = add_c;
                c_delta[1] = 1;
                c_count = 2;
            }

            int delta_na[M];
            int delta_nc[M];
            Cost next_cost = proposal_cost(
                &state,
                a_pos,
                a_delta,
                a_count,
                c_pos,
                c_delta,
                c_count,
                delta_na,
                delta_nc
            );

            int accept = 0;
            if (tuple_less(next_cost, state.cost)) {
                accept = 1;
            } else {
                double frac = (double)step / (double)steps_per_restart;
                double temp = t0 * pow(t1 / t0, frac);
                double delta_e = energy(next_cost) - energy(state.cost);
                if (delta_e <= 0.0 || rng_double(&rng) < exp(-delta_e / temp)) {
                    accept = 1;
                } else if ((rng_next(&rng) & UINT64_C(0xFFFFF)) == 0) {
                    accept = 1;
                }
            }

            if (accept) {
                apply_proposal(&state, remove_pair, add_pair, remove_c, add_c, delta_na, delta_nc);
                state.cost = next_cost;
                accepted_here += 1;
                if (tuple_less(state.cost, global_best.cost)) {
                    best_record_update(&global_best, &state, restart, step, started, &found_zero);
                    if (state.cost.max_excess == 0) {
                        break;
                    }
                }
            }
        }

#ifdef _OPENMP
#pragma omp atomic
#endif
        completed_restarts += 1;
#ifdef _OPENMP
#pragma omp atomic
#endif
        total_steps += steps_done;
#ifdef _OPENMP
#pragma omp atomic
#endif
        accepted_moves += accepted_here;
    }

    *out_A_lo = (unsigned long long)global_best.A_bits;
    *out_A_hi = (unsigned long long)(global_best.A_bits >> 64);
    *out_C_lo = (unsigned long long)global_best.C_bits;
    *out_C_hi = (unsigned long long)(global_best.C_bits >> 64);
    *out_best_restart = global_best.restart;
    *out_best_step = global_best.step;
    *out_max_excess = global_best.cost.max_excess;
    *out_total_excess = global_best.cost.total_excess;
    *out_positive_count = global_best.cost.positive_count;
    *out_sumsq = global_best.cost.sumsq;
    *out_completed_restarts = completed_restarts;
    *out_total_steps = total_steps;
    *out_accepted_moves = accepted_moves;

    printf(
        "eval212 C-search done elapsed=%.2fs completed_restarts=%d total_steps=%lld "
        "accepted=%lld best_restart=%d best_step=%lld objective=(%d,%d,%d,%d)\n",
        now_seconds() - started,
        completed_restarts,
        total_steps,
        accepted_moves,
        global_best.restart,
        global_best.step,
        global_best.cost.max_excess,
        global_best.cost.total_excess,
        global_best.cost.positive_count,
        global_best.cost.sumsq
    );
    fflush(stdout);

    return global_best.cost.max_excess == 0 ? 1 : 0;
}
