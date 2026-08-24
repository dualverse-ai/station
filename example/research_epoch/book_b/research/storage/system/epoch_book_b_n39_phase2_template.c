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
    C_SIZE = 38,
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
    uint8_t C_full[M];
    int NC[M];
    Cost cost;
} State;

typedef struct {
    u128 C_bits;
    Cost cost;
    int phase;
    int restart;
    long long step;
} BestRecord;

static const int A_FIXED[C_SIZE] = { /* EPOCH_BOOK_B_GENERATED_A */ };

static const int C_SEED[C_SIZE] = { /* EPOCH_BOOK_B_GENERATED_C_SEED */ };

static int add_table[M][M];
static int neg_table[M];
static uint8_t A_full[M];
static int NA[M];
static int CAP[M];

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
}

static void init_fixed_a(void) {
    memset(A_full, 0, sizeof(A_full));
    for (int i = 0; i < C_SIZE; ++i) {
        A_full[A_FIXED[i]] = 1;
    }
    for (int d = 0; d < M; ++d) {
        int na = 0;
        for (int x = 0; x < M; ++x) {
            na += (int)A_full[x] * (int)A_full[add_table[x][d]];
        }
        NA[d] = na;
        if (d == 0) {
            CAP[d] = 0;
        } else {
            CAP[d] = (A_full[d] ? IN_A_THRESHOLD : NOT_A_THRESHOLD) - na;
        }
    }
}

static void initialize_nc(State *state) {
    for (int d = 0; d < M; ++d) {
        int nc = 0;
        for (int x = 0; x < M; ++x) {
            nc += (int)state->C_full[x] * (int)state->C_full[add_table[x][d]];
        }
        state->NC[d] = nc;
    }
}

static Cost cost_from_nc(const int *nc) {
    Cost cost;
    cost.max_excess = 0;
    cost.total_excess = 0;
    cost.positive_count = 0;
    cost.sumsq = 0;
    for (int d = 1; d < M; ++d) {
        int excess = nc[d] - CAP[d];
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
    return 1600.0 * (double)c.max_excess
        + 24.0 * (double)c.total_excess
        + 2.5 * (double)c.positive_count
        + 0.08 * (double)c.sumsq;
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

static void init_seed_state(State *state) {
    memset(state, 0, sizeof(*state));
    for (int i = 0; i < C_SIZE; ++i) {
        state->C_full[C_SEED[i]] = 1;
    }
    initialize_nc(state);
    state->cost = cost_from_nc(state->NC);
}

static void init_random_state(State *state, uint64_t *rng) {
    memset(state, 0, sizeof(*state));
    int residues[M - 1];
    for (int i = 0; i < M - 1; ++i) {
        residues[i] = i + 1;
    }
    for (int i = M - 2; i > 0; --i) {
        int j = rng_bounded(rng, i + 1);
        int tmp = residues[i];
        residues[i] = residues[j];
        residues[j] = tmp;
    }
    state->C_full[0] = 1;
    for (int i = 0; i < C_NONZERO_SELECTED; ++i) {
        state->C_full[residues[i]] = 1;
    }
    initialize_nc(state);
    state->cost = cost_from_nc(state->NC);
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

static int choose_pressure_row(const State *state, uint64_t *rng) {
    int positive[76];
    int tight[76];
    int pos_count = 0;
    int tight_count = 0;
    for (int d = 1; d < M; ++d) {
        int slack = CAP[d] - state->NC[d];
        if (slack < 0) {
            positive[pos_count++] = d;
        }
        if (slack <= 1) {
            tight[tight_count++] = d;
        }
    }
    int roll = rng_bounded(rng, 100);
    if (pos_count > 0 && roll < 72) {
        return positive[rng_bounded(rng, pos_count)];
    }
    if (tight_count > 0 && roll < 94) {
        return tight[rng_bounded(rng, tight_count)];
    }
    return 1 + rng_bounded(rng, M - 1);
}

static int choose_active_for_row(const State *state, int d, uint64_t *rng) {
    int candidates[M];
    int count = 0;
    int neg_d = neg_table[d];
    for (int x = 1; x < M; ++x) {
        if (!state->C_full[x]) {
            continue;
        }
        if (state->C_full[add_table[x][d]] || state->C_full[add_table[x][neg_d]]) {
            candidates[count++] = x;
        }
    }
    if (count > 0) {
        return candidates[rng_bounded(rng, count)];
    }
    return choose_active_c(state, rng);
}

static int choose_inactive_for_row(const State *state, int d, uint64_t *rng) {
    int neg_d = neg_table[d];
    int best[M];
    int best_count = 0;
    int best_score = 1000000;
    for (int tries = 0; tries < 18; ++tries) {
        int x = choose_inactive_c(state, rng);
        int score = (int)state->C_full[add_table[x][d]]
            + (int)state->C_full[add_table[x][neg_d]];
        if (score < best_score) {
            best_score = score;
            best_count = 0;
            best[best_count++] = x;
        } else if (score == best_score && best_count < M) {
            best[best_count++] = x;
        }
        if (best_score == 0 && tries >= 5) {
            break;
        }
    }
    if (best_count > 0) {
        return best[rng_bounded(rng, best_count)];
    }
    return choose_inactive_c(state, rng);
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
    const int *flip_pos,
    const int *flip_delta,
    int flip_count,
    int *delta_nc
) {
    Cost cost;
    cost.max_excess = 0;
    cost.total_excess = 0;
    cost.positive_count = 0;
    cost.sumsq = 0;
    delta_nc[0] = 0;
    for (int d = 1; d < M; ++d) {
        int dnc = delta_corr_for_shift(state->C_full, flip_pos, flip_delta, flip_count, d);
        delta_nc[d] = dnc;
        int excess = state->NC[d] + dnc - CAP[d];
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

static void apply_proposal(State *state, const int *flip_pos, const int *flip_delta, int flip_count, const int *delta_nc, Cost cost) {
    for (int d = 1; d < M; ++d) {
        state->NC[d] += delta_nc[d];
    }
    for (int i = 0; i < flip_count; ++i) {
        state->C_full[flip_pos[i]] = (uint8_t)((int)state->C_full[flip_pos[i]] + flip_delta[i]);
    }
    state->cost = cost;
}

static u128 bits_after_flips(const State *state, const int *flip_pos, const int *flip_delta, int flip_count) {
    u128 bits = bits_from_full(state->C_full);
    for (int i = 0; i < flip_count; ++i) {
        if (flip_delta[i] > 0) {
            bits |= bit_mask(flip_pos[i]);
        } else {
            bits &= ~bit_mask(flip_pos[i]);
        }
    }
    return bits;
}

static void print_bits_hex(const char *label, u128 bits) {
    unsigned long long hi = (unsigned long long)(bits >> 64);
    unsigned long long lo = (unsigned long long)bits;
    printf("%s=0x%llx%016llx", label, hi, lo);
}

static int update_best_bits(
    BestRecord *global_best,
    u128 c_bits,
    Cost cost,
    int phase,
    int restart,
    long long step,
    double started,
    int *found_zero
) {
    int updated = 0;
#ifdef _OPENMP
#pragma omp critical(eval220_best)
#endif
    {
        if (tuple_less(cost, global_best->cost)) {
            global_best->C_bits = c_bits;
            global_best->cost = cost;
            global_best->phase = phase;
            global_best->restart = restart;
            global_best->step = step;
            updated = 1;
            printf(
                "eval220 fixed-A C improvement elapsed=%.2fs phase=%d restart=%d step=%lld "
                "objective=(%d,%d,%d,%d) ",
                now_seconds() - started,
                phase,
                restart,
                step,
                cost.max_excess,
                cost.total_excess,
                cost.positive_count,
                cost.sumsq
            );
            print_bits_hex("C_bits", c_bits);
            printf("\n");
            fflush(stdout);
            if (cost.max_excess == 0) {
                *found_zero = 1;
            }
        }
    }
    return updated;
}

static int update_best_state(
    BestRecord *global_best,
    const State *state,
    int phase,
    int restart,
    long long step,
    double started,
    int *found_zero
) {
    return update_best_bits(global_best, bits_from_full(state->C_full), state->cost, phase, restart, step, started, found_zero);
}

static void fill_selected_lists(const State *state, int *selected, int *selected_count, int *unselected, int *unselected_count) {
    *selected_count = 0;
    *unselected_count = 0;
    for (int x = 1; x < M; ++x) {
        if (state->C_full[x]) {
            selected[(*selected_count)++] = x;
        } else {
            unselected[(*unselected_count)++] = x;
        }
    }
}

static void exact_local_phase(
    BestRecord *global_best,
    const State *seed,
    double started,
    double total_time_limit_sec,
    double three_swap_cap_sec,
    int *found_zero,
    long long *one_checked,
    long long *two_checked,
    long long *three_checked,
    int *three_complete
) {
    int selected[C_NONZERO_SELECTED];
    int unselected[M - C_SIZE];
    int selected_count = 0;
    int unselected_count = 0;
    fill_selected_lists(seed, selected, &selected_count, unselected, &unselected_count);

    int delta_nc[M];
    int pos[6];
    int del[6];

    for (int ri = 0; ri < selected_count; ++ri) {
        for (int ai = 0; ai < unselected_count; ++ai) {
            pos[0] = selected[ri];
            del[0] = -1;
            pos[1] = unselected[ai];
            del[1] = 1;
            Cost cost = proposal_cost(seed, pos, del, 2, delta_nc);
            *one_checked += 1;
            update_best_bits(global_best, bits_after_flips(seed, pos, del, 2), cost, 1, 0, *one_checked, started, found_zero);
            if (*found_zero) return;
        }
    }

    for (int r1 = 0; r1 < selected_count; ++r1) {
        for (int r2 = r1 + 1; r2 < selected_count; ++r2) {
            for (int a1 = 0; a1 < unselected_count; ++a1) {
                for (int a2 = a1 + 1; a2 < unselected_count; ++a2) {
                    pos[0] = selected[r1];
                    del[0] = -1;
                    pos[1] = selected[r2];
                    del[1] = -1;
                    pos[2] = unselected[a1];
                    del[2] = 1;
                    pos[3] = unselected[a2];
                    del[3] = 1;
                    Cost cost = proposal_cost(seed, pos, del, 4, delta_nc);
                    *two_checked += 1;
                    update_best_bits(global_best, bits_after_flips(seed, pos, del, 4), cost, 2, 0, *two_checked, started, found_zero);
                    if (*found_zero) return;
                }
            }
        }
    }

    double three_started = now_seconds();
    *three_complete = 1;
    if (three_swap_cap_sec <= 0.0) {
        *three_complete = 0;
        return;
    }
    for (int r1 = 0; r1 < selected_count; ++r1) {
        for (int r2 = r1 + 1; r2 < selected_count; ++r2) {
            for (int r3 = r2 + 1; r3 < selected_count; ++r3) {
                for (int a1 = 0; a1 < unselected_count; ++a1) {
                    for (int a2 = a1 + 1; a2 < unselected_count; ++a2) {
                        for (int a3 = a2 + 1; a3 < unselected_count; ++a3) {
                            if (((*three_checked) & 8191LL) == 0) {
                                double now = now_seconds();
                                if (now - three_started >= three_swap_cap_sec || now - started >= total_time_limit_sec) {
                                    *three_complete = 0;
                                    return;
                                }
                            }
                            pos[0] = selected[r1];
                            del[0] = -1;
                            pos[1] = selected[r2];
                            del[1] = -1;
                            pos[2] = selected[r3];
                            del[2] = -1;
                            pos[3] = unselected[a1];
                            del[3] = 1;
                            pos[4] = unselected[a2];
                            del[4] = 1;
                            pos[5] = unselected[a3];
                            del[5] = 1;
                            Cost cost = proposal_cost(seed, pos, del, 6, delta_nc);
                            *three_checked += 1;
                            update_best_bits(global_best, bits_after_flips(seed, pos, del, 6), cost, 3, 0, *three_checked, started, found_zero);
                            if (*found_zero) return;
                        }
                    }
                }
            }
        }
    }
}

static void perturb_seed(State *state, int swaps, uint64_t *rng) {
    init_seed_state(state);
    for (int s = 0; s < swaps; ++s) {
        int remove_c = choose_active_c(state, rng);
        int add_c = choose_inactive_c(state, rng);
        state->C_full[remove_c] = 0;
        state->C_full[add_c] = 1;
    }
    initialize_nc(state);
    state->cost = cost_from_nc(state->NC);
}

int run_fixed_a_c_completion(
    int restarts,
    long long steps_per_restart,
    int threads,
    unsigned long long base_seed,
    double total_time_limit_sec,
    double three_swap_cap_sec,
    unsigned long long *out_C_lo,
    unsigned long long *out_C_hi,
    int *out_best_phase,
    int *out_best_restart,
    long long *out_best_step,
    int *out_max_excess,
    int *out_total_excess,
    int *out_positive_count,
    int *out_sumsq,
    long long *out_one_checked,
    long long *out_two_checked,
    long long *out_three_checked,
    int *out_three_complete,
    int *out_completed_restarts,
    long long *out_total_steps,
    long long *out_accepted_moves
) {
    init_group_tables();
    init_fixed_a();

    State seed;
    init_seed_state(&seed);

    BestRecord global_best;
    memset(&global_best, 0, sizeof(global_best));
    global_best.C_bits = bits_from_full(seed.C_full);
    global_best.cost = seed.cost;
    global_best.phase = 0;
    global_best.restart = 0;
    global_best.step = 0;

    int found_zero = seed.cost.max_excess == 0 ? 1 : 0;
    long long one_checked = 0;
    long long two_checked = 0;
    long long three_checked = 0;
    int three_complete = 0;
    int completed_restarts = 0;
    long long total_steps = 0;
    long long accepted_moves = 0;
    double started = now_seconds();
    double last_progress = started;

    printf(
        "eval220 fixed-A C helper setup group=Z7xZ11 |A|=38 |C|=38 C0_fixed=true "
        "seed_objective=(%d,%d,%d,%d) restarts=%d steps_per_restart=%lld threads=%d "
        "seed=%llu total_time_limit=%.1fs three_swap_cap=%.1fs\n",
        seed.cost.max_excess,
        seed.cost.total_excess,
        seed.cost.positive_count,
        seed.cost.sumsq,
        restarts,
        steps_per_restart,
        threads,
        base_seed,
        total_time_limit_sec,
        three_swap_cap_sec
    );
    fflush(stdout);

    exact_local_phase(
        &global_best,
        &seed,
        started,
        total_time_limit_sec,
        three_swap_cap_sec,
        &found_zero,
        &one_checked,
        &two_checked,
        &three_checked,
        &three_complete
    );

    printf(
        "eval220 exact local done elapsed=%.2fs one_checked=%lld two_checked=%lld "
        "three_checked=%lld three_complete=%d best=(%d,%d,%d,%d)\n",
        now_seconds() - started,
        one_checked,
        two_checked,
        three_checked,
        three_complete,
        global_best.cost.max_excess,
        global_best.cost.total_excess,
        global_best.cost.positive_count,
        global_best.cost.sumsq
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
        if (now_seconds() - started >= total_time_limit_sec) {
            continue;
        }

        uint64_t seed_state = (uint64_t)base_seed
            ^ (UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)(restart + 1));
        uint64_t rng = splitmix64(&seed_state);
        if (rng == 0) {
            rng = UINT64_C(0xD1B54A32D192ED03);
        }

        State state;
        if (restart == 0) {
            init_seed_state(&state);
        } else if (restart < 7000) {
            perturb_seed(&state, 1 + (restart % 18), &rng);
        } else {
            init_random_state(&state, &rng);
        }
        update_best_state(&global_best, &state, 4, restart, 0, started, &found_zero);

        double t0 = 135.0 * pow(0.88, (double)(restart % 24)) + 4.0 * rng_double(&rng);
        double t1 = 0.018 + 0.055 * rng_double(&rng);
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
                if (should_stop || now_seconds() - started >= total_time_limit_sec) {
                    break;
                }
                double now = now_seconds();
#ifdef _OPENMP
#pragma omp critical(eval220_progress)
#endif
                {
                    if (now - last_progress >= 30.0) {
                        last_progress = now;
                        printf(
                            "eval220 fixed-A C progress elapsed=%.1fs completed_restarts=%d "
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

            int flip_pos[4];
            int flip_delta[4];
            int flip_count = 2;
            int row = choose_pressure_row(&state, &rng);
            int biased = rng_bounded(&rng, 100) < 72;

            if (biased) {
                flip_pos[0] = choose_active_for_row(&state, row, &rng);
                flip_pos[1] = choose_inactive_for_row(&state, row, &rng);
            } else {
                flip_pos[0] = choose_active_c(&state, &rng);
                flip_pos[1] = choose_inactive_c(&state, &rng);
            }
            flip_delta[0] = -1;
            flip_delta[1] = 1;

            if (rng_bounded(&rng, 100) < 13) {
                int r2 = choose_active_c(&state, &rng);
                int a2 = choose_inactive_c(&state, &rng);
                int guard = 0;
                while (r2 == flip_pos[0] && guard++ < 20) {
                    r2 = choose_active_c(&state, &rng);
                }
                guard = 0;
                while (a2 == flip_pos[1] && guard++ < 20) {
                    a2 = choose_inactive_c(&state, &rng);
                }
                if (r2 != flip_pos[0] && a2 != flip_pos[1]) {
                    flip_pos[2] = r2;
                    flip_delta[2] = -1;
                    flip_pos[3] = a2;
                    flip_delta[3] = 1;
                    flip_count = 4;
                }
            }

            int delta_nc[M];
            Cost next_cost = proposal_cost(&state, flip_pos, flip_delta, flip_count, delta_nc);

            int accept = 0;
            if (tuple_less(next_cost, state.cost)) {
                accept = 1;
            } else {
                double frac = (double)step / (double)steps_per_restart;
                double temp = t0 * pow(t1 / t0, frac);
                double delta_e = energy(next_cost) - energy(state.cost);
                if (delta_e <= 0.0 || rng_double(&rng) < exp(-delta_e / temp)) {
                    accept = 1;
                } else if ((rng_next(&rng) & UINT64_C(0x7FFFF)) == 0) {
                    accept = 1;
                }
            }

            if (accept) {
                apply_proposal(&state, flip_pos, flip_delta, flip_count, delta_nc, next_cost);
                accepted_here += 1;
                if (tuple_less(state.cost, global_best.cost)) {
                    update_best_state(&global_best, &state, 4, restart, step, started, &found_zero);
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

    *out_C_lo = (unsigned long long)global_best.C_bits;
    *out_C_hi = (unsigned long long)(global_best.C_bits >> 64);
    *out_best_phase = global_best.phase;
    *out_best_restart = global_best.restart;
    *out_best_step = global_best.step;
    *out_max_excess = global_best.cost.max_excess;
    *out_total_excess = global_best.cost.total_excess;
    *out_positive_count = global_best.cost.positive_count;
    *out_sumsq = global_best.cost.sumsq;
    *out_one_checked = one_checked;
    *out_two_checked = two_checked;
    *out_three_checked = three_checked;
    *out_three_complete = three_complete;
    *out_completed_restarts = completed_restarts;
    *out_total_steps = total_steps;
    *out_accepted_moves = accepted_moves;

    printf(
        "eval220 fixed-A C done elapsed=%.2fs one_checked=%lld two_checked=%lld "
        "three_checked=%lld three_complete=%d completed_restarts=%d total_steps=%lld "
        "accepted=%lld best_phase=%d best_restart=%d best_step=%lld objective=(%d,%d,%d,%d)\n",
        now_seconds() - started,
        one_checked,
        two_checked,
        three_checked,
        three_complete,
        completed_restarts,
        total_steps,
        accepted_moves,
        global_best.phase,
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
