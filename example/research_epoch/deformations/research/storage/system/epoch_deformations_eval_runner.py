from __future__ import annotations

import json
import random
import textwrap
from collections import deque
from typing import Dict, List, Tuple

import sympy as sp
from sage.all import QQ, PolynomialRing, FractionField, MatrixSpace  # type: ignore


RESEARCH_SCORE_NA = "n.a."

SAMPLE_EPS = [0, 1, 2, 3, 5]
MAX_BASIS = 10000
SPECIAL_ISO_RANDOM_TRIALS = 200
SPECIAL_ISO_RANDOM_COEFF_BOUND = 3


def _emit_eval_json(score, details):
    payload = {"score": score, "details": details}
    print(f"EVAL_JSON: {json.dumps(payload, ensure_ascii=False)}", flush=True)


def _parse_sympy_snippet(snippet: str):
    env = {"sp": sp}
    exec(snippet, env)
    x_vars = env.get("x_vars")
    I_gens = env.get("I_gens")
    eps_sym = env.get("eps")

    if x_vars is None or I_gens is None or eps_sym is None:
        raise AssertionError("Snippet must define x_vars, I_gens, eps.")
    if not isinstance(x_vars, tuple):
        raise AssertionError("x_vars must be a tuple of SymPy symbols.")
    if not isinstance(I_gens, list):
        raise AssertionError("I_gens must be a list of SymPy expressions.")

    return x_vars, eps_sym, I_gens


def _require_integer_coeffs(x_vars, eps_sym, I_gens):
    symbols = list(x_vars) + [eps_sym]
    for g in I_gens:
        try:
            sp.Poly(g, *symbols, domain=sp.ZZ)
        except Exception as exc:
            raise AssertionError("All coefficients must be integers.") from exc


def _fiber_ideal(x_vars, eps_sym, I_gens, a: int):
    names = [str(x) for x in x_vars]
    R0 = PolynomialRing(QQ, names, order="lex")
    subs = {eps_sym: sp.Integer(a)}
    gens0 = [R0(str(g.subs(subs))) for g in I_gens]
    return R0, R0.ideal(gens0)


def _normalize_exp(exp, nvars):
    if isinstance(exp, int):
        return (exp,) if nvars == 1 else (exp,) + (0,) * (nvars - 1)
    if isinstance(exp, tuple):
        return exp
    return tuple(exp)


def _standard_monomials_dim(R, I):
    G = I.groebner_basis()
    nvars = R.ngens()
    lm_exps = [_normalize_exp(g.lm().exponents()[0], nvars) for g in G]

    def divisible(exp):
        for lm in lm_exps:
            if all(exp[i] >= lm[i] for i in range(nvars)):
                return True
        return False

    basis = set()
    q = deque()
    zero = (0,) * nvars
    basis.add(zero)
    q.append(zero)

    while q:
        exp = q.popleft()
        for i in range(nvars):
            new_exp = list(exp)
            new_exp[i] += 1
            new_exp = tuple(new_exp)
            if new_exp in basis:
                continue
            if divisible(new_exp):
                continue
            basis.add(new_exp)
            q.append(new_exp)
            if len(basis) > MAX_BASIS:
                raise AssertionError("Ideal does not appear zero-dimensional.")

    return len(basis)


def _edim_via_quadratic_truncation(R, I):
    gens = list(R.gens())
    quad = []
    for i in range(len(gens)):
        for j in range(len(gens)):
            quad.append(gens[i] * gens[j])
    J = R.ideal(list(I.gens()) + quad)

    images = []
    for x in gens:
        red = J.reduce(x)
        if red != 0:
            images.append(red)

    seen = set()
    for p in images:
        mon = _normalize_exp(p.lm().exponents()[0], R.ngens())
        seen.add(mon)
    return len(seen)


def _max_nilpotency_index(R, I):
    max_nilp = 0
    for var in R.gens():
        for k in range(1, 1000):
            if (var ** k) in I:
                max_nilp = max(max_nilp, k)
                break
    return max_nilp


# ------------------------
# Special-fiber isomorphism certificate
# ------------------------


class _SympyQuotientAlgebra:
    def __init__(self, symbols, ideal_gens):
        self.symbols = tuple(symbols)
        self.grob = sp.groebner(list(ideal_gens), *self.symbols, order="lex", domain=sp.QQ)
        self.lm_exps = [tuple(p.monoms()[0]) for p in self.grob.polys]
        self.basis_exps = self._enumerate_standard_monomials()
        self.exp_to_idx = {e: i for i, e in enumerate(self.basis_exps)}
        self.dim = len(self.basis_exps)

    def _enumerate_standard_monomials(self):
        n = len(self.symbols)
        zero = (0,) * n
        q = deque([zero])
        seen = {zero}
        out = []

        def divisible(exp):
            for lm in self.lm_exps:
                if all(exp[i] >= lm[i] for i in range(n)):
                    return True
            return False

        while q:
            exp = q.popleft()
            out.append(exp)
            for i in range(n):
                nxt = list(exp)
                nxt[i] += 1
                tup = tuple(nxt)
                if tup in seen or divisible(tup):
                    continue
                seen.add(tup)
                q.append(tup)
                if len(seen) > MAX_BASIS:
                    raise AssertionError("Special fiber does not appear zero-dimensional.")
        return out

    def nf(self, expr):
        return self.grob.reduce(sp.expand(expr))[1]

    def is_zero(self, expr):
        return self.nf(expr) == 0

    def vec(self, expr):
        rem = self.nf(expr)
        poly = sp.Poly(rem, *self.symbols, domain=sp.QQ)
        out = [sp.Rational(0)] * self.dim
        for mon, coeff in poly.terms():
            idx = self.exp_to_idx.get(tuple(mon))
            if idx is not None:
                out[idx] += sp.Rational(coeff)
        return out


def _spider_basis_exprs(u, v, w):
    basis = [sp.Integer(1)]
    basis.extend([u**i for i in range(1, 8)])
    basis.extend([v**i for i in range(1, 8)])
    basis.extend([w**i for i in range(1, 8)])
    return basis


def _is_independent(vectors):
    if not vectors:
        return False
    M = sp.Matrix(vectors).T
    return M.rank() == len(vectors)


def _inverse_generation_ok(A, residues, basis_exprs):
    cols = [sp.Matrix(A.vec(b)) for b in basis_exprs]
    B = sp.Matrix.hstack(*cols) if cols else sp.zeros(A.dim, 0)
    if B.shape != (A.dim, 22) or B.rank() != 22:
        return False
    for r in residues:
        v = sp.Matrix(A.vec(r))
        try:
            _ = B.LUsolve(v)
        except Exception:
            return False
    return True


def _check_spider_witness(A, residues, u, v, w):
    if not A.is_zero(u * v) or not A.is_zero(u * w) or not A.is_zero(v * w):
        return False
    for t in (u, v, w):
        if not A.is_zero(t**8):
            return False
        if A.is_zero(t**7):
            return False

    basis = _spider_basis_exprs(u, v, w)
    vecs = [A.vec(b) for b in basis]
    if A.dim != 22 or not _is_independent(vecs):
        return False
    if not _inverse_generation_ok(A, residues, basis):
        return False
    return True


def _random_linear_combo(residues, rng):
    coeffs = [rng.randint(-SPECIAL_ISO_RANDOM_COEFF_BOUND, SPECIAL_ISO_RANDOM_COEFF_BOUND) for _ in residues]
    if all(c == 0 for c in coeffs):
        coeffs[rng.randrange(len(coeffs))] = 1
    return sp.expand(sum(coeffs[i] * residues[i] for i in range(len(residues))))


def _special_fiber_isomorphic_to_spider(x_vars, eps_sym, I_gens):
    gens0 = [sp.expand(g.subs({eps_sym: sp.Integer(0)})) for g in I_gens]
    A = _SympyQuotientAlgebra(x_vars, gens0)
    if A.dim != 22:
        return False

    residues = [A.nf(v) for v in x_vars]
    n = len(residues)

    # Deterministic witness search in coordinate residues.
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            u = sp.expand(si * residues[i])
                            v = sp.expand(sj * residues[j])
                            w = sp.expand(sk * residues[k])
                            if _check_spider_witness(A, residues, u, v, w):
                                return True

    # Randomized witness search to avoid fixed-coefficient exploitability.
    rng = random.Random(17)
    for _ in range(SPECIAL_ISO_RANDOM_TRIALS):
        u = _random_linear_combo(residues, rng)
        v = _random_linear_combo(residues, rng)
        w = _random_linear_combo(residues, rng)
        if _check_spider_witness(A, residues, u, v, w):
            return True

    return False


# ------------------------
# Generic intrinsic invariants
# ------------------------


def _monomial_from_exp(R, exp):
    if isinstance(exp, int):
        return R.gen(0) ** exp
    out = R(1)
    for i, e in enumerate(exp):
        if e:
            out *= R.gen(i) ** e
    return out


def _basis_from_generators(field, n: int, mats: List):
    if not mats:
        return None
    cols = []
    for M in mats:
        cols.extend([M.column(j) for j in range(M.ncols())])
    if not cols:
        return None
    G = MatrixSpace(field, n, len(cols))(0)
    for j, c in enumerate(cols):
        for i in range(n):
            G[i, j] = c[i]
    E = G.echelon_form()
    pivots = E.pivots()
    if not pivots:
        return None
    return G.matrix_from_columns(pivots)


def _subspace_dim(B) -> int:
    if B is None:
        return 0
    return B.ncols()


def _generic_intrinsic_invariants(x_vars, eps_sym, I_gens):
    names = [str(x) for x in x_vars]
    eps_name = str(eps_sym)

    eps_ring = PolynomialRing(QQ, [eps_name], order="lex")
    coeff_field = FractionField(eps_ring)
    R = PolynomialRing(coeff_field, names, order="lex")

    gens = [R(str(g)) for g in I_gens]
    I = R.ideal(gens)
    G = I.groebner_basis()

    nvars = R.ngens()
    lm_exps = [_normalize_exp(g.lm().exponents()[0], nvars) for g in G]

    def divisible(exp):
        for lm in lm_exps:
            if all(exp[i] >= lm[i] for i in range(nvars)):
                return True
        return False

    basis_exps = []
    seen = set()
    q = deque()
    zero = (0,) * nvars
    q.append(zero)
    seen.add(zero)
    while q:
        exp = q.popleft()
        basis_exps.append(exp)
        for i in range(nvars):
            nxt = list(exp)
            nxt[i] += 1
            nxt = tuple(nxt)
            if nxt in seen or divisible(nxt):
                continue
            seen.add(nxt)
            q.append(nxt)
            if len(seen) > MAX_BASIS:
                raise AssertionError("Ideal does not appear zero-dimensional.")

    dimA = len(basis_exps)
    if dimA == 0:
        return {
            "length": 0,
            "single_support": False,
            "edim_intr": None,
            "nilp_intr": None,
            "support_point": None,
        }

    exp_to_idx = {e: i for i, e in enumerate(basis_exps)}
    mul_mats = []
    for vi in range(nvars):
        M = MatrixSpace(R.base_ring(), dimA, dimA)(0)
        for col, exp in enumerate(basis_exps):
            prod = R.gen(vi) * _monomial_from_exp(R, exp)
            red = I.reduce(prod)
            for mexp, coeff in red.dict().items():
                nexp = _normalize_exp(mexp, nvars)
                row = exp_to_idx.get(nexp)
                if row is not None:
                    M[row, col] += coeff
        mul_mats.append(M)

    K = R.base_ring()
    I_mat = MatrixSpace(K, dimA, dimA).identity_matrix()
    support_point = []
    shifted = []
    single_support = True
    for M in mul_mats:
        tr = M.trace()
        a = tr / K(dimA)
        support_point.append(a)
        N = M - a * I_mat
        if N ** dimA != 0:
            single_support = False
        shifted.append(N)

    if not single_support:
        return {
            "length": dimA,
            "single_support": False,
            "edim_intr": None,
            "nilp_intr": None,
            "support_point": tuple(support_point),
        }

    B1 = _basis_from_generators(K, dimA, shifted)
    dim_m = _subspace_dim(B1)

    B2 = None
    if B1 is not None and B1.ncols() > 0:
        gens_m2 = [N * B1 for N in shifted]
        B2 = _basis_from_generators(K, dimA, gens_m2)
    dim_m2 = _subspace_dim(B2)
    edim_intr = max(0, dim_m - dim_m2)

    nilp_intr = None
    Bk = B1
    for k in range(1, dimA + 2):
        if Bk is None or Bk.ncols() == 0:
            nilp_intr = k
            break
        gens_next = [N * Bk for N in shifted]
        Bk = _basis_from_generators(K, dimA, gens_next)

    return {
        "length": dimA,
        "single_support": True,
        "edim_intr": edim_intr,
        "nilp_intr": nilp_intr,
        "support_point": tuple(support_point),
    }


def _generic_over_eps_invariants(x_vars, eps_sym, I_gens):
    names = [str(x) for x in x_vars]
    eps_name = str(eps_sym)
    eps_ring = PolynomialRing(QQ, [eps_name], order="lex")
    coeff_field = FractionField(eps_ring)
    poly_ring = PolynomialRing(coeff_field, names, order="lex")
    gens = [poly_ring(str(g)) for g in I_gens]
    I = poly_ring.ideal(gens)
    length = _standard_monomials_dim(poly_ring, I)
    edim = _edim_via_quadratic_truncation(poly_ring, I)
    nilp = _max_nilpotency_index(poly_ring, I)
    return length, edim, nilp


def _length_closeness(length: float, target: int = 22) -> float:
    return max(0.0, 1.0 - abs(float(length) - target) / float(target))


def evaluate_snippet(snippet: str) -> Tuple[float, bool, Dict[str, object]]:
    if snippet is None:
        return RESEARCH_SCORE_NA, False, "construct_deformation returned None"
    if not isinstance(snippet, str):
        raise AssertionError("Submission must return a string.")

    snippet = textwrap.dedent(snippet).strip()
    if not snippet:
        raise AssertionError("Submission returned an empty snippet.")

    x_vars, eps_sym, I_gens = _parse_sympy_snippet(snippet)
    s = len(x_vars)
    if s < 3:
        raise AssertionError("Must use s >= 3 variables.")

    _require_integer_coeffs(x_vars, eps_sym, I_gens)

    lengths: Dict[int, int] = {}
    edims: Dict[int, int] = {}
    nilps: Dict[int, int] = {}

    for a in SAMPLE_EPS:
        R0, I0 = _fiber_ideal(x_vars, eps_sym, I_gens, a)
        lengths[a] = _standard_monomials_dim(R0, I0)
        edims[a] = _edim_via_quadratic_truncation(R0, I0)
        nilps[a] = _max_nilpotency_index(R0, I0)

    special_ok = _special_fiber_isomorphic_to_spider(x_vars, eps_sym, I_gens)

    gen_len, gen_edim, gen_nilp = _generic_over_eps_invariants(x_vars, eps_sym, I_gens)
    intrinsic = _generic_intrinsic_invariants(x_vars, eps_sym, I_gens)
    intr_single = bool(intrinsic.get("single_support"))
    intr_edim = intrinsic.get("edim_intr")
    intr_nilp = intrinsic.get("nilp_intr")

    solved = (
        special_ok
        and lengths[0] == 22
        and gen_len == 22
        and intr_single
        and intr_edim == 1
        and intr_nilp == 22
        and all(lengths[a] == 22 for a in SAMPLE_EPS)
    )

    intrinsic_shape = (
        (1.0 if intr_single else 0.0)
        + (1.0 if intr_edim == 1 else 0.0)
        + (_length_closeness(intr_nilp) if intr_nilp is not None else 0.0)
    ) / 3.0
    generic_score = (_length_closeness(gen_len) + intrinsic_shape) / 2.0
    special_score = (1.0 if special_ok else 0.0) * _length_closeness(lengths[0])
    sample_score = sum(_length_closeness(lengths[a]) for a in SAMPLE_EPS) / len(SAMPLE_EPS)

    score = 50.0 * generic_score + 30.0 * special_score + 20.0 * sample_score
    if not solved:
        score = min(score, 99.999)

    details = {
        "L0": lengths[0],
        "L1": lengths[1],
        "edim1": edims[1],
        "nilp1": nilps[1],
        "gen_len": gen_len,
        "gen_edim_proxy": gen_edim,
        "gen_nilp_proxy": gen_nilp,
        "gen_single_support": intr_single,
        "gen_edim_intr": intr_edim,
        "gen_nilp_intr": intr_nilp,
        "gen_support_point": str(intrinsic.get("support_point")),
        "Message": "Solved" if solved else "Not solved",
    }
    return score, solved, details


def main() -> None:
    try:
        from run import construct_deformation
    except Exception as exc:
        _emit_eval_json(RESEARCH_SCORE_NA, {"Message": f"Import error: {type(exc).__name__}: {exc}"})
        return

    try:
        snippet = construct_deformation()
    except Exception as exc:
        _emit_eval_json(RESEARCH_SCORE_NA, {"Message": f"Runtime error: {type(exc).__name__}: {exc}"})
        raise

    try:
        score, _solved, details = evaluate_snippet(snippet)
        _emit_eval_json(score, details)
    except AssertionError as exc:
        _emit_eval_json(RESEARCH_SCORE_NA, {"Message": f"Verification failed: {type(exc).__name__}: {exc}"})
    except Exception as exc:
        _emit_eval_json(RESEARCH_SCORE_NA, {"Message": f"Evaluation error: {type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
