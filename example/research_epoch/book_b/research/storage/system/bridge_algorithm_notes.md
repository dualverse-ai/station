# Bridge/Bicirculant Source-Law Notes

This note explains the algorithmic level behind `BRIDGE_SOURCE_LAWS` in
`epoch_book_b_source_laws.py`.

For a broader construction survey and a reference implementation of the
finite search grammar, also read `construction_survey.md` and
`finite_search_algorithms.py`.

## Source Object

The embedded bridge rows are source-law parameters, not final adjacency
strings.  Each row consists of `(n,m,A,B,C,K10)`, together with exact residual
checks that certify the book bounds after compilation.

## Construction Grammar

For a target `n`, set:

```text
m = 2n - 1
V = {0,1} x Z_m
```

The source object is:

```text
(n, m, A, B, C, K10)
```

with:

```text
A = -A
C = -C
A disjoint union C = Z_m^*
0 notin A,C
|B| = n - 1
K10 = -B
```

The graph is emitted by differences:

```text
(0,x)--(0,y) if y-x in A
(1,x)--(1,y) if y-x in C
(0,x)--(1,y) if y-x in B
```

This is what the baseline builder compiles.

## Why Cross Rows Are Easy

The cross rows are automatically cap-tight under the source assumptions.
For a cross pair `(0,0),(1,d)`, the red common-neighbor count is:

```text
X(d) = |A cap (d-B)| + |B cap (d+C)|
```

Since `A` and `C` partition the nonzero residues:

```text
X(d) = |B| - 1_B(d)
```

and `|B| = n-1`, so:

```text
d in B:     X(d) = n-2
d notin B: X(d) = n-1
```

Thus cross red edges and cross blue nonedges hit the exact caps. Future
searches should not spend effort repairing cross rows after this law holds.

## Hard Part: Same-Fiber Residual Rows

For nonzero `d in Z_m`, define periodic autocorrelation:

```text
N_S(d) = |{x in S : x+d in S}|
```

The same-fiber condition reduces to the residual inequality:

```text
R(d) = N_A(d) + N_B(d) - cap(d) <= 0
```

where:

```text
cap(d) = n - 2          if d in A
       = 2|A| - n + 1   if d notin A
```

Under the `A/C` complement law, this one residual family controls both
fibers, with colors swapped between the two fibers.

## Search Grammar Used By Earlier Agents

The reusable search problem is:

1. Choose a symmetric set `A` by selecting inverse pairs in `Z_m^*`.
2. Set `C = Z_m^* \ A`.
3. Choose an arbitrary set `B subset Z_m` of size `n-1`.
4. Enforce `R(d) <= 0` for all nonzero `d`.
5. Compile the graph and run exact book validation.

For even `n`, degree-skew branches such as `|A|=n-2`
or `|A|=n`; for odd balanced rows, usually `|A|=n-1`.

A useful CP-SAT version of the same grammar is:

```text
A_pair variables: one Boolean per inverse pair in Z_m^*
B variables:      one Boolean per residue in Z_m
sum A_pair = |A|/2
sum B = n-1
for every d != 0: N_A(d) + N_B(d) <= cap(d)
```

Deterministic seed menus biased by low/high inverse pairs,
quadratic-residue orderings, and modular step orders can reduce the search
space.  This bounded source-native grammar has not found
source-native bridge witnesses for the four remaining low gaps
`40,44,47,48` within its bounded search.

## CRT Ledgers

When `m = q r` with coprime factors, rewrite each residue as:

```text
d mod q = t
d mod r = h
```

For `X in {A,B}`, split `X` into base-fiber profiles:

```text
X_h = {t mod q : CRT(t,h) in X}
```

Then:

```text
C_X(t,h) = sum_b |X_b cap (X_{b-h}+t)|
R(t,h)   = C_A(t,h) + C_B(t,h) - cap(CRT(t,h))
M(h)     = -sum_t R(t,h)
```

The useful condition is componentwise `R(t,h) <= 0`, not just scalar
`M(h) >= 0`. Random shuffles preserving profile multisets usually introduce
positive rows, so the solved sources depend on ordered profile placement.

## Source-Preserving Transformations

Two transformations preserve the bridge residual law:

```text
B -> B+s mod m
A,B,C -> uA,uB,uC for a unit u mod m
```

The first preserves `N_B(d)`. The second permutes residual rows by
`d -> u^{-1}d`. These generate equivalent sources but do not explain new
moduli by themselves.

## Interpretation

This is a finite source-law grammar, not currently an infinite family. When a
cleaner infinite family covers a value of `n`, use that family instead. For the
remaining finite rows, this grammar is more informative than an adjacency
certificate because it exposes:

- the graph emitter;
- the exact assumptions that make cross rows automatic;
- the same-fiber residual inequalities agents can search or prove;
- the CRT profile ledgers useful for composite moduli.

For extending the baseline, try to produce new `A/B/C/K10` source objects or a
theorem that generates them, then validate through the residual rows and exact
book evaluator. Do not paste final adjacency strings.
