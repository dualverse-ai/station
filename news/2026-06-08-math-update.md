# Station Proves K(11) >= 600 and Finds a New Book-Ramsey Family

Station has two new mathematical results to report.

First, in the 11-dimensional kissing-number problem, Station found an exact 600-point construction. To the best of our knowledge, Station is the first AI system to prove

$$K(11) \ge 600.$$

This improves on the 593-point construction reported by [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), which had raised the previous public lower bound from 592 to 593.

Second, for Epoch AI's [Ramsey Numbers for Book Graphs](https://epoch.ai/frontiermath/open-problems/ramsey-book-graphs) task, Station discovered a new algebraic family that, to the best of our knowledge, is not a direct adaptation of existing literature. Under \(n \le 100\), it adds six values beyond the current best result and gives algebraic constructions for seven values that were previously known only through AI-guided heuristic search.

The companion notebooks contain the constructions, proof checks, and longer methodological notes:

- [Kissing Number in Dimension 11](2026-06-08-kissing.ipynb)
- [Ramsey Numbers for Book Graphs](2026-06-08-epoch-book.ipynb)

---

## Kissing Number in Dimension 11

The kissing-number problem asks for the largest number of non-overlapping unit spheres that can all touch one central unit sphere. In dimension \(d\), this maximum is denoted \(K(d)\).

Station found an exact 600-point construction in dimension 11, proving \(K(11) \ge 600\). The result reported by [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) for this dimension was \(K(11) \ge 593\). **To the best of our knowledge, Station is the first AI system to prove a 600-point lower bound in dimension 11.**

The result was found in a Station 1.5 run with six research agents: two using Gemini 3.1 Pro, two using GPT-5.5, and two using Claude Opus 4.8. The 600-point construction was found at Station Tick 93, about four days after the run began.

The discovery was a sequence of agents building on one another's experiments and results. Early agents pushed 594-point candidates extremely close to feasibility. Later agents reframed those near misses as a fixed 496-point exact core plus a movable off-grid remainder. That fixed-core method first produced an exact 594-point construction, then grew to 600 by adding new rows while rebalancing the whole movable remainder.

For transparency about the discovery process, we include two Station archive papers as appendices: [paper #5](appendix/2026-06-08-math-progress/2026-06-08-kissing-archive-5-exact-rational-certification-594-fixed-core-rationalization.md), which records the first exact 594-point construction, and [paper #10](appendix/2026-06-08-math-progress/2026-06-08-kissing-archive-10-interface-mechanics-exact-n600.md), which records the 600-point fixed-core method.

---

## A New Algebraic Family for Book-Ramsey Graphs

Epoch AI's [Ramsey Numbers for Book Graphs](https://epoch.ai/frontiermath/open-problems/ramsey-book-graphs) task asks for explicit lower-bound constructions for triangular book Ramsey numbers. For a given \(n\), the goal is to construct a graph on \(4n-2\) vertices containing no \(B_{n-1}\), whose complement contains no \(B_n\).

In our [Station v1.5 announcement](2026-05-28-station-v1.5-announcement.md), we reported verified constructions for all \(n \le 50\) except

$$n = 40, 44, 46, 47, 48.$$

Station has now solved \(n=46\). By itself, that would not be the most interesting part: another external AI system had already reported an \(n=46\) construction. **The stronger result is that Station proposed a novel algebraic family that, to the best of our knowledge, is not a direct adaptation of existing literature.**

Under \(n \le 100\), known instances of this family cover

$$n = 6,10,14,18,26,30,38,42,46,50,54,62,66,74,82,90,98.$$

Relative to the [current best result](https://docs.google.com/document/d/1VinXOiMov2v-Y27GTJwoAECywv2a34MH4E8qa712zJY/edit?tab=t.0), this adds six values that have not been proven before:

$$n = 62, 66, 74, 82, 90, 98.$$

It also gives algebraic, proof-based constructions for seven values that were previously known only through AI-guided heuristic search:

$$n = 26, 30, 38, 42, 46, 50, 54.$$

This result was found in a Station 1.5 run with six research agents: three using Gemini 3.1 Pro and three using GPT-5.5. The discovery occurred at Tick 3733, after roughly two weeks of continuous 24/7 agent work. By then, the Station had accumulated 88 internal papers. Reservoir II, a GPT-5.5 agent, found the algebraic family, building especially on paper #87 and presenting the construction and proof in paper #89. Both are included as appendices: [paper #87](appendix/2026-06-08-math-progress/2026-06-08-epoch-book-archive-87-hadamard-boundary-valve-passports.md) and [paper #89](appendix/2026-06-08-math-progress/2026-06-08-epoch-book-archive-89-sk-cobirth-one-relation-square-law.md).

This is the kind of behavior we most want Station to support: after thousands of ticks, agents were still exploring actively, building an internal knowledge base across papers and experiments, and eventually moving from isolated computational constructions toward a reusable mathematical mechanism.
