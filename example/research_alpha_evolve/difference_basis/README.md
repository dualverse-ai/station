## Developer Notes

This package is the difference-basis upper-bound task template v2. It comes from
Problem 6.7 in the AlphaEvolve paper.

Agent-facing text lives in `research/research_task.md`. Keep that file focused
on the mathematical objective, accepted format, scoring rule, and baseline. Do
not include paper provenance or maintainer notes there. In v2, do not expose the
AlphaEvolve construction score, target threshold, or construction details in
agent-facing text or evaluator output.

Maintainer-facing defaults:

- Primary score: `|B|^2 / n`, lower is better
- Ranking: evaluator returns `sort_key=(-score, n, -|B|)`
- Historical AlphaEvolve upper-bound reference: `2.6390`
- Submitted object: one interval length `n` and one finite integer difference basis `B`
- Maximum verified `n`: `50000000`
- Maximum basis size: `20000`
- Maximum absolute basis element: `10^12`
- Execution: one station-managed GPU per official submission
- CPU allocation: 10 CPUs per evaluation

The Research Center task verifies upper-bound constructions only. The original
problem also asks for lower bounds, but lower bounds require mathematical proof
rather than a finite construction check.

The agent-facing "Mathematical Exploration Goal" section is based on the
AlphaEvolve paper's discussion of Problem 6.7: the broader scientific objective
is to establish strong upper and lower bounds for the asymptotic interval
difference-basis constant and to produce interpretable constructions or
structural evidence, not only a single finite score.

## Setup

```bash
station init example/research_alpha_evolve/difference_basis "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.
