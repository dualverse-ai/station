## Task Specification

This is the Kissing Number lower-bound task for dimension `d=11`.

The phase-1 task asks agents to return exactly `N=594` non-zero direction vectors in `R^11`. The evaluator normalizes each vector to a sphere center and scores the total overlap loss. Lower is better, and score `<= 0` means the submitted configuration is accepted as a non-overlapping kissing configuration.

The task supports floating-point search submissions and preferred exact submissions using integer, rational, or algebraic coordinates.

## Version

This task spec is **v2**.

### v2 Updates

1. Added phase-2 task/evaluator files.
- Phase 2 keeps `d=11` but allows any `N >= 594`.
- Accepted zero-overlap configurations are ranked by larger `N`.

2. Exact-coordinate support is included in the evaluator.
- Integer, rational, and algebraic submissions can be preserved and checked more rigorously when the numerical overlap loss is zero.

### v1

Initial fixed-count kissing-number margin task converted from the free-count template.

## Setup

The following additional packages are recommended:

```bash
pip install "jax[cuda]==0.6.0" flax==0.10.6 optuna==4.5.0 ray==2.48.0
```

Copy this task into station data the same way as other research tasks:

```bash
cp -r example/station_default station_data
cp -r example/research_kissing_margin/research station_data/rooms
cp example/research_kissing_margin/constant_config.yaml station_data/constant_config.yaml
```

This task benefits from GPU search. The default `constant_config.yaml` enables 8 evaluation workers and assigns GPUs `0` through `7`; adjust those settings for your machine before starting the Station.

Then start the Station.

## Phase 2 Transition

Once an agent reaches an exact accepted configuration for the phase-1 target, the phase-1 task is solved. You can then transition to Phase 2, where agents may submit any `N >= 594` and the goal becomes proving the strongest lower bound they can.

First, broadcast this system message to all agents:

```text
**Architect Message**

Congratulations on reaching an exact certificate. The current phase-1 task is officially solved.

The task is now transitioning to Phase 2. The goal is no longer only to certify the previously specified N; it is to find exact certificates for configurations with N as large as possible. You may now submit any configuration with more spheres than the previous phase-1 target. Mathematically, the objective is to prove the strongest kissing-number lower bound you can.

Analysis of an exactly certified configuration remains a meaningful research direction alongside maximizing N. You may analyze its structural properties, whether it can be derived from first principles without search, and whether it corresponds to or is equivalent to a known mathematical object. This should be focused post-certificate understanding, not endless local diagnostics.
```

Then copy the phase-2 files over the live Research Center task and evaluator:

```bash
cp example/research_kissing_margin/research/research_task_phase2.md station_data/rooms/research/research_task.md
cp example/research_kissing_margin/research/evaluators/evaluator_phase2.py station_data/rooms/research/evaluators/evaluator.py
```

Restart the Station after replacing these files.
