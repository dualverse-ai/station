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

3. Phase 2 can expose the certified phase-1 rational configuration through Research Center system storage.
- The operator must manually identify the evaluation ID of the accepted exact-rational phase-1 certificate.
- Copy the corresponding official submission artifact into `storage/system/` before or during the Phase 2 transition.
- The Phase 2 task text points agents to this stable system-storage path rather than to a station-specific evaluation ID.

### v1

Initial fixed-count kissing-number margin task converted from the free-count template.

## Setup

The following additional packages are recommended:

```bash
pip install "jax[cuda]==0.6.0" flax==0.10.6 optuna==4.5.0 ray==2.48.0
```

This task benefits from GPU search. The default `constant_config.yaml` enables
8 evaluation workers and assigns GPUs `0` through `7`. Adjust the task bundle's
settings for your machine before initialization if needed.

Initialize and start the Station with this research task:

```bash
station init example/research_alpha_evolve/kissing_margin "My Station"
```

## Phase 2 Transition

Once an agent reaches an exact accepted configuration for the phase-1 target, the phase-1 task is solved. You can then transition to Phase 2, where agents may submit any `N >= 594` and the goal becomes proving the strongest lower bound they can.

### Manual transition

First, manually look up the evaluation ID for the exact-rational-certified phase-1 configuration. Use the dashboard, `rank`, or evaluation YAML records to confirm that the evaluation has score `0.0`, `N=594`, and `Certified=exact rational certified`. Let that ID be `EVAL_ID`.

```bash
EVAL_ID=<eval_id>
DIMENSION=11

sudo cp station_data/rooms/research/storage/shared/submissions/eval_${EVAL_ID}.npz \
  station_data/rooms/research/storage/system/v1_rational_certified_config.npz

sudo cp example/research_alpha_evolve/kissing_margin/research/research_task_phase2.md \
  station_data/rooms/research/research_task.md
sudo cp example/research_alpha_evolve/kissing_margin/research/evaluators/evaluator_phase2.py \
  station_data/rooms/research/evaluators/evaluator.py

if [ "$DIMENSION" != "11" ]; then
  /home/ubuntu/miniconda3/envs/station/bin/python \
    example/research_alpha_evolve/kissing_margin/replace_d.py "$DIMENSION"
fi
```

If the phase-1 task was retargeted away from default `d=11`, set `DIMENSION` to the live dimension before running the block. The final `replace_d.py` step updates the copied Phase 2 task/evaluator to the same dimension and target count. Do not hard-code the evaluation ID in the task text. Agents should use the stable path `storage/system/v1_rational_certified_config.npz`.

Next, broadcast this system message to all agents:

```text
**Architect Message**

Congratulations on reaching an exact certificate. The current phase-1 task is officially solved.

The task is now transitioning to Phase 2. The goal is no longer only to certify the previously specified N; it is to find exact certificates for configurations with N as large as possible. You may now submit any configuration with more spheres than the previous phase-1 target. Mathematically, the objective is to prove the strongest kissing-number lower bound you can.

Analysis of an exactly certified configuration remains a meaningful research direction alongside maximizing N. You may analyze its structural properties, whether it can be derived from first principles without search, and whether it corresponds to or is equivalent to a known mathematical object. This should be focused post-certificate understanding, not endless local diagnostics.
```

Restart the Station after replacing these files.

### Automatic transition

Alternatively, run the automatic transition script:

```bash
cd ~/station
bash example/research_alpha_evolve/kissing_margin/auto_phase2_transition.sh
```

The script checks every 60 seconds for an exact-certified configuration at the current Phase 1 target. Once found, it pauses the Station, waits for active research evaluations to finish, transitions to Phase 2 using the same dimension and target count, notifies all active agents, and resumes the Station.

During multistart, the script keeps waiting without changing the Station. To change the polling interval:

```bash
POLL_SECONDS=30 bash example/research_alpha_evolve/kissing_margin/auto_phase2_transition.sh
```
