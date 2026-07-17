# Reward design

This document records *why* the reward looks the way it does, with the audit
that motivated it. The implementation lives in
`swarm_sar/environment/reward_engine.py`; all weights are in
`RewardConfig` (`swarm_sar/config.py`).

## Principles

1. **Mission events dominate.** A rescue must out-earn anything a drone can
   collect from dense shaping in the same time. If shaping ever competes with
   the mission, the policy optimizes the shaping.
2. **Every dense term is bounded per step.** Cell-count signals are normalized
   by the sensor footprint (~197 cells at the default camera radius), so each
   dense term contributes at most `|weight|` per drone per step — invariant to
   camera range, map size and step rate.
3. **Rescue is time-critical (triage).** The rescue reward is scaled by
   `(floor + (1-floor)·(1 - t/T)) · (0.5 + 0.5·severity)`: severe victims
   rescued early pay full value; late, low-severity rescues pay as little as
   ~20% of it. This turns "eventually rescue everyone" into a scheduling
   problem — which victim first, given distance, severity and battery.
4. **Safety is an event, not shaping.** Collisions, hazard entry and battery
   depletion fire in *both* shaped and sparse modes, so reward ablations never
   compare against a safety-blind baseline.

## The audit that forced the redesign

A *perfect* heuristic episode on the benchmark env (8/8 rescued, 97% coverage,
71 steps) was scored with both reward versions:

| component                    | unnormalized (before) | normalized (after) |
|------------------------------|----------------------:|-------------------:|
| exploration bonuses          |                +1025.5 |              +42.6 |
| duplicate-search penalty     |            **−2311.7** |              −11.7 |
| rescue + detection events    |                  +96.0 |     +96.0 (·urgency) |
| mission complete             |                  +60.0 |              +60.0 |
| collisions / hazard / misc   |                 −144.2 |             −144.2 |
| **episode total**            |            **−1221.6** |          **+74.8** |

Before normalization the duplicate-search term was charged **per already-seen
cell in the footprint, per step** — up to −19.7 per drone-step, recurring. Two
consequences:

* *Structurally negative returns*: no policy could score above ≈ −1200, and
  the critic had to fit huge, high-variance targets (V-loss 300–430).
* *A gradient that fights the mission*: flying back through explored terrain
  to a detected victim cost ~−20/step against +2 approach shaping and +10 for
  the rescue, so improving at the shaping meant getting worse at rescuing.

After normalization the same term is a bounded nudge (≤ −0.1 per drone-step)
and the biggest penalties in a competent episode are genuine safety failures.

## Per-step bounds (shaped mode, per drone)

| term                 | weight | bound/step | recurs? |
|----------------------|-------:|-----------:|---------|
| explore (own new)    |   +1.0 |      ≤ 1.0 | one-shot per cell |
| team coverage        |   +0.5 |      ≤ 0.5 | one-shot per cell |
| duplicate search     |   −0.1 |     ≥ −0.1 | recurring |
| frontier             |  +0.05 |      ≤ 0.2 | recurring |
| approach victim      |   +2.0 |     ≤ ±2.0 | recurring |
| rescue dwell         |   +2.0 |      ≤ 2.0 | recurring |
| separation / near-miss | +0.04 / −2.0 | bounded | recurring |
| useful broadcast     |   +1.0 |      ≤ 1.0 | only for *new* info |
| time / hover / idle  | −0.02 / −0.01 / −0.05 | bounded | recurring |
| **victim detected**  |   +2.0 |        — | event |
| **victim rescued**   |  +10.0 × urgency ∈ [0.15, 1.0] | — | event |
| **mission complete** |  +20.0 |        — | event |
| **collision**        |  −10.0 |        — | event |
| **hazard / battery** | −5.0 / −100.0 | — | event |

Sanity checks are enforced by `tests/test_reward_engine.py`, including an
integration test asserting a competent policy earns a **positive** episode
return.

## Reading training curves

* Episode return should now sit roughly in **[−300, +300]**; a competent
  policy is positive, and mission progress (rescues) is visible in the return
  rather than drowned by shaping.
* Curriculum stage changes still shift the return level (different maps,
  victim counts and episode lengths); the trainer logs `curriculum_stage` and
  prints every transition so level shifts are attributable.
* With `curriculum_mode: performance` (default), stages advance when the
  policy clears the per-stage rescue threshold on a deterministic evaluation
  of its *current* stage; the linear schedule remains a fallback floor so
  training always reaches the full benchmark within budget.
