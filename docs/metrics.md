# Metrics & the Swarm Intelligence Score

`swarm_sar.evaluation.metrics.episode_metrics(log)` returns, per episode:

| Metric | Meaning |
|--------|---------|
| `coverage` | fraction of the map explored |
| `mission_success` | 1 if all victims rescued |
| `victims_rescued` / `victims_detected` | counts |
| `avg_rescue_time` | mean timestep of rescue |
| `path_efficiency` | explored area per unit distance travelled (normalized) |
| `collision_rate` | collisions per timestep |
| `energy_wh` | total energy consumed |
| `communication_efficiency` | delivered / attempted messages |
| `exploration_entropy` | normalized spatial spread of the search (0–1) |
| `robustness` | alive-fraction × rescue-fraction |
| `swarm_intelligence_score` | composite, below |

Plus, at the evaluation level: **robustness** (SIS with faults ÷ SIS without),
**generalization** (test-seed SIS ÷ train-seed SIS), and **inference time** (ms/decision).

## Swarm Intelligence Score (SIS)

A single number in **[0, 100]** combining five normalized dimensions:

```
SIS = 100 · Π_k  cᵢ^{wᵢ}          (weighted geometric mean)
```

with `c = {coverage, rescue, energy, communication, safety}` and default weights
`{0.25, 0.30, 0.15, 0.10, 0.20}`.

**Why geometric?** A weighted *sum* lets a swarm hide a fatal weakness (e.g. huge
coverage bought with constant collisions). The weighted *geometric* mean collapses
toward zero if **any** dimension is near zero, so a high SIS requires *balanced*
competence — a better operationalization of "swarm intelligence". Example: a swarm
scoring 1.0 on everything except safety = 0.01 drops from 100 to ≈ 40.

```python
from swarm_sar.evaluation.metrics import swarm_intelligence_score
swarm_intelligence_score({"coverage":.6,"rescue":.6,"energy":.6,"communication":.6,"safety":.6})  # balanced
```
