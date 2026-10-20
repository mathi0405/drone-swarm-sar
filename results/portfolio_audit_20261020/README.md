# Communication study — 20 October 2026

## Scope

Measured locally with the scripted heuristic auction controller, not a learned policy. Self-contained simulation, 3 drones, 180 steps per episode, 20 map seeds (200–219), 60 episodes total. Population standard deviation describes map variation, not independent training-seed uncertainty.

| Packet loss | Mean rescued | SIS mean ± std |
|---|---:|---:|
| 0% | 6.35/8 | 65.65 ± 22.19 |
| 30% | 6.00/8 | 64.31 ± 21.97 |
| 100% | 6.75/8 | 72.67 ± 8.54 |

## Interpretation

The disabled condition has the highest mean rescue count in this run. This does not establish that communication is harmful: at 100% loss the script also sets radio range to zero, changing peer visibility, and the conditions share map seeds but not necessarily matched stochastic trajectories. No statistical superiority claim is made. This is a heuristic reference experiment, not validation of learned communication or hardware deployment.

The next experiment should isolate message content while keeping the physical communication graph fixed, evaluate frozen learned checkpoints, and report paired outcomes and uncertainty across independent training seeds. No new neural training or shuffled-message experiment is claimed here.

## Reproduce

```bash
python scripts/comm_study.py --losses 0 0.3 1 --seeds 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 --out results/portfolio_audit_20261020/communication-study.png
```

The adjacent JSON records the exact code revision, clean working-tree state, resolved configurations, command and per-episode metrics. The MP4 is an illustrative replay of map seed 200 under the two endpoint conditions; it is not the aggregate benchmark.
