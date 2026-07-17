# Architecture

## Design principles

- **Modular & configurable.** Each concern (world, sensing, energy, comms, faults,
  allocation, policy, training, evaluation, viz) is an independent module wired
  together by a single typed config. No global state; per-object RNGs.
- **Backend-agnostic.** The learning code talks to a PettingZoo-*parallel* API. The
  fast NumPy backend and the AirSim backend are interchangeable.
- **CTDE.** Actors are fully decentralized (local obs + neighbour messages only);
  the critic is centralized and sees the global state *during training only*.

## Environment API (`SARSwarmEnv`)

```python
obs, info                      = env.reset(seed=0)          # dict keyed by agent id
obs, rewards, term, trunc, info = env.step({agent: action}) # parallel step
```

- **Observation** (per drone, `env.obs_dim` floats): own pose/vel/yaw, altitude,
  battery, k-nearest visible drones (rel-pos + SoC + alive), LiDAR (8 rays), local
  occupancy patch, local victim-probability patch, received-message summary, and
  global progress (fraction found / explored).
- **Action** (discrete, 10): `hover, north, south, east, west, ascend, descend,
  rotate, broadcast, return_to_base`.
- **Reward** (shaped): `+ explore_new_cell, victim_detected, victim_rescued,
  efficient_comm, mission_complete`; `− collision, battery_depleted,
  duplicate_explore, idle, excessive_comm, time`.

## World generation

`SARWorld` builds a 2.5-D occupancy grid each episode: roads are carved, then
buildings/trees/rubble scattered, fire zones with surrounding smoke, rectangular
no-fly zones, charging stations, victims (some under rubble/smoke), and moving
dynamic obstacles. Line-of-sight uses Bresenham traversal; smoke/fire/buildings/
trees occlude vision and shape detection probability.

## Policies

All four encoders feed a shared Actor-Critic (`policies/base.py`):

| Arch | Intra-agent reasoning | Inter-agent reasoning |
|------|-----------------------|-----------------------|
| `mlp` | MLP | — |
| `gnn` | MLP embed | GNN message passing over comm graph |
| `transformer` | Self-attention over obs tokens | — |
| `transformer_gnn` | Transformer | GNN (our proposed model) |

The GNN uses PyTorch-Geometric if available, otherwise a built-in dense GAT layer.

## Training

- `training/mappo.py` — self-contained MAPPO (GAE, clipped objective, centralized
  critic, shared or independent parameters). Used for CI and small runs.
- `training/rllib_trainer.py` — Ray RLlib PPO with parameter sharing for scale.
- `training/rollout.py` — actor-agnostic rollout worker used by evaluation & figures.

## Data flow

```
world ─┐
sensors┼─► SARSwarmEnv ─► obs ─► Encoder(TF+GNN) ─► Actor ─► action
comms ─┘         ▲                                   │
faults           └──────────────── reward ◄──────────┘
                          (centralized critic sees global state; CTDE)
```
