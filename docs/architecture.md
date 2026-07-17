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

- **Observation** (per drone, `env.obs_dim` floats — an 8-frame temporal stack):
  own noisy-GPS pose, velocity, altitude, battery; a victim-belief summary
  (count, density, two nearest believed victims); locally sensed fire/smoke
  distance and density; an egocentric `obs_map_patch`² map patch with three
  channels (occupancy, own explored mask, own victim belief); LiDAR (8 rays);
  the k-nearest in-range peers with intent (rel-pos, rescue target, mode); and
  a received-message summary (count, age, signal strength, loss rate).
- **Action** (discrete, 10; 7 enabled by default): `hover, north, south, east,
  west, broadcast, return_to_base` (+ `ascend, descend, rotate`, masked out by
  default because altitude/yaw only affect energy in the 2.5-D sim).
- **Reward** (shaped): `+ explore/frontier/team-new cells, victim detected/
  classified/rescued, mission_complete, useful_broadcast, separation`;
  `− collision, near_collision, hazard, battery_depleted, duplicate_explore,
  hover/idle, excessive_comm, time`. Sparse mode keeps only mission/safety
  events.

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
