# Configuration

Everything is driven by nested dataclasses in `swarm_sar/config.py`, loadable from
YAML and overridable in code. A run is fully described by its resolved config,
which is serialized next to the logs.

```python
from swarm_sar.config import load_config
cfg = load_config("configs/training/mappo_transformer_gnn.yaml",
                  env={"num_drones": 6})           # kwargs override YAML
cfg.env.comm.packet_loss = 0.2                       # or edit fields directly
```

## Key groups

| Group | Examples |
|-------|----------|
| `env` | `num_drones`, `max_steps`, `max_speed_mps`, `obs_patch`, `wind_max_mps` |
| `env.world` | `size`, `n_buildings`, `n_victims`, `n_fire_zones`, `randomize_each_episode` |
| `env.sensors` | `camera_range_m`, `gps_noise_std_m`, `gps_drift_m_per_s`, `detection_base_prob` |
| `env.battery` | `capacity_wh`, `move_draw_w`, `low_battery_frac`, `degradation_per_cycle` |
| `env.comm` | `range_m`, `latency_steps`, `packet_loss`, `bandwidth_msgs` |
| `env.faults` | per-step probabilities for drone/GPS/camera/comm/motor faults |
| `env.reward` | every reward/penalty weight |
| `model` | `arch`, `hidden_dim`, `n_heads`, `gnn_type`, `gnn_message_rounds` |
| `train` | `algo`, `policy_sharing`, `lr`, `gamma`, `clip`, `centralized_critic`, `seed` |
| `task_alloc` | `strategy` ∈ {random, nearest, hungarian, auction, rl} |
| `log` | `out_dir`, `experiment_name`, `seeds`, `log_tensorboard` |

Ready-made configs live in `configs/` (`env/small.yaml`, `env/large.yaml`,
`training/mappo_*.yaml`, `experiments/ablation_*.yaml`).
