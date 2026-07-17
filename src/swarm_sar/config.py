"""Typed, hierarchical configuration for Swarm-SAR."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

class WorldConfig(BaseModel):
    size: int = 64
    cell_size_m: float = 2.0
    max_altitude_m: float = 40.0
    n_buildings: int = 14
    n_trees: int = 40
    n_rubble: int = 30
    n_fire_zones: int = 4
    smoke_spread: float = 1.8
    n_dynamic_obstacles: int = 6
    n_charging_stations: int = 3
    n_victims: int = 8
    n_no_fly_zones: int = 3
    randomize_each_episode: bool = True
    enable_fire_spread: bool = False
    fire_spread_prob: float = 0.04
    smoke_spread_prob: float = 0.12
    smoke_decay_prob: float = 0.03
    enable_collapses: bool = False
    collapse_prob: float = 0.002
    moving_victims: bool = False
    victim_move_prob: float = 0.04
    weather_enabled: bool = False
    wind_gust_prob: float = 0.04
    wind_gust_mps: float = 2.0
    min_visibility: float = 0.55

class SensorConfig(BaseModel):
    camera_range_m: float = 16.0
    camera_fov_deg: float = 90.0
    lidar_range_m: float = 20.0
    gps_noise_std_m: float = 0.8
    gps_drift_m_per_s: float = 0.02
    imu_noise_std: float = 0.01
    detection_base_prob: float = 0.85
    detection_false_positive: float = 0.03
    smoke_occlusion: float = 0.6

class BatteryConfig(BaseModel):
    capacity_wh: float = 90.0
    hover_draw_w: float = 120.0
    move_draw_w: float = 180.0
    ascend_extra_w: float = 60.0
    comm_draw_w: float = 8.0
    sensor_draw_w: float = 12.0
    low_battery_frac: float = 0.25
    reserve_frac: float = 0.08
    wh_per_meter_return: float = 0.015
    charge_rate_w: float = 900.0
    degradation_per_cycle: float = 0.004

class CommConfig(BaseModel):
    range_m: float = 30.0
    latency_steps: int = 1
    packet_loss: float = 0.08
    bandwidth_msgs: int = 6
    min_bandwidth_msgs: int = 1
    adaptive_bandwidth: bool = True
    excessive_comm_threshold: int = 20
    auto_broadcast: bool = True
    heartbeat_interval_steps: int = 6
    coverage_delta_threshold: int = 12
    message_types: List[str] = Field(
        default_factory=lambda: ["victim", "coverage", "battery", "target", "heartbeat"]
    )
    priority_by_type: Dict[str, int] = Field(
        default_factory=lambda: {
            "victim": 100,
            "target": 80,
            "battery": 60,
            "coverage": 30,
            "heartbeat": 10,
        }
    )
    compress_payload_cells: int = 40

class FaultConfig(BaseModel):
    enable: bool = True
    drone_failure_prob: float = 0.002
    gps_failure_prob: float = 0.004
    camera_failure_prob: float = 0.004
    comm_loss_prob: float = 0.01
    motor_degradation_prob: float = 0.003
    motor_degradation_factor: float = 0.6

class RewardConfig(BaseModel):
    # Penalties are expressed as negative weights and applied additively.
    explore_new_cell: float = 0.1
    frontier_cell: float = 0.2
    coverage_new_cell: float = 0.1
    victim_detected: float = 2.0
    victim_classified: float = 1.0
    victim_rescued: float = 10.0
    mission_complete: float = 20.0
    duplicate_explore: float = -0.1
    hover_penalty: float = -0.01
    hazard_penalty: float = -5.0
    idle: float = -0.05
    collision: float = -10.0
    useful_broadcast: float = 1.0
    approach_victim: float = 2.0
    rescue_dwell: float = 2.0
    separation: float = 0.04
    near_collision: float = -2.0
    battery_depleted: float = -100.0
    excessive_comm: float = -0.3
    time_penalty: float = -0.02
    mode: str = "shaped"

class EnvConfig(BaseModel):
    num_drones: int = 3
    max_drones: int = 10
    max_steps: int = 200
    # Wall-clock seconds represented by one environment step; used for energy
    # accounting so battery management matters on mission timescales.
    step_seconds: float = 5.0
    max_speed_mps: float = 6.0
    obs_patch: int = 11
    # Side length of the egocentric P x P map patch in the observation
    # (channels: occupancy, own explored mask, own victim belief).
    obs_map_patch: int = 7
    k_nearest_drones: int = 3
    dwell_steps_to_rescue: int = 1
    confirmations_required: int = 1
    collision_radius_m: float = 1.5
    safety_margin_m: float = 3.0
    min_spawn_separation_m: float = 4.0
    log_reward_breakdown: bool = False
    # Kept on the environment so allocation also works during MARL rollouts,
    # not only in the heuristic demonstration controller.  ``task_alloc`` in
    # the top-level config is the canonical user-facing setting; load_config
    # synchronises it here.
    task_allocation_strategy: str = "auction"
    # ascend/descend/rotate are excluded by default: altitude and yaw affect
    # only energy in the 2.5-D simulation, so they act as pure noise actions
    # for the learner.  Re-enable them explicitly for altitude experiments.
    allowed_actions: List[str] = Field(default_factory=lambda: [
        "hover", "north", "south", "east", "west", "broadcast", "return_to_base",
    ])
    world: WorldConfig = Field(default_factory=WorldConfig)
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    comm: CommConfig = Field(default_factory=CommConfig)
    faults: FaultConfig = Field(default_factory=FaultConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    wind_max_mps: float = 3.0
    motion_noise_std: float = 0.05

class ModelConfig(BaseModel):
    arch: str = "transformer_gnn"        # mlp | gru | gnn | transformer | transformer_gnn
    hidden_dim: int = 256
    n_layers: int = 2
    n_heads: int = 4
    gnn_type: str = "gat"
    gnn_message_rounds: int = 2
    use_attention_comm: bool = True
    # Max learned messages an agent may attend to per step (top-k gate).
    comm_bandwidth: int = 4

class TrainConfig(BaseModel):
    algo: str = "mappo"
    policy_sharing: str = "shared"
    total_timesteps: int = 3_000_000
    rollout_len: int = 256
    num_envs: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    minibatches: int = 8
    epochs: int = 4
    grad_clip: float = 0.5
    reward_scale: float = 0.01
    seed: int = 0
    device: str = "auto"
    centralized_critic: bool = True
    checkpoint_every: int = 100_000
    eval_every: int = 50_000
    bc_warmstart_episodes: int = 0
    bc_epochs: int = 2
    imitation_coef: float = 0.0
    curriculum: bool = True
    collapse_action_threshold: float = 0.8

class TaskAllocConfig(BaseModel):
    strategy: str = "auction"

class LogConfig(BaseModel):
    out_dir: str = "results"
    experiment_name: str = "swarm_sar"
    log_tensorboard: bool = True
    log_csv: bool = True
    log_video: bool = True
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])

class Config(BaseModel):
    env: EnvConfig = Field(default_factory=EnvConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    task_alloc: TaskAllocConfig = Field(default_factory=TaskAllocConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

def _deep_merge_dict(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _deep_merge_dict(d[k], v)
        else:
            d[k] = v
    return d

def load_config(path: Optional[str | Path] = None, **overrides: Any) -> Config:
    data = {}
    if path is not None:
        with open(path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_merge_dict(data, overrides)
    cfg = Config.model_validate(data)
    # SARSwarmEnv receives EnvConfig rather than the full Config.  Propagate the
    # top-level experiment choice so train/evaluate/demo all use the same
    # battery-aware allocation strategy.
    cfg.env.task_allocation_strategy = cfg.task_alloc.strategy
    return cfg

__all__ = ["Config", "load_config", "EnvConfig", "ModelConfig", "TrainConfig",
           "WorldConfig", "SensorConfig", "BatteryConfig", "CommConfig",
           "FaultConfig", "RewardConfig", "TaskAllocConfig", "LogConfig"]
