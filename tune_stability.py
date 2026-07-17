from pathlib import Path

c = Path("src/swarm_sar/config.py"); s = c.read_text(encoding="utf-8")
if "reward_scale" not in s:
    s = s.replace("    grad_clip: float = 0.5\n",
                  "    grad_clip: float = 0.5\n    reward_scale: float = 1.0\n")
    c.write_text(s, encoding="utf-8"); print("config.py patched")
else: print("config.py already ok")

m = Path("src/swarm_sar/training/mappo.py"); s = m.read_text(encoding="utf-8")
old = "b.rewards.append(rew.get(self.env.possible_agents[i], 0.0))"
new = "b.rewards.append(rew.get(self.env.possible_agents[i], 0.0) * self.cfg.train.reward_scale)"
if new not in s:
    assert old in s, "reward line not found"; s = s.replace(old, new)
    m.write_text(s, encoding="utf-8"); print("mappo.py patched")
else: print("mappo.py already ok")

Path("configs/training/mappo_easy.yaml").write_text("""env:
  num_drones: 3
  max_steps: 150
  world: { size: 40, n_victims: 4, n_buildings: 6, n_no_fly_zones: 1, n_dynamic_obstacles: 2 }
  faults: { enable: false }
model: { arch: transformer_gnn, hidden_dim: 128 }
train:
  algo: mappo
  lr: 0.0002
  entropy_coef: 0.01
  reward_scale: 0.1
  rollout_len: 1024
  value_coef: 0.5
  gamma: 0.99
  clip: 0.2
  epochs: 4
  total_timesteps: 1000000
  seed: 0
log: { experiment_name: mappo_easy }
""", encoding="utf-8")
print("mappo_easy.yaml retuned"); print("DONE")
