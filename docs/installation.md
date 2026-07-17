# Installation

Swarm-SAR is layered so you only install what you need.

## 1. Core (runs the full simulation, figures, dashboard, tests)

Requires only NumPy, SciPy, Matplotlib, NetworkX, pandas, imageio, PyYAML.

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -e .
python scripts/run_demo.py --episodes 1 --figures   # smoke test
```

> No GPU, AirSim or ROS2 is required for the core path — everything in the paper's
> measured results can be reproduced here.

## 2. Reinforcement-learning extras (train neural policies)

Adds PyTorch, PyTorch-Geometric, Gymnasium, PettingZoo, Ray RLlib, TensorBoard.

```bash
pip install -e ".[rl]"
python scripts/train.py --config configs/training/mappo_transformer_gnn.yaml
```

A CUDA GPU is strongly recommended for the Transformer+GNN model. PyTorch-Geometric
is optional — the GNN falls back to a built-in attention layer if it is absent.

## 3. Visualization extras (Plotly, Streamlit, OpenCV, MP4 export)

```bash
pip install -e ".[viz]"
streamlit run src/swarm_sar/dashboard/app.py
```

MP4 export additionally needs `ffmpeg` on the PATH (or `imageio-ffmpeg`); GIF export
works out of the box via Pillow.

## 4. High-fidelity simulation — AirSim + Unreal Engine (optional)

```bash
pip install -r requirements-sim.txt        # airsim, msgpack-rpc-python
```

1. Install Unreal Engine 4.27+ and build/download an AirSim environment.
2. Configure `settings.json` with N `SimpleFlight` vehicles named `Drone0..DroneN`.
3. Launch the environment, then use `swarm_sar.environment.airsim_adapter.AirSimSwarmEnv`
   as a drop-in backend (same `reset`/`step` API as the NumPy env).

## 5. ROS2 deployment (optional)

The `ros2_ws/src/swarm_sar_ros` package provides a per-drone `PolicyNode` (one node
per drone = decentralized execution) that works against AirSim's `airsim_ros_pkgs`
or real PX4/MAVROS drones.

```bash
cd ros2_ws && colcon build && source install/setup.bash
ros2 run swarm_sar_ros policy_node
```

Tested with ROS2 Humble on Ubuntu 22.04.

## Docker

```bash
docker build -t swarm-sar .
docker compose up swarm-sar      # demo + figures -> ./results
docker compose up dashboard      # http://localhost:8501
```
