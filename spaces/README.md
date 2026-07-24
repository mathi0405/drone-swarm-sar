---
title: Swarm-SAR Demo
emoji: 🚁
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Swarm-SAR — live demo

Interactive demo of decentralized Multi-Agent RL for cooperative search-and-rescue
drone swarms. Run an episode and replay it frame by frame — positions, battery,
coverage, the communication graph, and mission status.

Source: https://github.com/mathi0405/drone-swarm-sar

## Deploy this Space (from the repo root)

```bash
pip install huggingface_hub
# PowerShell:  $env:HF_TOKEN="hf_your_write_token"
# bash:        export HF_TOKEN=hf_your_write_token
python scripts/deploy_hf_space.py
```

This creates a **Docker** Space (HF removed `streamlit` as a create-time SDK) and
uploads `Dockerfile`, `app.py`, and this `README.md`. The container installs the
package from GitHub with the torch-free `viz` extra and runs the Streamlit
dashboard — no GPU, no training dependencies. The build takes a few minutes.
