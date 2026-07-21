---
title: Swarm-SAR Demo
emoji: 🚁
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
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
# 1. Create a Streamlit Space named 'swarm-sar' on huggingface.co
# 2. Push the spaces/ folder as the Space repo root:
huggingface-cli repo create swarm-sar --type space --space_sdk streamlit
git clone https://huggingface.co/spaces/<your-hf-username>/swarm-sar hf-space
cp spaces/app.py spaces/requirements.txt spaces/README.md hf-space/
cd hf-space && git add . && git commit -m "Swarm-SAR demo" && git push
```

The Space installs the package from GitHub (see `requirements.txt`) and launches
the dashboard. Only the self-contained simulation backend is needed — no GPU,
no training dependencies.
