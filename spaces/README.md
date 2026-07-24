---
title: Swarm-SAR Demo
emoji: 🚁
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Swarm-SAR — live demo

Interactive, zero-dependency replay of a decentralized Multi-Agent RL swarm
searching a procedurally generated disaster site. Play, pause and scrub through
an episode; watch coverage grow, comm links form, and victims get rescued —
all client-side in the browser, no server.

Source: https://github.com/mathi0405/drone-swarm-sar

## Deploy this Space (from the repo root)

```bash
pip install huggingface_hub
# PowerShell:  $env:HF_TOKEN="hf_your_write_token"
# bash:        export HF_TOKEN=hf_your_write_token
python scripts/deploy_hf_space.py
```

This deploys a **Static** Space (free on Hugging Face — Docker/Gradio Spaces now
require PRO) serving the self-contained `index.html` replay viewer and its
`replay_data.json`. Regenerate the episode data with
`python scripts/export_replay.py --out spaces/replay_data.json`.

The full interactive Streamlit dashboard (`app.py` + `Dockerfile`) is also here
for anyone running it locally (`streamlit run src/swarm_sar/dashboard/app.py`)
or on a PRO Docker Space.
