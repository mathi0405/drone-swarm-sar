#!/usr/bin/env python3
"""Professional Streamlit dashboard for Swarm-SAR.

Tabs:
  • Live Simulation - run a decentralized-swarm episode and replay it frame by
    frame (positions, battery, coverage, communication graph, mission status).
  • Events & Explainability - tracking tables, fault timeline, comms, traces.
  • Training Metrics - scalar curves from a selected run's scalars.csv.
  • Evaluation - multi-seed metric table incl. the Swarm Intelligence Score.
  • Figures - gallery of generated publication figures.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.drone.drone import ACTIONS
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.metrics import aggregate, episode_metrics
from swarm_sar.training.rollout import EpisodeLog, HeuristicActor, load_neural_actor
from swarm_sar.visualization.style import CELL_CMAP, DRONE_COLORS

ROOT = Path(__file__).resolve().parents[3]
st.set_page_config(page_title="Swarm-SAR Dashboard", layout="wide", page_icon="🚁")


def _run_async(seed, drones, steps, size, strategy, comm_range, packet_loss, checkpoint=None):
    """Generator that yields the live simulation state frame by frame."""
    cfg = EnvConfig(num_drones=drones, max_steps=steps,
                    world=WorldConfig(size=size, n_victims=8))
    cfg.comm.range_m = comm_range; cfg.comm.packet_loss = packet_loss
    env = SARSwarmEnv(cfg, seed=seed)
    if checkpoint and checkpoint != "None":
        actor = load_neural_actor(checkpoint, deterministic=True)
    else:
        actor = HeuristicActor(env, strategy, seed)

    obs, _ = env.reset(seed=seed)
    if hasattr(actor, "bind_env"):
        actor.bind_env(env)

    log = EpisodeLog(world_grid=env.world.grid.copy())
    total = 0.0

    while env.agents:
        actions = actor(obs)
        obs, rewards, term, trunc, _ = env.step(actions)
        total += float(sum(rewards.values()))

        # update intermediate state
        log.frames = env.history
        log.victims = [{"idx": v.idx, "pos": v.pos.tolist(), "detected": v.detected,
                        "rescued": v.rescued, "detected_step": v.detected_step,
                        "rescued_step": v.rescued_step, "severity": v.severity}
                       for v in env.world.victims]
        yield log, None

    log.collisions = env.collisions
    log.summary = env.episode_summary()
    log.total_reward = total
    log.charging = [c.pos.tolist() for c in env.world.charging]

    yield log, episode_metrics(log, grid_size=size)

@st.cache_data(show_spinner=False)
def _run_cached(*args, **kwargs):
    # Consume the generator to return final state for caching
    gen = _run_async(*args, **kwargs)
    log, mets = None, None
    for frame_log, frame_mets in gen:
        log, mets = frame_log, frame_mets
    return log, mets

def _draw_frame(log, k, size):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(np.asarray(log.world_grid), cmap=CELL_CMAP, vmin=0, vmax=8, origin="lower")
    n = len(log.frames[0]["pos"])
    for i in range(n):
        xs = [f["pos"][i][0] for f in log.frames[: k + 1]]
        ys = [f["pos"][i][1] for f in log.frames[: k + 1]]
        ax.plot(xs, ys, color=DRONE_COLORS[i % 10], lw=1.4)
        if xs:
            ax.scatter(xs[-1], ys[-1], color=DRONE_COLORS[i % 10], s=70, ec="k", zorder=6)
    f = log.frames[k]
    for (a, b) in f["comm_edges"]:
        pa, pb = f["pos"][a], f["pos"][b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#17becf", lw=1.5, alpha=.6, zorder=3)
    for v in log.victims:
        rescued = v["rescued"] and v["rescued_step"] <= f["t"]
        det = v["detected"] and 0 <= v["detected_step"] <= f["t"]
        ax.scatter(v["pos"][0], v["pos"][1], marker="*", s=150,
                   c="#00c853" if rescued else ("#ffd600" if det else "#c62828"),
                   ec="k", zorder=7)
    ax.set_title(f"t={f['t']} | coverage {f['coverage']*100:.0f}% | rescued {f['rescued']}")
    ax.set_xticks([]); ax.set_yticks([])
    return fig

def main():
    st.title("🚁 Swarm-SAR — Decentralized Multi-Agent RL for Search & Rescue")
    st.caption("CTDE · MAPPO · Transformer + GNN · procedural disaster worlds")

    with st.sidebar:
        st.header("Simulation settings")
        drones = st.slider("Drones", 3, 10, 3)
        steps = st.slider("Max steps", 60, 300, 200, 20)
        size = st.select_slider("World size", [48, 56, 64, 80, 96], 64)
        strategy = st.selectbox("Task allocation", ["cbaa", "auction", "hungarian", "nearest", "random"])
        seed = st.number_input("Seed", 0, 9999, 7)
        comm_range = st.slider("Comm range (m)", 0.0, 60.0, 30.0, 5.0)
        packet_loss = st.slider("Packet loss", 0.0, 1.0, 0.08, 0.02)

        checkpoints = ["None"] + sorted(
            {str(p) for pat in ("**/final.pt", "**/best.pt")
             for p in (ROOT / "results").glob(pat)})
        checkpoint = st.selectbox("Neural Policy", checkpoints, format_func=lambda x: "None" if x == "None" else Path(x).parent.parent.name)

        run = st.button("▶ Run episode", type="primary")

        st.divider()
        st.header("Hardware")
        if st.button("Check GPU Status"):
            try:
                res = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_gpu.py")], capture_output=True, text=True)
                st.code(res.stdout or res.stderr)
            except Exception as e:
                st.error(str(e))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Simulation", "Events & Explainability", "Training Metrics", "Evaluation", "Figures"])

    with tab1:
        if run:
            content_container = st.empty()
            with st.spinner("Simulating swarm async..."):
                gen = _run_async(int(seed), drones, steps, size, strategy, comm_range, packet_loss, checkpoint)
                for i, (tmp_log, tmp_mets) in enumerate(gen):
                    # update UI every 5 frames or at the end
                    if i % 5 == 0 or tmp_mets is not None:
                        with content_container.container():
                            left, right = st.columns([3, 2])
                            with left:
                                fig = _draw_frame(tmp_log, len(tmp_log.frames) - 1, size)
                                st.pyplot(fig)
                                plt.close(fig)
                            with right:
                                fig1, ax1 = plt.subplots(figsize=(5, 2.6))
                                for d in range(len(tmp_log.frames[0]["pos"])):
                                    ax1.plot([f["soc"][d] * 100 for f in tmp_log.frames], color=DRONE_COLORS[d % 10])
                                ax1.axhline(25, ls="--", c="r"); ax1.set_title("Battery (%)"); ax1.set_ylim(0, 105)
                                st.pyplot(fig1)
                                plt.close(fig1)

                                fig2, ax2 = plt.subplots(figsize=(5, 2.6))
                                ax2.plot([f["coverage"] * 100 for f in tmp_log.frames], c="#2ca02c", label="coverage")
                                ax2.plot([f["rescued"] for f in tmp_log.frames], c="#1f77b4", label="rescued")
                                ax2.legend(); ax2.set_title("Progress")
                                st.pyplot(fig2)
                                plt.close(fig2)

                st.session_state["log"] = tmp_log
                st.session_state["mets"] = tmp_mets
                st.session_state["size"] = size
                content_container.empty()

        elif "log" not in st.session_state:
            with st.spinner("Loading initial..."):
                log, mets = _run_cached(int(seed), drones, steps, size, strategy, comm_range, packet_loss, checkpoint)
                st.session_state["log"] = log; st.session_state["mets"] = mets; st.session_state["size"] = size

        if "log" in st.session_state:
            log = st.session_state["log"]; mets = st.session_state["mets"]; size = st.session_state["size"]
            c = st.columns(6)
            c[0].metric("Coverage", f"{mets['coverage']*100:.0f}%")
            c[1].metric("Rescued", int(mets['victims_rescued']))
            c[2].metric("Collisions/step", f"{mets['collision_rate']:.3f}")
            c[3].metric("Comm eff.", f"{mets['communication_efficiency']*100:.0f}%")
            c[4].metric("Energy", f"{mets['energy_wh']:.0f} Wh")
            c[5].metric("SIS", f"{mets['swarm_intelligence_score']:.1f}")
            left, right = st.columns([3, 2])
            with left:
                k = st.slider("Playback frame", 0, len(log.frames) - 1, len(log.frames) - 1)
                st.pyplot(_draw_frame(log, k, size))
            with right:
                fig, ax = plt.subplots(figsize=(5, 2.6))
                for i in range(len(log.frames[0]["pos"])):
                    ax.plot([f["soc"][i] * 100 for f in log.frames], color=DRONE_COLORS[i % 10])
                ax.axhline(25, ls="--", c="r"); ax.set_title("Battery (%)"); ax.set_ylim(0, 105)
                st.pyplot(fig)
                fig2, ax2 = plt.subplots(figsize=(5, 2.6))
                ax2.plot([f["coverage"] * 100 for f in log.frames], c="#2ca02c", label="coverage")
                ax2.plot([f["rescued"] for f in log.frames], c="#1f77b4", label="rescued")
                ax2.legend(); ax2.set_title("Progress"); st.pyplot(fig2)

                # Action histogram
                st.subheader("Action Histogram")
                acts = [ACTIONS[a] for f in log.frames for a in f["actions"]]
                act_counts = pd.Series(acts).value_counts()
                st.bar_chart(act_counts)

    with tab2:
        if "log" in st.session_state:
            log = st.session_state["log"]
            st.subheader("Victim Tracking")
            v_df = pd.DataFrame(log.victims)
            st.dataframe(v_df, use_container_width=True)

            st.subheader("Fault Timeline")
            faults = []
            for f in log.frames:
                for d in f["faults"]:
                    if not all([d["gps"], d["cam"], d["comm"], d["motor"]]):
                        faults.append({"step": f["t"], **d})
            if faults:
                st.dataframe(pd.DataFrame(faults), use_container_width=True)
            else:
                st.info("No faults injected/detected.")

            st.subheader("Communication Events")
            comms = [{"step": f["t"], "edges": len(f["comm_edges"])} for f in log.frames if f["comm_edges"]]
            st.dataframe(pd.DataFrame(comms), use_container_width=True)

            st.subheader("Explainability Traces (last frame)")
            st.info("Traces for the last timestep's decision process.")
            last_f = log.frames[-1]
            st.json(last_f)

    with tab3:
        runs = sorted((ROOT / "results").glob("**/scalars.csv"))
        if not runs:
            st.info("No training runs found. Run `python scripts/train_and_report.py` to produce scalars.csv.")
        else:
            sel = st.selectbox("Run", runs, format_func=lambda p: p.parent.name)
            df = pd.read_csv(sel)
            cols = [c for c in df.columns if c not in ("step", "wall_s")]
            for col in cols:
                st.line_chart(df, x="step", y=col)

    with tab4:
        st.subheader("Multi-seed evaluation")
        seeds = st.multiselect("Seeds", list(range(8)), [0, 1, 2, 3])
        if st.button("Evaluate"):
            rows = []
            for s in seeds:
                _, m = _run_cached(int(s), drones, steps, size, strategy,
                                   comm_range, packet_loss, checkpoint)
                rows.append(m)
            agg = aggregate(rows)
            table = pd.DataFrame({k: [f"{v['mean']:.3f} ± {v['std']:.3f}"] for k, v in agg.items()}).T
            table.columns = ["mean ± std"]
            st.dataframe(table, use_container_width=True)

    with tab5:
        figs = sorted((ROOT / "results" / "figures").glob("*.png"))
        if not figs:
            st.info("No figures yet. Run `python scripts/generate_figures.py`.")
        cols = st.columns(3)
        for i, f in enumerate(figs):
            cols[i % 3].image(str(f), caption=f.stem, use_container_width=True)

if __name__ == "__main__":
    main()
