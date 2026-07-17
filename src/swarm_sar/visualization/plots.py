"""All publication figures. Each function takes plain data and writes a PNG.

Figures produced from *real* simulation rollouts: world map, 3-D trajectories,
animation, coverage heatmap, dashboard snapshot, communication graph, battery,
rescue/detection/failure timelines, exploration entropy, task allocation.
Figures marked ``[illustrative]`` (reward/learning curves, attention, 4-arch
comparison) use documented placeholder data pending a full GPU training run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from swarm_sar.visualization import synthetic as syn
from swarm_sar.visualization.style import CELL_CMAP, DRONE_COLORS, set_pub_style

set_pub_style()
_CELL_LABELS = ["free", "road", "building", "tree", "rubble", "fire", "smoke", "no-fly", "charging"]


def _save(fig, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    if p.suffix == ".png":
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------- #
# Fig 1 - system architecture                                             #
# ----------------------------------------------------------------------- #
def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(11, 6.5)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    def box(x, y, w, h, text, c):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="#333", lw=1.4, alpha=.95, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, zorder=3)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    box(0.3, 3.2, 2.3, 1.6, "Disaster World\n(procedural gen)", "#cfe8ff")
    box(0.3, 1.0, 2.3, 1.6, "Sensors\ncam/IMU/GPS/LiDAR", "#d7f0d0")
    box(0.3, 5.4, 2.3, 1.6, "AirSim / Unreal\n(optional backend)", "#eee")
    box(3.2, 3.0, 2.6, 2.0, "SARSwarmEnv\n(PettingZoo parallel)\nobs / act / reward", "#ffe6a8")
    box(6.4, 5.2, 2.6, 1.7, "Per-drone Encoder\nTransformer + GNN", "#f3d0e6")
    box(6.4, 3.0, 2.6, 1.7, "Actor (decentralized)\nπ(a|o)", "#f3d0e6")
    box(6.4, 0.8, 2.6, 1.7, "Centralized Critic\nV(s) [CTDE]", "#f7c9c9")
    box(9.5, 3.0, 2.2, 2.0, "MAPPO Trainer\nGAE + clip", "#d9d2ff")
    box(3.2, 0.6, 2.6, 1.6, "Comms / Battery /\nFaults / TaskAlloc", "#d0eef0")
    box(9.5, 5.6, 2.2, 1.4, "Eval + SIS\n& Figures", "#e8e8c0")
    arrow(2.6, 4.0, 3.2, 4.0); arrow(2.6, 1.8, 3.2, 1.4); arrow(2.6, 6.2, 3.2, 4.7)
    arrow(5.8, 4.3, 6.4, 5.6); arrow(7.7, 5.2, 7.7, 4.7); arrow(7.7, 3.0, 7.7, 2.5)
    arrow(9.0, 4.0, 9.5, 4.0); arrow(9.0, 1.6, 9.5, 3.2); arrow(5.8, 1.4, 6.4, 1.5)
    arrow(10.6, 5.0, 10.6, 5.6)
    ax.text(6, 7.6, "Swarm-SAR: CTDE Multi-Agent RL Architecture", ha="center", fontsize=13, weight="bold")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 2 - environment map                                                 #
# ----------------------------------------------------------------------- #
def fig_world_map(log, path, title="Simulation Environment (procedural disaster world)"):
    grid = np.asarray(log.world_grid)
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    ax.imshow(grid, cmap=CELL_CMAP, vmin=0, vmax=8, origin="lower", interpolation="nearest")
    for i, c in enumerate(log.charging):
        ax.scatter(*c, marker="s", s=90, c="#2b8cff", edgecolors="k", label="charging" if i == 0 else None)
    for v in log.victims:
        ax.scatter(v["pos"][0], v["pos"][1], marker="*", s=170,
                   c="#00c853" if v["rescued"] else ("#ffd600" if v["detected"] else "#c62828"),
                   edgecolors="k", zorder=5)
    f0 = log.frames[0]
    for i, p in enumerate(f0["pos"]):
        ax.scatter(p[0], p[1], marker="^", s=90, c=DRONE_COLORS[i % 10], edgecolors="k", zorder=6)
    from matplotlib.patches import Patch
    handles = [Patch(fc=CELL_CMAP(i), ec="k", label=_CELL_LABELS[i]) for i in range(9)]
    handles += [plt.Line2D([], [], marker="*", ls="", mfc="#c62828", mec="k", ms=12, label="victim (lost)"),
                plt.Line2D([], [], marker="*", ls="", mfc="#00c853", mec="k", ms=12, label="victim (rescued)")]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title(title); ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 3 - 3D trajectories                                                 #
# ----------------------------------------------------------------------- #
def fig_trajectory_3d(log, path):
    frames = log.frames; n = len(frames[0]["pos"])
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        xs = [f["pos"][i][0] for f in frames]; ys = [f["pos"][i][1] for f in frames]
        zs = [f["alt"][i] for f in frames]
        ax.plot(xs, ys, zs, color=DRONE_COLORS[i % 10], lw=1.8, alpha=.85, label=f"drone {i}")
        ax.scatter(xs[0], ys[0], zs[0], marker="^", s=80, color=DRONE_COLORS[i % 10], ec="k")
        ax.scatter(xs[-1], ys[-1], zs[-1], marker="s", s=70, color=DRONE_COLORS[i % 10], ec="k")
    for c in log.collisions:
        ax.scatter(c["pos"][0], c["pos"][1], 10, marker="X", s=120, color="red", ec="k", zorder=9)
    for v in log.victims:
        if v["rescued"]:
            ax.scatter(v["pos"][0], v["pos"][1], 2, marker="*", s=200, color="#00c853", ec="k", zorder=9)
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)"); ax.set_zlabel("altitude (m)")
    ax.set_title("3-D Drone Trajectories  (▲ start  ■ end  ✖ collision  ★ rescue)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 4 - animated replay (GIF, optional MP4)                             #
# ----------------------------------------------------------------------- #
def animate_trajectory(log, gif_path, mp4_path=None, fps=12, stride=1):
    frames = log.frames[::stride]; n = len(frames[0]["pos"]); grid = np.asarray(log.world_grid)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    ax.imshow(grid, cmap=CELL_CMAP, vmin=0, vmax=8, origin="lower", interpolation="nearest")
    for v in log.victims:
        ax.scatter(v["pos"][0], v["pos"][1], marker="*", s=120, c="#c62828", ec="k", zorder=4)
    trails = [ax.plot([], [], color=DRONE_COLORS[i % 10], lw=1.6)[0] for i in range(n)]
    dots = [ax.scatter([], [], color=DRONE_COLORS[i % 10], s=60, ec="k", zorder=6) for i in range(n)]
    title = ax.set_title("")
    xs = [[] for _ in range(n)]; ys = [[] for _ in range(n)]

    def update(k):
        f = frames[k]
        for i in range(n):
            xs[i].append(f["pos"][i][0]); ys[i].append(f["pos"][i][1])
            trails[i].set_data(xs[i], ys[i]); dots[i].set_offsets([f["pos"][i][:2]])
        title.set_text(f"t={f['t']}  coverage={f['coverage']*100:.0f}%  "
                       f"found={f['found']}  rescued={f['rescued']}")
        return trails + dots + [title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
    saved = [str(gif_path)]
    if mp4_path:
        try:
            anim.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
            saved.append(str(mp4_path))
        except Exception:
            pass
    plt.close(fig)
    return saved


# ----------------------------------------------------------------------- #
# Fig 5 - coverage heatmap                                                #
# ----------------------------------------------------------------------- #
def fig_coverage_heatmap(log, grid_size, path):
    bins = grid_size
    density = np.zeros((bins, bins))
    for f in log.frames:
        for p in f["pos"]:
            x = min(bins - 1, int(p[0])); y = min(bins - 1, int(p[1]))
            density[y, x] += 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    im0 = axes[0].imshow(np.log1p(density), cmap="magma", origin="lower")
    axes[0].set_title("Search Density (visitation, log)"); fig.colorbar(im0, ax=axes[0], fraction=.046)
    explored = (density > 0).astype(float)
    axes[1].imshow(explored, cmap="Greens", origin="lower", vmin=0, vmax=1)
    axes[1].set_title(f"Explored vs Unexplored  ({explored.mean()*100:.0f}% covered)")
    for ax in axes:
        ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 6 - dashboard snapshot                                              #
# ----------------------------------------------------------------------- #
def fig_dashboard_snapshot(log, grid_size, path):
    frames = log.frames; n = len(frames[0]["pos"]); grid = np.asarray(log.world_grid)
    fig = plt.figure(figsize=(14, 8)); gs = fig.add_gridspec(2, 3)
    # positions
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(grid, cmap=CELL_CMAP, vmin=0, vmax=8, origin="lower")
    for i in range(n):
        xs = [f["pos"][i][0] for f in frames]; ys = [f["pos"][i][1] for f in frames]
        ax.plot(xs, ys, color=DRONE_COLORS[i % 10], lw=1.3)
        ax.scatter(xs[-1], ys[-1], color=DRONE_COLORS[i % 10], s=55, ec="k", zorder=5)
    for v in log.victims:
        ax.scatter(v["pos"][0], v["pos"][1], marker="*", s=130,
                   c="#00c853" if v["rescued"] else "#c62828", ec="k", zorder=6)
    ax.set_title("Swarm positions & paths")
    # battery
    ax = fig.add_subplot(gs[0, 1])
    for i in range(n):
        ax.plot([f["soc"][i] * 100 for f in frames], color=DRONE_COLORS[i % 10], label=f"d{i}")
    ax.axhline(25, ls="--", c="r", lw=1); ax.set_title("Battery (%)"); ax.set_ylim(0, 105); ax.legend(fontsize=7, ncol=n)
    # coverage & victims
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([f["coverage"] * 100 for f in frames], c="#2ca02c", label="coverage %")
    ax.plot([f["found"] for f in frames], c="#ff7f0e", label="found")
    ax.plot([f["rescued"] for f in frames], c="#1f77b4", label="rescued")
    ax.set_title("Mission progress"); ax.legend(fontsize=8)
    # comm activity
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(len(frames)), [len(f["comm_edges"]) for f in frames], color="#17becf")
    ax.set_title("Communication activity (links/step)")
    # status
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    s = log.summary
    txt = (f"STATUS\n\ncoverage: {s['coverage']*100:.0f}%\n"
           f"victims: {s['victims_rescued']}/{s['victims_total']} rescued\n"
           f"collisions: {s['collisions']}\n"
           f"comm delivery: {s['delivery_ratio']*100:.0f}%\n"
           f"energy: {s['energy_wh']:.0f} Wh\n"
           f"success: {s['success']}")
    ax.text(0.05, 0.95, txt, va="top", fontsize=12, family="monospace")
    fig.suptitle("Live Swarm Dashboard (snapshot)", fontsize=14, weight="bold")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 7 - communication graph (NetworkX w/ fallback)                      #
# ----------------------------------------------------------------------- #
def fig_comm_graph(log, path, snapshots=3):
    frames = [f for f in log.frames if f["comm_edges"]] or log.frames
    idxs = np.linspace(0, len(frames) - 1, snapshots).astype(int)
    n = len(frames[0]["pos"])
    fig, axes = plt.subplots(1, snapshots, figsize=(5 * snapshots, 5))
    if snapshots == 1:
        axes = [axes]
    try:
        import networkx as nx
        have_nx = True
    except Exception:
        have_nx = False
    for ax, k in zip(axes, idxs):
        f = frames[k]; pos = {i: f["pos"][i][:2] for i in range(n)}
        if have_nx:
            G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(f["comm_edges"])
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#17becf", width=2, alpha=.7)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[DRONE_COLORS[i % 10] for i in range(n)],
                                   node_size=420, edgecolors="k")
            nx.draw_networkx_labels(G, pos, ax=ax, font_color="w", font_size=9)
        else:
            for (a, b) in f["comm_edges"]:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], c="#17becf", lw=2, alpha=.7, zorder=1)
            for i in range(n):
                ax.scatter(*pos[i], s=420, c=DRONE_COLORS[i % 10], ec="k", zorder=2)
                ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center", color="w", fontsize=9, zorder=3)
        ax.set_title(f"t={f['t']}  ({len(f['comm_edges'])} links)"); ax.set_aspect("equal")
    fig.suptitle("Decentralized Communication Graph (message passing)", fontsize=13, weight="bold")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Figs 8-11 - training / evaluation curves (mixed real + illustrative)    #
# ----------------------------------------------------------------------- #
def fig_reward_curves(path, steps=60):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for arch in syn.ARCHS:
        m, sd = syn.multi_seed(arch, steps)
        x = np.linspace(0, 3, steps)
        r = m * 220 - 40
        ax.plot(x, r, label=arch); ax.fill_between(x, (m - sd) * 220 - 40, (m + sd) * 220 - 40, alpha=.15)
    ax.set_xlabel("environment steps (millions)"); ax.set_ylabel("episode return")
    ax.set_title("Training Reward Curves  [illustrative — mean±std, 5 seeds]"); ax.legend()
    return _save(fig, path)


def fig_metric_vs_episodes(real_series: dict[str, list], path, ylabel, title, illustrative=True):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, series in real_series.items():
        ax.plot(series, marker="o", ms=3, label=name)
    ax.set_xlabel("episode"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    return _save(fig, path)


def fig_success_rate(success_by_arch: dict[str, float], path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    archs = list(success_by_arch.keys()); vals = [success_by_arch[a] * 100 for a in archs]
    ax.bar(archs, vals, color=DRONE_COLORS[: len(archs)], edgecolor="k")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center")
    ax.set_ylabel("mission success rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("Mission Success Rate by Architecture"); plt.setp(ax.get_xticklabels(), rotation=15)
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 12 - battery consumption                                            #
# ----------------------------------------------------------------------- #
def fig_battery(log, path):
    frames = log.frames; n = len(frames[0]["pos"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n):
        ax.plot([f["soc"][i] * 100 for f in frames], color=DRONE_COLORS[i % 10], label=f"drone {i}")
    ax.axhline(25, ls="--", c="r", label="emergency-return threshold")
    ax.set_xlabel("timestep"); ax.set_ylabel("state of charge (%)"); ax.set_ylim(0, 105)
    ax.set_title("Battery Consumption & Recharge"); ax.legend(fontsize=8, ncol=2)
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 13 & 19 - rescue / detection timelines                              #
# ----------------------------------------------------------------------- #
def fig_rescue_timeline(log, path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    frames = log.frames
    ax.plot([f["t"] for f in frames], [f["found"] for f in frames], c="#ff7f0e", lw=2, label="detected")
    ax.plot([f["t"] for f in frames], [f["rescued"] for f in frames], c="#1f77b4", lw=2, label="rescued")
    for v in log.victims:
        if v["detected_step"] >= 0:
            ax.axvline(v["detected_step"], color="#ff7f0e", alpha=.2)
        if v["rescued_step"] >= 0:
            ax.scatter(v["rescued_step"], sum(1 for u in log.victims if 0 <= u["rescued_step"] <= v["rescued_step"]),
                       marker="*", s=120, c="#00c853", ec="k", zorder=5)
    ax.set_xlabel("timestep"); ax.set_ylabel("cumulative victims")
    ax.set_title("Rescue Timeline"); ax.legend()
    return _save(fig, path)


def fig_detection_timeline(log, path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vic = sorted(log.victims, key=lambda v: (v["detected_step"] if v["detected_step"] >= 0 else 1e9))
    for row, v in enumerate(vic):
        ds = v["detected_step"]; rs = v["rescued_step"]; last = log.frames[-1]["t"]
        ax.plot([0, last], [row, row], color="#eee", lw=6, solid_capstyle="round")
        if ds >= 0:
            end = rs if rs >= 0 else last
            ax.plot([ds, end], [row, row], color="#ffb300", lw=6, solid_capstyle="round")
        if rs >= 0:
            ax.scatter(rs, row, marker="*", s=140, color="#00c853", ec="k", zorder=5)
        ax.text(-2, row, f"V{v['idx']}", ha="right", va="center", fontsize=8)
    ax.set_xlabel("timestep"); ax.set_yticks([]); ax.set_title("Victim Detection → Rescue Timeline")
    ax.legend(handles=[plt.Line2D([], [], color="#ffb300", lw=6, label="detected, awaiting rescue"),
                       plt.Line2D([], [], marker="*", ls="", mfc="#00c853", mec="k", ms=12, label="rescued")],
              loc="lower right", fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 14 - failure & recovery timeline                                    #
# ----------------------------------------------------------------------- #
def fig_failure_recovery(log, path):
    frames = log.frames; n = len(frames[0]["pos"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    comps = ["gps", "cam", "comm", "motor"]
    for i in range(n):
        for j, comp in enumerate(comps):
            row = i * len(comps) + j
            status = [1 if f["faults"][i][comp] else 0 for f in frames]
            t = [f["t"] for f in frames]
            ax.fill_between(t, row, row + 0.8, where=np.array(status) > 0, color="#c8e6c9", step="mid")
            ax.fill_between(t, row, row + 0.8, where=np.array(status) == 0, color="#ef9a9a", step="mid")
            ax.text(-2, row + 0.4, f"d{i}.{comp}", ha="right", va="center", fontsize=7)
    ax.set_xlabel("timestep"); ax.set_yticks([])
    ax.set_title("Fault & Recovery Timeline (green=OK, red=failed)")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 15 - attention heatmap  (real if torch, else illustrative)          #
# ----------------------------------------------------------------------- #
def fig_attention_heatmap(path, n_tokens=8, seed=0):
    rng = np.random.default_rng(seed)
    tokens = ["self", "peer1", "peer2", "peer3", "occ-map", "victim-map", "msg", "cls"][:n_tokens]
    A = rng.dirichlet(np.ones(n_tokens) * 0.6, size=n_tokens)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(A, cmap="viridis")
    ax.set_xticks(range(n_tokens)); ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(n_tokens)); ax.set_yticklabels(tokens)
    fig.colorbar(im, ax=ax, fraction=.046, label="attention weight")
    ax.set_title("Transformer Attention  [illustrative]")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 16 - GNN message passing                                            #
# ----------------------------------------------------------------------- #
def fig_gnn_message_passing(log, path):
    frames = [f for f in log.frames if f["comm_edges"]] or log.frames[:1]
    f = frames[len(frames) // 2]; n = len(f["pos"])
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = {i: f["pos"][i][:2] for i in range(n)}
    for (a, b) in f["comm_edges"]:
        ax.annotate("", pos[b][:2], pos[a][:2],
                    arrowprops=dict(arrowstyle="-|>", color="#17becf", lw=2.5, alpha=.8))
    deg = {i: sum(1 for e in f["comm_edges"] if i in e) for i in range(n)}
    for i in range(n):
        ax.scatter(*pos[i], s=300 + 120 * deg[i], c=DRONE_COLORS[i % 10], ec="k", zorder=5)
        ax.text(pos[i][0], pos[i][1], f"h{i}", color="w", ha="center", va="center", zorder=6)
    ax.set_title("GNN Message Passing over Comm Graph\n(node size ∝ messages aggregated)")
    ax.set_aspect("equal")
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 17 - exploration entropy                                            #
# ----------------------------------------------------------------------- #
def fig_exploration_entropy(entropy_series: dict[str, list], path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, series in entropy_series.items():
        ax.plot(series, label=name)
    ax.set_xlabel("timestep"); ax.set_ylabel("normalized spatial entropy")
    ax.set_title("Exploration Entropy (higher = better spread)"); ax.legend()
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 18 - task allocation                                                #
# ----------------------------------------------------------------------- #
def fig_task_allocation(drone_pos, victim_pos, assignment: dict[int, int], path, strategy="auction"):
    fig, ax = plt.subplots(figsize=(7, 6))
    dp = np.asarray(drone_pos); vp = np.asarray(victim_pos)
    for i, p in enumerate(dp):
        ax.scatter(*p, marker="^", s=160, c=DRONE_COLORS[i % 10], ec="k", zorder=5, label=f"drone {i}")
    ax.scatter(vp[:, 0], vp[:, 1], marker="*", s=180, c="#c62828", ec="k", zorder=5)
    for i, j in assignment.items():
        if j is not None and 0 <= j < len(vp):
            ax.annotate("", vp[j], dp[i], arrowprops=dict(arrowstyle="-|>", color=DRONE_COLORS[i % 10], lw=2))
    ax.set_title(f"Task Allocation ({strategy})"); ax.legend(fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Fig 20 - algorithm comparison                                           #
# ----------------------------------------------------------------------- #
def fig_algo_comparison(agg_by_arch: dict[str, dict], path):
    labels = ["Coverage", "Victims Rescued", "SIS/100", "Safety (1-coll)"]
    archs = list(agg_by_arch.keys())
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(labels)); w = 0.8 / len(archs)
    for a_i, arch in enumerate(archs):
        agg = agg_by_arch[arch]
        vals = [agg["coverage"]["mean"],
                agg["victims_rescued"]["mean"] / 8.0,
                agg["swarm_intelligence_score"]["mean"] / 100.0,
                1 - min(1, agg["collision_rate"]["mean"] * 5)]
        errs = [agg["coverage"]["std"], agg["victims_rescued"]["std"] / 8.0,
                agg["swarm_intelligence_score"]["std"] / 100.0, agg["collision_rate"]["std"] * 5]
        ax.bar(x + a_i * w, vals, w, yerr=errs, capsize=3, label=arch,
               color=DRONE_COLORS[a_i % 10], edgecolor="k")
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(labels)
    ax.set_ylabel("normalized score"); ax.set_ylim(0, 1.05)
    ax.set_title("Algorithm Comparison (mean±std across seeds)")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------- #
# Real training curves from logged scalars.csv (post-training)            #
# ----------------------------------------------------------------------- #
def fig_reward_curves_from_csv(csv_by_arch: dict, path, ycol="ep_return",
                               xcol="step", title="Training Reward Curves (measured)"):
    """Plot measured learning curves. `csv_by_arch` maps arch-name -> list of
    scalars.csv paths (one per seed). Aggregates mean +/- std across seeds."""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for arch, paths in csv_by_arch.items():
        series = []
        steps = None
        for p in paths:
            df = pd.read_csv(p)
            if ycol not in df.columns or xcol not in df.columns:
                continue
            df[ycol] = df[ycol].interpolate(method="linear").bfill().ffill()
            series.append(df[ycol].to_numpy()); steps = df[xcol].to_numpy()
        if not series:
            continue
        L = min(len(s) for s in series)
        M = np.stack([s[:L] for s in series]); x = steps[:L]
        m, sd = M.mean(0), M.std(0)
        ax.plot(x, m, label=arch); ax.fill_between(x, m - sd, m + sd, alpha=.15)
    ax.set_xlabel("environment steps"); ax.set_ylabel(ycol)
    ax.set_title(title); ax.legend()
    return _save(fig, path)


def fig_method_comparison(results: dict, path, title="Benchmark: SIS by method (IQM ± 95% CI)"):
    """Horizontal bar chart of IQM with bootstrap-CI error bars, ranked."""
    names = list(results.keys())
    iqms = np.array([results[n]["summary"]["iqm"] for n in names])
    lo = np.array([results[n]["summary"]["ci95_low"] for n in names])
    hi = np.array([results[n]["summary"]["ci95_high"] for n in names])
    order = np.argsort(iqms)
    names = [names[i] for i in order]; iqms = iqms[order]; lo = lo[order]; hi = hi[order]
    err = np.vstack([iqms - lo, hi - iqms])
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(names) + 1.5))
    colors = [DRONE_COLORS[i % 10] for i in range(len(names))]
    ax.barh(names, iqms, xerr=err, capsize=4, color=colors, edgecolor="k")
    for i, v in enumerate(iqms):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set_xlabel("Swarm Intelligence Score (IQM, 20 frozen maps)")
    ax.set_title(title); ax.set_xlim(0, max(hi) * 1.15)
    return _save(fig, path)


def fig_sis_sensitivity(rhos, top1_retention, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(rhos, bins=30, color="#4f8cff", edgecolor="k", alpha=.85)
    ax.axvline(np.mean(rhos), color="#d62728", lw=2, label=f"mean ρ={np.mean(rhos):.3f}")
    ax.set_xlabel("Spearman ρ (perturbed vs default weights)"); ax.set_ylabel("count")
    ax.set_title(f"SIS ranking stability under weight perturbation\n(top-1 retained {top1_retention*100:.0f}% of the time)")
    ax.legend()
    return _save(fig, path)


def fig_sis_pareto(components_by_method, front, path):
    """Parallel-coordinates over the 5 objectives; Pareto-optimal methods bold."""
    dims = ["coverage", "rescue", "energy", "communication", "safety"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(dims))
    for i, (name, c) in enumerate(components_by_method.items()):
        y = [c[d] for d in dims]
        onfront = name in front
        ax.plot(x, y, marker="o", lw=3 if onfront else 1.2,
                alpha=1.0 if onfront else 0.5, color=DRONE_COLORS[i % 10],
                label=f"{name}{' ★' if onfront else ''}")
    ax.set_xticks(x); ax.set_xticklabels(dims); ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalized objective"); ax.set_title("SIS objectives — Pareto-optimal methods in bold (★)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save(fig, path)


def fig_comm_degradation(packet_loss, sis_mean, sis_std, rescue_mean, path):
    """Graceful-degradation curve: SIS & rescues vs packet loss (a curve, not 3 points)."""
    packet_loss = np.asarray(packet_loss); sis_mean = np.asarray(sis_mean)
    sis_std = np.asarray(sis_std); rescue_mean = np.asarray(rescue_mean)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(packet_loss * 100, sis_mean, "-o", color="#4f8cff", label="SIS")
    ax1.fill_between(packet_loss * 100, sis_mean - sis_std, sis_mean + sis_std,
                     color="#4f8cff", alpha=.15)
    ax1.set_xlabel("packet loss (%)"); ax1.set_ylabel("Swarm Intelligence Score", color="#4f8cff")
    ax2 = ax1.twinx()
    ax2.plot(packet_loss * 100, rescue_mean, "--s", color="#2ca02c", label="victims rescued")
    ax2.set_ylabel("victims rescued", color="#2ca02c")
    ax1.set_title("Graceful degradation vs communication packet loss")
    return _save(fig, path)
