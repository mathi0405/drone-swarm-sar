"""Publication styling: consistent fonts, palette and per-drone colors."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import ListedColormap

# distinct, colour-blind-friendly per-drone colours (scales to 10)
DRONE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f"]

# CellType -> colour (indices match swarm_sar.environment.entities.CellType)
_CELL = ["#f7f7f7",  # FREE
         "#d9d9d9",  # ROAD
         "#525252",  # BUILDING
         "#3f9b52",  # TREE
         "#b8875a",  # RUBBLE
         "#e6291f",  # FIRE
         "#9c9c9c",  # SMOKE
         "#8153c9",  # NO_FLY
         "#2b8cff"]  # CHARGING
CELL_CMAP = ListedColormap(_CELL)


def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })
