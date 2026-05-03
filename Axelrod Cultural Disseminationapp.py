import numpy as np
import solara
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
)
from mesa.visualization.utils import update_counter

from model import AxelrodModel

# "black <= 20%, dark gray = 40%, gray = 60%, light gray = 80%,
# white = 100%". 
_SIMILARITY_LEVELS = [
    (0.20, "#000000"),  # black
    (0.40, "#404040"),  # dark gray
    (0.60, "#808080"),  # mid gray
    (0.80, "#c0c0c0"),  # light gray
    (1.01, "#ffffff"),  # white
]

def _similarity_to_color(sim):
    """Bin a [0, 1] similarity into one of Axelrod's five gray levels."""
    for upper, color in _SIMILARITY_LEVELS:
        if sim <= upper:
            return color
    return "#ffffff"  # unreachable; defensive


def _segment_between(p1, p2, width, height, torus):
    """Return endpoints of the line segment to draw between two adjacent
    cells centered at p1 and p2 (in grid coordinates).

    The segment is perpendicular to the vector p1 -> p2 and centered on
    the midpoint between them. Length 1 = one full cell side, so:
      - vN neighbors get a full cell-edge segment (half-length 0.5)
      - diagonal (Moore-only) neighbors get a short tick at the corner
      - distance-2 cardinal neighbors (the 12-neighborhood "diamond"
        extras) get an even shorter tick straddling the cell between
        the two endpoints

    The non-vN cases never sit on a real cell wall so we cannot fully
    replicate Axelrod's Figure 1 for them; the short tick is a
    reasonable visual proxy that still shows similarity.

    On a torus we use the shortest wrap-around displacement so segments
    near the edge stay near the edge instead of stretching across the
    whole grid.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if torus:
        if dx >  width  / 2: dx -= width
        if dx < -width  / 2: dx += width
        if dy >  height / 2: dy -= height
        if dy < -height / 2: dy += height
    mx, my = x1 + dx / 2.0, y1 + dy / 2.0
    plen = (dx * dx + dy * dy) ** 0.5
    if plen == 0:
        return None
    # Perpendicular direction, normalized.
    px, py = -dy / plen, dx / plen
    if plen <= 1.01:
        half = 0.5      # vN neighbor — the segment IS the cell wall
    elif plen <= 1.5:
        half = 0.30     # diagonal — short corner tick
    else:
        half = 0.20     # distance-2 cardinal
    return ((mx - px * half, my - py * half),
            (mx + px * half, my + py * half))


@solara.component
def AxelrodSpace(model):
    """Custom Solara component: Axelrod-style boundary grayscale figure.

    On every model step Solara re-renders this component because we
    call `update_counter.get()`, which subscribes us to the global
    "model has stepped" tick. We rebuild the figure from scratch each
    frame; cheaper than maintaining mutable matplotlib state across
    frames, and the cost is negligible on grids up to 30x30.
    """
    update_counter.get()  # subscribe to model step updates

    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot()

    W, H = model.width, model.height

    # Light gray cell-grid backdrop (purely for orientation, not
    # information-bearing). It helps the user see individual cells
    # once internal boundaries dissolve to white near convergence.
    for x in range(W + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, H - 0.5],
                color="#eeeeee", linewidth=0.5, zorder=1)
    for y in range(H + 1):
        ax.plot([-0.5, W - 0.5], [y - 0.5, y - 0.5],
                color="#eeeeee", linewidth=0.5, zorder=1)

    # Collect all neighbor-pair segments, grouped by similarity color.
    # We deduplicate (a, b) and (b, a) by keeping only pairs where
    # nb.unique_id > a.unique_id; without this every edge is drawn
    # twice (and at double the linewidth, which looks wrong on torus).
    segments_by_color = {color: [] for _, color in _SIMILARITY_LEVELS}
    F = model.num_features
    for agent in model._agent_list:
        a_id = agent.unique_id
        for nb in model.get_neighbors(agent.pos):
            if nb.unique_id <= a_id:
                continue
            matches = int(np.sum(agent.culture == nb.culture))
            sim = matches / F
            color = _similarity_to_color(sim)
            seg = _segment_between(agent.pos, nb.pos, W, H, model.grid.torus)
            if seg is not None:
                segments_by_color[color].append(seg)

    # Draw heaviest (black, most-different) on top; white edges (fully
    # converged) are skipped entirely. Linewidths echo Axelrod's Figure
    # 1 styling: thick black region borders, thin pale internal edges.
    LINEWIDTHS = {
        "#000000": 3.0,
        "#404040": 2.2,
        "#808080": 1.6,
        "#c0c0c0": 1.0,
        "#ffffff": 0.0,   # don't draw at all — saves cycles
    }
    ZORDERS = {
        "#000000": 5, "#404040": 4, "#808080": 3,
        "#c0c0c0": 2, "#ffffff": 1,
    }
    for color, segs in segments_by_color.items():
        if not segs or LINEWIDTHS[color] == 0.0:
            continue
        lc = LineCollection(
            segs, colors=color, linewidths=LINEWIDTHS[color],
            zorder=ZORDERS[color],
        )
        ax.add_collection(lc)

    ax.set_xlim(-0.6, W - 0.4)
    ax.set_ylim(-0.6, H - 0.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()       # row 0 at top, matching Axelrod's Table 1
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"step {model.steps}  |  events {model.total_events:,}  |  "
        f"regions {_count_regions_quick(model)}  "
        f"(black = different, white = identical)",
        fontsize=9,
    )

    return solara.FigureMatplotlib(fig)


def _count_regions_quick(model):
    """Lightweight region count for the title bar. Same flood-fill rule
    as model.count_regions but inlined here so we don't add an import
    dependency just for a label.
    """
    visited = set()
    count = 0
    for agent in model._agent_list:
        if agent.unique_id in visited:
            continue
        count += 1
        culture = agent.culture
        stack = [agent]
        while stack:
            cur = stack.pop()
            if cur.unique_id in visited:
                continue
            visited.add(cur.unique_id)
            for nb in model.get_neighbors(cur.pos):
                if (nb.unique_id not in visited and
                        np.array_equal(nb.culture, culture)):
                    stack.append(nb)
    return count

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random seed",
    },
    "width": Slider("Grid width", value=10, min=4, max=30, step=1),
    "height": Slider("Grid height", value=10, min=4, max=30, step=1),
    "num_features": Slider(
        "F (cultural features)", value=5, min=2, max=15, step=1
    ),
    "num_traits": Slider(
        "q (traits per feature)", value=10, min=2, max=20, step=1
    ),
    "neighborhood_size": {
        "type": "Select",
        "value": 4,
        "values": [4, 8, 12],
        "label": "Neighborhood (4=vN, 8=Moore, 12=diamond)",
    },
    "torus": {
        "type": "Checkbox",
        "value": False,
        "label": "Torus (wrap boundaries)",
    },
    "events_per_step": Slider(
        "Events per step (display speed only)",
        value=200, min=10, max=2000, step=10,
    ),
}

# Initial model instance.
model = AxelrodModel()

# Plot components (unchanged from before):
#   1. Region count + zone count on the same axes (paper Figure 3).
#   2. Largest region — monoculture indicator.
RegionsAndZonesPlot = make_plot_component(
    {"Cultural Regions": "tab:blue", "Cultural Zones": "tab:orange"}
)
LargestPlot = make_plot_component({"Largest Region (%)": "tab:green"})

# We pass renderer=None and put AxelrodSpace as the FIRST component,
# so it occupies the slot SpaceRenderer would normally fill. SolaraViz
# accepts plain solara components as entries in `components=[...]`.
page = SolaraViz(
    model,
    renderer=None,
    components=[AxelrodSpace, RegionsAndZonesPlot, LargestPlot],
    model_params=model_params,
    name="Axelrod (1997): Cultural Dissemination",
)
page  # noqa
