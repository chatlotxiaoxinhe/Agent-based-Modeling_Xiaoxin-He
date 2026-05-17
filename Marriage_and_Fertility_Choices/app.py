"""
Launch with:
    solara run app.py

* Left panel: parameter sliders (γ, grid scale, random seed).
* Center: colored grid of agents (blue / pink / red / orange).
* Right top: live state-share line chart.
* Right bottom: Moran's I for ORANGE and RED — the spatial autocorrelation
  measure (positive → clustering, negative → anti-clustering, near 0 →
  random spatial distribution).

GUI scale note
--------------
The default ``scale_factor = 0.5`` gives a 50×20 grid with ~900 agents so
the dashboard renders responsively in a browser. This preserves the column
widths (10/25/15) of the full configuration, so the proportion of border vs
core agents — the structural driver of H4 — is unchanged. 

"""

from __future__ import annotations

from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

from agents import BLUE, PINK, RED, ORANGE
from model import MarriageFertilityModel

# Color scheme for the four states
# Chosen to match the substantive names. Hex codes are slightly
# desaturated so groups remain visually distinct on a white background.
STATE_COLOR = {
    BLUE: "#3498db",     # blue:   single, career-focused
    PINK: "#e91e63",     # pink:   married, no children
    RED: "#c0392b",      # red:    working mother
    ORANGE: "#f39c12",   # orange: stay-at-home mother
}

# Agent portrayal callback
def agent_portrayal(agent):
    """
    Tell Mesa's space component how to draw each agent.
    ...
    """
    return AgentPortrayalStyle(
        color=STATE_COLOR[agent.state],
        marker="s",   # square — fills the cell more visibly than "o"
        size=20,
    )


# Plot components (use DataCollector model reporters)
# State-share line chart: four traces, one per state.
StatePlot = make_plot_component(
    {"Blue": STATE_COLOR[BLUE],
     "Pink": STATE_COLOR[PINK],
     "Red": STATE_COLOR[RED],
     "Orange": STATE_COLOR[ORANGE]},
)

# Moran's I time series for the two motherhood states (the H3/H5 outcome).
MoranPlot = make_plot_component(
    {"MoransI_Orange": STATE_COLOR[ORANGE],
     "MoransI_Red": STATE_COLOR[RED]},
)

# Space component (the colored grid)
SpaceComponent = make_space_component(agent_portrayal)

# Model parameters — exposed as sliders / inputs in the dashboard
# The Solara dashboard uses this dict to render input controls. Each entry
# is either:
#   • a primitive (fixed parameter, not shown), OR
#   • a dict describing a slider / select with "type", "value", "min", "max",
#     "step", "label".
model_params = {
    # --- the headline parameter ---
    "gamma": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "γ (social comparison strength)",
        "min": 0.0,
        "max": 0.6,
        "step": 0.05,
    },
    # --- grid scale (height only; column widths fixed) ---
    "scale_factor": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Grid scale (height fraction)",
        "min": 0.2,
        "max": 1.0,
        "step": 0.1,
    },
    # --- random seed ---
    # Note: using SliderInt (not InputText) because the model expects an int
    # and Solara's InputText returns strings, which would require manual
    # casting on every model rebuild.
    "seed": {
        "type": "SliderInt",
        "value": 42,
        "label": "Random seed",
        "min": 0,
        "max": 1000,
        "step": 1,
    },
    # NOTE: economic parameters (transfer ranges, dissolution probabilities,
    # spike probabilities, etc.) are defined in agents.py as module-level
    # constants. To expose any of them here, add a corresponding constructor
    # argument to MarriageFertilityModel.__init__ and a slider entry above.
}

# Build a starting model and the SolaraViz page
# Mesa expects a concrete model instance. Sliders rebuild the model on reset.
_initial_model = MarriageFertilityModel(
    gamma=0.3,
    scale_factor=0.5,
    seed=42,
)

page = SolaraViz(
    _initial_model,
    components=[SpaceComponent, StatePlot, MoranPlot],
    model_params=model_params,
    name="Marriage & Fertility ABM (V1 / V2)",
)

# `page` is the Solara component that `solara run app.py` will display.
