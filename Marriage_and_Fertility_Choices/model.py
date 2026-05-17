"""
Implements the model-level container, grid layout, scheduling, end-of-tick
reference-income update, and aggregate data collection. Also exposes a Moran's I helper to
quantify spatial autocorrelation of the ORANGE state.

The model is written to support both V1 and V2 from the same code path:
  • V1 corresponds exactly to gamma = 0.
  • V2 corresponds to gamma > 0 (asymmetric status-anxiety penalty active).

Grid scaling ("split-environment" strategy)
-------------------------------------------
The grid's horizontal layout (50 columns split into three regions of width
10/25/15) is FIXED at full size. Only the grid HEIGHT can be scaled down
via ``scale_factor`` to give the GUI a smaller, responsive demo environment.
Holding column boundaries fixed preserves the proportion of border vs core
agents — which is what H4 in §5.3 actually depends on — so the demo's
mechanism is faithful to production runs.
"""

from __future__ import annotations

from typing import List

import mesa
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid

# Import the agent class and its constants. agents.py defines the type-keyed
# parameters and life-cycle bounds; model.py owns the grid geometry.
from agents import (
    Woman,
    BLUE, PINK, RED, ORANGE,
    TYPE_SHARES, START_AGE, END_AGE,
)

# Grid geometry
GRID_WIDTH = 50                  # ALWAYS 50; region boundaries depend on this.
GRID_HEIGHT_FULL = 40            # Production height. GUI may use less.
GRID_OCCUPANCY = 0.90            # 1800 of 2000 cells occupied at full size.

# Region boundaries (half-open column intervals). Fixed regardless of scale.
HIGH_REGION_COLS = (0, 10)       # cols 0–9   → high  (400 cells at full size)
MEDIAN_REGION_COLS = (10, 35)    # cols 10–34 → median (1000 cells)
LOW_REGION_COLS = (35, 50)       # cols 35–49 → low   (600 cells)

# Convenience: ordered list of (col_range, type_name).
REGIONS = [
    (HIGH_REGION_COLS, "high"),
    (MEDIAN_REGION_COLS, "median"),
    (LOW_REGION_COLS, "low"),
]

# Aggregate reporters (used by DataCollector)
def _count_state(model: "MarriageFertilityModel", state: str) -> int:
    """
    Number of agents currently in the named state.
    """
    return sum(1 for a in model.agents if a.state == state)


def _mean_income(model: "MarriageFertilityModel") -> float:
    """
    Mean realized income across all agents this tick.
    """
    n = len(model.agents)
    if n == 0:
        return 0.0
    return sum(a.current_income() for a in model.agents) / n


def compute_moran_i(model: "MarriageFertilityModel", target_state: str) -> float:
    """
    Moran's I for the indicator x_i = 1{agent i is in target_state}.

    Formula (binary outcome, Moore-neighborhood weights w_ij ∈ {0, 1}):

        I = (N / W) · Σ_i Σ_j w_ij · (x_i - x̄)(x_j - x̄) / Σ_i (x_i - x̄)²

    Returns 0.0 in degenerate cases (no agents, no neighbor pairs, zero
    variance — e.g. if no agent is in the target state).
    """
    agents = list(model.agents)
    N = len(agents)
    if N < 2:
        return 0.0

    # x_i indicator vector.
    x = {a: (1.0 if a.state == target_state else 0.0) for a in agents}
    mean = sum(x.values()) / N

    # Build deviation cache once.
    dev = {a: x[a] - mean for a in agents}

    # Variance denominator.
    denom = sum(d * d for d in dev.values())
    if denom == 0.0:
        # All agents share the same value (e.g. no oranges yet). Moran's I is
        # mathematically undefined; we report 0.0 as the most neutral default.
        return 0.0

    # Numerator: sum over ordered Moore-neighbor pairs.
    # cell.neighborhood gives the surrounding cells (radius 1, Moore).
    # cell.agents gives any agents in a cell. We exclude self-pairs.
    numerator = 0.0
    W = 0
    for a in agents:
        for nbr_cell in a.cell.neighborhood:
            for b in nbr_cell.agents:
                if b is a:
                    continue
                numerator += dev[a] * dev[b]
                W += 1

    if W == 0:
        return 0.0

    return (N / W) * (numerator / denom)


def moran_i_orange(model: "MarriageFertilityModel") -> float:
    return compute_moran_i(model, ORANGE)


def moran_i_red(model: "MarriageFertilityModel") -> float:
    return compute_moran_i(model, RED)

# Model
class MarriageFertilityModel(mesa.Model):
    """Marriage & Fertility ABM with optional local social comparison.

    Parameters
    ----------
    gamma : float
        Strength of the asymmetric status-anxiety penalty.
        gamma == 0 reproduces V1 exactly (no penalty for any agent).
        gamma > 0 activates V2 (penalty fires when projected own income falls
        below the agent's neighborhood-mean reference income).
    scale_factor : float
        Fraction of the full grid height to use. The width and column
        boundaries are always fixed (50 cols, split 10/25/15).
        scale_factor = 1.0 → 50×40 grid, ~1800 agents.
        scale_factor = 0.5 → 50×20 grid, ~900 agents (GUI-friendly).
    seed : int or None
        Master random seed. Controls Type assignment, grid placement, offer
        arrivals, transfer draws — everything stochastic in the run.
    """

    def __init__(
        self,
        gamma: float = 0.0,
        scale_factor: float = 1.0,
        seed: int | None = None,
    ):
        # Mesa 3.5+ deprecated the `seed=` kwarg in favor of `rng=`.
        # We accept `seed` from the GUI/external callers and translate it.
        # Passing an int to rng= is fine; Mesa builds a Generator from it.
        super().__init__(rng=seed)

        # Store parameters so agents (and the DataCollector) can read them.
        self.gamma = float(gamma)
        self.scale_factor = float(scale_factor)

        # ----- grid -----
        # Height scales; width and region boundaries are fixed.
        self.grid_width = GRID_WIDTH
        self.grid_height = max(2, int(round(GRID_HEIGHT_FULL * scale_factor)))
        # Non-toroidal: edge/corner agents intentionally have fewer Moore
        # neighbors (§4.7).
        self.grid = OrthogonalMooreGrid(
            (self.grid_width, self.grid_height),
            torus=False,
            capacity=1,
            random=self.random,
        )

        # ----- place agents -----
        self._place_agents()

        # ----- bootstrap reference income for tick-1 decisions-----
        # Without this, the very first tick's decisions in V2 would see
        # reference_income == own initial wage (the default in Woman.__init__),
        # which would understate the cross-type R differences at borders.
        for a in self.agents:
            self._update_one_reference_income(a)

        # ----- data collector -----
        self.datacollector = DataCollector(
            model_reporters={
                # State counts (used by the GUI line chart).
                "Blue": lambda m: _count_state(m, BLUE),
                "Pink": lambda m: _count_state(m, PINK),
                "Red": lambda m: _count_state(m, RED),
                "Orange": lambda m: _count_state(m, ORANGE),
                # Diagnostics.
                "MeanIncome": _mean_income,
                # Moran's I for the two motherhood states (the ones whose
                # spatial clustering H3/H5 actually predict).
                "MoransI_Orange": moran_i_orange,
                "MoransI_Red": moran_i_red,
            },
            agent_reporters={
                # Per-agent reporters are not strictly required for the GUI,
                # but useful for post-hoc analysis. Comment out if memory
                # becomes a concern at full scale × many seeds.
                "state": "state",
                "type": "type",
                "wage": "wage",
                "transfer": "spousal_transfer",
                "age": "age",
            },
        )
        # Collect t = 0 baseline before any step.
        self.datacollector.collect(self)

        # Solara dashboard checks this to decide whether to keep auto-stepping.
        self.running = True

    # Setup helpers
    def _place_agents(self) -> None:
        """Place agents in three contiguous regions, type-segregated.

        Within each region, fill GRID_OCCUPANCY (= 90%) of the cells.
        Cells are chosen uniformly at random from each region.

        Note: This satisfies the 2:5:3 type ratio AT FULL SCALE
        (height = 40, total cells = 2000, agents = 1800: 360 high + 900 median
        + 540 low → 0.2 : 0.5 : 0.3). At smaller scale_factor the totals shrink
        proportionally; the per-region 90% density is preserved.
        """
        # Materialize the cell list once. all_cells iterates lazily otherwise.
        # OrthogonalMooreGrid.all_cells is a CellCollection — .cells gives the
        # underlying list.
        all_cells = list(self.grid.all_cells.cells)

        for (col_lo, col_hi), agent_type in REGIONS:
            # Filter cells in this region.
            region_cells = [
                c for c in all_cells
                if col_lo <= c.coordinate[0] < col_hi
            ]
            n_target = int(round(len(region_cells) * GRID_OCCUPANCY))
            # Shuffle (in-place) for random selection.
            self.random.shuffle(region_cells)
            for cell in region_cells[:n_target]:
                # Constructor handles registration with the cell + agent set.
                Woman(self, cell, agent_type)

    # Reference-income update
    def _update_one_reference_income(self, agent: Woman) -> None:
        """Recompute one agent's reference_income from her Moore neighbors.

        Mean of (wage + spousal_transfer) over occupied neighbor cells, with
        ORANGE neighbors contributing their transfer-only income. If the agent
        has no occupied neighbors (an isolated edge case at low density), set
        R = own current income, which neutralizes the asymmetric penalty
        (max(0, log(1)) = 0) per §4.8.
        """
        incomes: List[float] = []
        for nbr_cell in agent.cell.neighborhood:
            for nbr in nbr_cell.agents:
                if nbr is agent:
                    continue  # shouldn't happen with capacity=1, but be safe
                incomes.append(nbr.current_income())
        if incomes:
            agent.reference_income = sum(incomes) / len(incomes)
        else:
            agent.reference_income = agent.current_income()

    def _update_all_reference_incomes(self) -> None:
        """Refresh every agent's reference_income at end-of-tick.

        Skipped when gamma == 0 because V1 doesn't consume R. (Saves a sweep
        over ~1800 agents × ~8 neighbors each tick, ~14k operations.)
        """
        if self.gamma <= 0.0:
            return
        for a in self.agents:
            self._update_one_reference_income(a)

    # Main step
    def step(self) -> None:
        """Advance the model by one year.

        Order of operations:
          1. All agents step in random order (each does her own Stage 1–3).
          2. End-of-tick: recompute every agent's reference_income (V2 only).
          3. Collect data.
          4. Check termination.
        """
        # 1. Activate agents (random order; per §4.9 the order is not specified
        #    to matter, but using shuffle_do avoids spurious correlations from
        #    a fixed iteration order).
        self.agents.shuffle_do("step")

        # 2. End-of-tick: refresh reference incomes for next tick (§4.8).
        self._update_all_reference_incomes()

        # 3. Collect aggregate + per-agent data.
        self.datacollector.collect(self)

        # 4. Termination: stop when all agents have reached END_AGE.
        if all(a.age >= END_AGE for a in self.agents):
            self.running = False
