import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from agents import CulturalAgent

class AxelrodModel(Model):
    """Axelrod's cultural dissemination model on a grid of fixed sites.

    Parameters
    ----------
    width, height : int
        Grid dimensions. Axelrod tested 5x5 to 100x100.
    num_features : int
        F, number of cultural features (Axelrod's Table 2: 5, 10, 15).
    num_traits : int
        q, number of possible traits per feature (Axelrod's Table 2:
        5, 10, 15).
    neighborhood_size : int
        Must be 4, 8, or 12. See module docstring item 2.
    torus : bool
        If True, the grid wraps (no boundaries). Axelrod tests both
        boundary conditions on p. 215.
    events_per_step : int
        Number of asynchronous events per model step. Affects only
        visualization refresh rate, not the dynamics: each event
        independently selects one active site, exactly as in the paper.
    seed : int or None
        Seed for the model's RNG.
    """

    def __init__(
        self,
        width=10,
        height=10,
        num_features=5,
        num_traits=10,
        neighborhood_size=4,
        torus=False,
        events_per_step=200,
        seed=None,
    ):
        # Mesa deprecated `seed=` in favor of `rng=`; both produce a
        # seeded self.random. We keep `seed` as our public parameter
        # because that's the more familiar term in the modeling literature.
        super().__init__(rng=seed)

        self.width = width
        self.height = height
        self.num_features = int(num_features)
        self.num_traits = int(num_traits)
        self.neighborhood_size = int(neighborhood_size)
        self.events_per_step = int(events_per_step)

        # SingleGrid: exactly one agent per cell. Sites never move; we
        # never call grid.move_agent(). torus=True wraps the boundary.
        self.grid = SingleGrid(width, height, torus=torus)

        # Place one agent per cell. Each agent samples its own culture
        # from model.random in __init__
        for x in range(width):
            for y in range(height):
                agent = CulturalAgent(self)
                self.grid.place_agent(agent, (x, y))

        # Cache the agent list. The set of agents never changes during a
        # run, so building it once saves work on every event selection.
        self._agent_list = list(self.agents)

        # Track total events for diagnostics. (Mesa's plot uses model
        # step number as the x-axis automatically; total_events is the
        # equivalent of Axelrod's "events" axis but at finer granularity.)
        self.total_events = 0

        # Outcome measures collected each model step (= every
        # events_per_step events).
        self.datacollector = DataCollector(
            model_reporters={
                "Cultural Regions": count_regions,
                "Cultural Zones": count_zones,
                "Largest Region (%)": largest_region_pct,
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def get_neighbors(self, pos):
        """Return the list of neighbor agents for the cell at `pos`.

        Mesa's SingleGrid supports von Neumann (4) and Moore (8) directly.
        For Axelrod's 12-neighborhood (p. 213), we extend Moore-8 with the
        four cells two units away in cardinal directions.
        """
        if self.neighborhood_size == 4:
            return list(self.grid.iter_neighbors(pos, moore=False))
        if self.neighborhood_size == 8:
            return list(self.grid.iter_neighbors(pos, moore=True))
        if self.neighborhood_size == 12:
            return self._get_diamond12_neighbors(pos)
        raise ValueError(
            f"neighborhood_size must be 4, 8, or 12; got {self.neighborhood_size}"
        )

    def _get_diamond12_neighbors(self, pos):
        """Axelrod's 12-cell diamond neighborhood: Moore-8 plus the four
        cells at distance 2 in the cardinal directions.

        We implement this manually because Mesa's grid only knows about
        Moore (Chebyshev distance <= r) and von Neumann (Manhattan
        distance <= r) neighborhoods. The 12-neighborhood is neither —
        it's Moore-1 ∪ {(±2, 0), (0, ±2)}.
        """
        # 8 Moore offsets + 4 cardinal-2 offsets = 12 candidate cells
        offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
            (-2,  0), ( 2, 0),
            ( 0, -2), ( 0, 2),
        )
        x, y = pos
        seen = set()  # dedupe: on a small torus, two offsets can map to
                      # the same cell (e.g. (2,0) and (-3,0) on width=5)
        out = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if self.grid.torus:
                nx %= self.width
                ny %= self.height
            else:
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
            cell = (nx, ny)
            if cell == pos or cell in seen:
                continue
            seen.add(cell)
            contents = self.grid.get_cell_list_contents([cell])
            if contents:
                out.append(contents[0])
        return out

    def step(self):
        """Run `events_per_step` asynchronous events, then collect data
        and check whether the configuration is frozen.

        Each event:
          - picks one random active site uniformly over all N sites
            (matches "at random pick a site to be active")
          - the active site executes one interact() call
          - total_events is incremented

        Crucially, this loop is sequential, by the time event k+1 begins,
        event k's culture change is already visible. That preserves the
        asynchronous semantics required by the paper.
        """
        if not self.running:
            return

        for _ in range(self.events_per_step):
            agent = self.random.choice(self._agent_list)
            agent.interact()
            self.total_events += 1

        self.datacollector.collect(self)

        # Stop if no further change is possible. The check is cheap
        # relative to the events_per_step events we just ran.
        if self.is_frozen():
            self.running = False

    def is_frozen(self):
        """True if every adjacent pair has similarity in {0, 1}.

        "every pair of neighboring sites has
        cultures that are either identical or completely different."
        Once true, no future event can change anything: 
        identical neighbors interact but the change is a no-op; completely
        different neighbors will not interact (probability 0).

        Cost: O(N · max_neighbors · F). Cheap enough to run every step.
        """
        F = self.num_features
        for agent in self._agent_list:
            for nb in self.get_neighbors(agent.pos):
                # Count matches without computing the full ratio.
                matches = int(np.sum(agent.culture == nb.culture))
                if 0 < matches < F:
                    return False
        return True

def count_regions(model):
    """Number of cultural regions: maximal connected components
    of sites with identical cultures.
    """
    visited = set()
    count = 0
    for agent in model._agent_list:
        if agent.unique_id in visited:
            continue
        count += 1
        culture = agent.culture
        # Iterative DFS through neighbors with the same culture.
        stack = [agent]
        while stack:
            current = stack.pop()
            if current.unique_id in visited:
                continue
            visited.add(current.unique_id)
            for nb in model.get_neighbors(current.pos):
                if (nb.unique_id not in visited and
                        np.array_equal(nb.culture, culture)):
                    stack.append(nb)
    return count


def count_zones(model):
    """Number of cultural zones: maximal connected
    components where each adjacent pair on the path shares >= 1 feature.

    Zones are the upper bound on the eventual region count: when no two
    adjacent sites in a zone differ on every feature, zones cannot
    fragment further. Figure 3 shows zones converge much faster than
    regions, so plotting both tells the convergence story clearly.
    """
    visited = set()
    count = 0
    for agent in model._agent_list:
        if agent.unique_id in visited:
            continue
        count += 1
        stack = [agent]
        while stack:
            current = stack.pop()
            if current.unique_id in visited:
                continue
            visited.add(current.unique_id)
            for nb in model.get_neighbors(current.pos):
                if nb.unique_id in visited:
                    continue
                # Compatibility = at least one shared feature
                if bool(np.any(current.culture == nb.culture)):
                    stack.append(nb)
    return count


def largest_region_pct(model):
    """Size of the largest cultural region as a percentage of total cells.

    Useful as a "monoculture" indicator: 100% means a single culture has
    taken over the entire grid.
    """
    visited = set()
    largest = 0
    total = model.width * model.height
    for agent in model._agent_list:
        if agent.unique_id in visited:
            continue
        size = 0
        culture = agent.culture
        stack = [agent]
        while stack:
            current = stack.pop()
            if current.unique_id in visited:
                continue
            visited.add(current.unique_id)
            size += 1
            for nb in model.get_neighbors(current.pos):
                if (nb.unique_id not in visited and
                        np.array_equal(nb.culture, culture)):
                    stack.append(nb)
        if size > largest:
            largest = size
    return 100.0 * largest / total
