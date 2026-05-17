# Marriage & Fertility ABM

An agent-based model exploring how forward-looking economic utility, alone or in
combination with social comparison, shapes women's marriage and fertility
decisions. 
Implements Versions 1 (γ = 0) and 2 (γ > 0)of the model specified in the paper.

## Files

- **`agents.py`** — `Woman` agent class. State machine (blue/pink/red/orange),
  five-year forward-looking decision rule, asymmetric status-anxiety utility.
- **`model.py`** — `MarriageFertilityModel` class. Grid layout, scheduling,
  end-of-tick reference-income update, data collection, Moran's I helper.
- **`app.py`** — Solara dashboard. Sliders for γ / grid scale / seed; colored
  grid; live state-share chart; Moran's I time series.

## Requirements

- Python 3.10+
- Mesa 3.1 – 3.x (tested on 3.5.1)
- Solara, matplotlib, numpy, altair

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
solara run app.py
```

then open the URL Solara prints (default `http://localhost:8765`). Adjust the
sliders, press the **play** button to auto-step, **step** to advance one year.
Simulation auto-terminates when all agents reach age 60 (38 ticks).

## Run headlessly

```python
from model import MarriageFertilityModel

m = MarriageFertilityModel(gamma=0.3, scale_factor=1.0, seed=42)
while m.running:
    m.step()

df = m.datacollector.get_model_vars_dataframe()
agent_df = m.datacollector.get_agent_vars_dataframe()
```

## Parameters

| Name           | Default | Description                                              |
| `gamma`        | 0.3     | Strength of asymmetric social-comparison penalty. γ = 0 reproduces V1. |
| `scale_factor` | 0.5 (GUI) / 1.0 (headless) | Fraction of grid height. Width and column boundaries (10/25/15) are always fixed; only height scales. |
| `seed`         | 42      | Master random seed. Controls type assignment, placement, offers, transfers. |

To expose other parameters (transfer ranges, dissolution probabilities, etc.)
as sliders, add a constructor argument to `MarriageFertilityModel.__init__` and
a slider entry in `app.py`'s `model_params` dict.

## Visualization legend

- **Blue squares**: single, career-focused (`blue`)
- **Pink squares**: married, no children (`pink`)
- **Red squares**: mother continuing to work after a two-year break (`red`)
- **Orange squares**: mother fully withdrawn from labor force (`orange`)

The three contiguous columnar regions correspond to ability types (high-type
left, median middle, low-type right).

## Known behavior: orange state stays empty

Under the deterministic five-year forecast specified in §4.6 of the paper,
accept-orange is strictly dominated by accept-red whenever the agent's
pre-break wage is positive. This follows directly from the per-year utility
algebra: red's years 3–5 add `log(0.6 · pre_break_wage · 1.01^k + T + 1)` over
orange's `log(T + 1)` — a strictly positive quantity. The model is implemented
faithfully to this specification, so empirically zero agents ever enter the
orange state across all γ and seeds. This is a property of the specified
decision rule, not a coding error. Possible substantive resolutions (e.g.
higher transfer in the orange branch reflecting renegotiated household
sharing, or a leisure premium in orange) are discussed in the paper.

## Code structure

`agents.py` is organized in this order:

1. State constants (`BLUE`, `PINK`, `RED`, `ORANGE`)
2. Type-keyed economic parameters (initial wages, growth rates, spike rates)
3. Offer-arrival probability functions (`marriage_offer_prob`, `fertility_offer_prob`)
4. Decision-rule constants (horizon, discount)
5. `Woman` class:
   - `__init__` (attributes; including memory variable `pre_break_wage`)
   - `current_income()` — actual income this tick
   - `step()` — orchestrator: clocks, wage growth, state-specific actions
   - Eligibility helpers (`_earns_wage_this_tick`, `_baseline_growth_rate`)
   - Action methods (`_maybe_spike`, `_maybe_marriage_offer`, etc.)
   - Transition methods (`_accept_marriage`, `_accept_red`, `_accept_orange`, `_return_to_blue`)
   - Decision rule (`_forecast_value`, `_project_income_stream`)

`model.py` is organized:

1. Grid-geometry constants (always-fixed width/regions, scalable height)
2. Aggregate reporters (state counts, Moran's I)
3. `MarriageFertilityModel`:
   - `__init__` (grid build, agent placement, reference-income bootstrap, DataCollector)
   - `_place_agents()` — region-wise random placement with 90% density
   - Reference-income update (skipped when γ = 0 for performance)
   - `step()` — `shuffle_do("step")` → reference update → collect → check termination
