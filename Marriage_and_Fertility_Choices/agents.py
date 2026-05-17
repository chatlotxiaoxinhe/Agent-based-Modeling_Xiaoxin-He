"""
Design
------------
* Each Woman represents one career and life-cycle horizon (age 22 to 60).
* State machine: BLUE → PINK → {RED, ORANGE} → (transfer-loss) → BLUE.
* Decision rule: deterministic 5-year discounted-utility maximization,
  evaluated whenever a marriage or fertility offer arrives.
* In V1 (gamma=0), utility is log(income+1).
* In V2 (gamma>0), utility = log(income+1) - gamma * max(0, log((R+1)/(I+1))),
  i.e. an *asymmetric* penalty that fires only when projected own income I
  falls below the agent's neighborhood-mean reference income R.

Timing
------------
Each tick, in order, inside Woman.step():
  Stage 1 — Clocks: age increments; spike_cooldown and career_break_timer
            decrement; if the break timer just hit 0, the 60% wage reset is
            applied here, BEFORE Stage 2.
  Stage 2 — Wage growth: applied only if the agent earns wage this tick
            (BLUE/PINK always; RED only when career_break_timer == 0; ORANGE
            never).
  Stage 3 — State-specific actions: spike check + offer evaluation for BLUE;
            fertility offer THEN dissolution roll for PINK; transfer-loss
            rolls for RED/ORANGE.

Reference income is *not* updated here; the model updates every agent's
reference_income at end-of-tick (so decisions in tick t use the R computed
at end of tick t-1).
"""

from __future__ import annotations

import math
from typing import List

from mesa.discrete_space import CellAgent

BLUE = "blue"      # single, career-focused
PINK = "pink"      # married, no children
RED = "red"        # mother, working (after a 2-year break)
ORANGE = "orange"  # mother, fully withdrawn from labor force

ALL_STATES = (BLUE, PINK, RED, ORANGE)

# Population shares for the three ability types.
TYPE_SHARES = {"high": 0.20, "median": 0.50, "low": 0.30}

# Initial wages at age 22.
INIT_WAGE = {"high": 300_000, "median": 200_000, "low": 100_000}

# Annual wage growth rates. Apply at start of each tick to non-orange agents
# whose career_break_timer == 0.
GROWTH_YOUNG = {"high": 0.20, "median": 0.10, "low": 0.05}  # ages 22–34
GROWTH_OLD = {"high": 0.05, "median": 0.03, "low": 0.01}    # ages 35+

# Once an agent has ever entered RED or ORANGE, growth_rate is permanently
# capped at this value, even after returning to BLUE.
GROWTH_CAP = 0.01

# Wage spikes
SPIKE_PROB = {"high": 0.05, "median": 0.03, "low": 0.01}
SPIKE_MULT = {"high": 2.00, "median": 1.50, "low": 1.30}
SPIKE_COOLDOWN_TICKS = 5  # length of cooldown imposed after any spike OR any
                          # return to BLUE (treated as re-integration period)

# Marriage offer parameters
def marriage_offer_prob(age: int, prev_married: bool) -> float:
    """
    Age- and history-dependent probability of receiving a marriage offer.

    A previously-married agent receives offers at half the standard rate
    at every age.
    """
    if age < 30:
        p = 1.00
    elif age < 35:
        p = 0.50
    elif age < 40:
        p = 0.30
    else:
        p = 0.05
    if prev_married:
        p *= 0.5
    return p

# Marriage transfer is drawn from Uniform(0.10, 0.30) × wage(t).
MARRIAGE_TRANSFER_FRAC_LO = 0.10
MARRIAGE_TRANSFER_FRAC_HI = 0.30

# Pink growth-rate attenuation on entry.
PINK_GROWTH_FACTOR = 0.70

# Pink dissolution probabilities.
PINK_DISSOLUTION_YOUNG = 0.02  # age < 35
PINK_DISSOLUTION_OLD = 0.05    # age >= 35

# Fertility offer parameters
def fertility_offer_prob(age: int) -> float:
    """
    Age-dependent probability of receiving a fertility offer.
    """
    if age < 35:
        return 1.00
    elif age < 40:
        return 0.30
    else:
        return 0.05

# Fertility transfer: 90% Uniform(0.20, 0.50) × wage(t); 10% high-percentile
# Uniform(200,000, 1,000,000) RMB absolute. 
FERTILITY_HIGH_TRANSFER_PROB = 0.10
FERTILITY_TYPICAL_FRAC_LO = 0.20
FERTILITY_TYPICAL_FRAC_HI = 0.50
FERTILITY_HIGH_TRANSFER_LO = 200_000
FERTILITY_HIGH_TRANSFER_HI = 1_000_000

# Career break and post-break wage resets.
CAREER_BREAK_DURATION = 2     # years of zero wage income after entering RED
RED_WAGE_RESET = 0.60         # year-3 reset factor for RED (of pre-break wage)
ORANGE_WAGE_RESET = 0.40      # return-to-blue reset factor for ORANGE

# Transfer-loss probabilities (§4.5).
RED_TRANSFER_LOSS = 0.03
ORANGE_TRANSFER_LOSS = 0.05

# Decision rule
FORECAST_HORIZON = 5  # 5-year forward window
DISCOUNT = 0.95       # per-year discount factor

# Life cycle
START_AGE = 22
END_AGE = 60          # last simulated age (inclusive)

# Woman agent
class Woman(CellAgent):
    """A single woman over her 22 to 60 life cycle.

    Inherits from Mesa 3.x ``CellAgent``, which provides cell-based positioning
    (``self.cell``) and automatic registration with both the model's agent set
    and the cell.
    """

    def __init__(self, model, cell, agent_type: str):
        """Initialize.

        model : MarriageFertilityModel
            The owning model.
        cell : mesa.discrete_space.Cell
            Grid cell this agent occupies. Position is fixed for the entire
            simulation.
        agent_type : {"high", "median", "low"}
            Ability type. Fixed at initialization.
        """
        super().__init__(model)
        self.cell = cell

        self.type: str = agent_type
        self.age: int = START_AGE
        self.state: str = BLUE

        self.wage: float = float(INIT_WAGE[agent_type])
        self.growth_rate: float = GROWTH_YOUNG[agent_type]
        self.spousal_transfer: float = 0.0

        # --- memory variables ---
        # Snapshot of own wage at the moment of entering RED or ORANGE.
        # Used to compute the 60% (RED) or 40% (ORANGE) reset wage.
        # Stored even in BLUE/PINK for consistency; only consumed at break end
        # or transfer loss.
        self.pre_break_wage: float = self.wage

        # --- flags & timers ---
        self.growth_capped: bool = False    # once True, stays True forever
        self.spike_cooldown: int = 0
        self.career_break_timer: int = 0    # ticks left in zero-wage break
        self.prev_married: bool = False     # ever been pink/red/orange?

        # --- reference income (V2 only) ---
        # The model bootstraps this after grid placement (see model.__init__)
        # so tick-1 decisions have a well-defined R.
        self.reference_income: float = self.wage

    # Public derived quantity
    def current_income(self) -> float:
        """
        Realized annual income this tick (used by both data collection and
        the model's reference_income update).

        ORANGE receives only transfer; RED on break receives only
        transfer; RED off break and PINK receive wage + transfer; 
        BLUE receives wage only.
        """
        if self.state == BLUE:
            return self.wage
        if self.state == PINK:
            return self.wage + self.spousal_transfer
        if self.state == RED:
            if self.career_break_timer > 0:
                return self.spousal_transfer
            return self.wage + self.spousal_transfer
        if self.state == ORANGE:
            return self.spousal_transfer
        raise ValueError(f"Unknown state: {self.state}")

    # Main step (called once per tick by the model)
    def step(self) -> None:
        """One year in this agent's life."""
        # End-of-life: agent stops acting once she reaches END_AGE.
        # The model schedules termination separately (model.step checks ages).
        if self.age >= END_AGE:
            return

        # ===== Stage 1: clocks =====
        # All clocks tick down BEFORE any state-dependent action, to avoid the
        # "wage grew, then we said it shouldn't have" time-travel bug.
        self.age += 1
        if self.spike_cooldown > 0:
            self.spike_cooldown -= 1
        if self.career_break_timer > 0:
            self.career_break_timer -= 1
            if self.career_break_timer == 0 and self.state == RED:
                # Break ends THIS tick: apply 60% reset against pre-break wage.
                # From the NEXT line onward (Stage 2), wage growth resumes at
                # the capped 1% rate (already set in _accept_red).
                self.wage = RED_WAGE_RESET * self.pre_break_wage

        # Age-35 boundary: shift growth rate to the older band ONLY for BLUE
        # non-capped agents. PINK keeps its frozen growth_rate, RED/ORANGE are
        # already capped at 0.01, and BLUE-after-break is also capped.
        # (See _baseline_growth_rate for the analogous logic on return-to-blue.)
        if (
            self.age == 35
            and self.state == BLUE
            and not self.growth_capped
        ):
            self.growth_rate = GROWTH_OLD[self.type]

        # ===== Stage 2: wage growth =====
        # Applies only if the agent earns wage income this tick.
        if self._earns_wage_this_tick():
            self.wage *= 1.0 + self.growth_rate

        # ===== Stage 3: state-specific actions =====
        if self.state == BLUE:
            self._maybe_spike()
            self._maybe_marriage_offer()
        elif self.state == PINK:
            # Per the user-confirmed schedule: fertility offer FIRST, then
            # (if she didn't accept and is still pink) dissolution roll.
            self._maybe_fertility_offer()
            if self.state == PINK:
                self._roll_pink_dissolution()
        elif self.state == RED:
            self._roll_red_transfer_loss()
        elif self.state == ORANGE:
            self._roll_orange_transfer_loss()

    # Helpers: eligibility
    def _earns_wage_this_tick(self) -> bool:
        """
        True iff the agent has positive wage income this tick.

        Per §4.2/§4.3:
          • BLUE, PINK: yes
          • RED with career_break_timer > 0: no (on break)
          • RED with career_break_timer == 0: yes (post-break)
          • ORANGE: no
        """
        if self.state == ORANGE:
            return False
        if self.state == RED and self.career_break_timer > 0:
            return False
        return True

    def _baseline_growth_rate(self) -> float:
        """
        Type/age/capped-aware growth rate. Used when an agent returns to
        BLUE (§4.4 dissolution; §4.5 transfer loss).

        Returns the cap (0.01) if growth_capped, else the age-band default.
        """
        if self.growth_capped:
            return GROWTH_CAP
        if self.age < 35:
            return GROWTH_YOUNG[self.type]
        return GROWTH_OLD[self.type]

    # Stage 3a: wage spike
    def _maybe_spike(self) -> None:
        """
        Roll for a promotion (BLUE agents, cooldown permitting).
        """
        if self.spike_cooldown > 0:
            return
        if self.random.random() < SPIKE_PROB[self.type]:
            self.wage *= SPIKE_MULT[self.type]
            self.spike_cooldown = SPIKE_COOLDOWN_TICKS

    # Stage 3b: marriage offer (BLUE → PINK)
    def _maybe_marriage_offer(self) -> None:
        """
        Roll for a marriage offer, then evaluate accept-pink vs reject.
        """
        p = marriage_offer_prob(self.age, self.prev_married)
        if self.random.random() >= p:
            return
        # Draw transfer offer.
        transfer = self.random.uniform(
            MARRIAGE_TRANSFER_FRAC_LO, MARRIAGE_TRANSFER_FRAC_HI
        ) * self.wage
        # Compare 5-year discounted utilities of the two scenarios.
        v_reject = self._forecast_value("reject_blue", offered_transfer=0.0)
        v_accept = self._forecast_value("accept_pink", offered_transfer=transfer)
        if v_accept > v_reject:
            self._accept_marriage(transfer)

    def _accept_marriage(self, transfer: float) -> None:
        """
        Transition BLUE → PINK with attenuated growth and offered transfer.
        """
        self.state = PINK
        self.growth_rate = PINK_GROWTH_FACTOR * self.growth_rate
        self.spousal_transfer = transfer
        self.prev_married = True

    # Stage 3c: pink dissolution (PINK → BLUE)
    def _roll_pink_dissolution(self) -> None:
        """
        Pink agents face an age-dependent dissolution risk each tick.
        """
        p = (
            PINK_DISSOLUTION_YOUNG if self.age < 35 else PINK_DISSOLUTION_OLD
        )
        if self.random.random() < p:
            self._return_to_blue(orange_wage_reset=False)

    # Stage 3d: fertility offer (PINK → RED or ORANGE)
    def _maybe_fertility_offer(self) -> None:
        """
        Roll for a fertility offer, then choose argmax over three scenarios.

        On accept-red or accept-orange the new transfer REPLACES (does not
        stack with) the existing marriage transfer, representing renegotiated
        intra-household resource sharing under Chiappori's collective-model
        sharing rule.
        """
        p = fertility_offer_prob(self.age)
        if self.random.random() >= p:
            return
        # Draw fertility transfer (mixture of typical and high-percentile).
        transfer = self._draw_fertility_transfer()
        # Three scenarios.
        v_reject = self._forecast_value("reject_pink", offered_transfer=0.0)
        v_red = self._forecast_value("accept_red", offered_transfer=transfer)
        v_orange = self._forecast_value("accept_orange", offered_transfer=transfer)
        # Pick argmax. Ties broken deterministically by max() returning first.
        best_value = max(v_reject, v_red, v_orange)
        if best_value == v_red and v_red >= v_orange and v_red >= v_reject:
            self._accept_red(transfer)
        elif best_value == v_orange and v_orange > v_red and v_orange >= v_reject:
            self._accept_orange(transfer)
        # else: reject (do nothing, stay PINK)

    def _draw_fertility_transfer(self) -> float:
        """Mixture: 90% typical (proportional to wage), 10% high absolute."""
        if self.random.random() < FERTILITY_HIGH_TRANSFER_PROB:
            return self.random.uniform(
                FERTILITY_HIGH_TRANSFER_LO, FERTILITY_HIGH_TRANSFER_HI
            )
        return self.random.uniform(
            FERTILITY_TYPICAL_FRAC_LO, FERTILITY_TYPICAL_FRAC_HI
        ) * self.wage

    def _accept_red(self, transfer: float) -> None:
        """Transition PINK → RED with 2-year break, capped growth, transfer
        replacement."""
        # CRITICAL: snapshot BEFORE any wage mutation.
        self.pre_break_wage = self.wage
        self.state = RED
        self.spousal_transfer = transfer       # overwrites marriage transfer
        self.growth_capped = True
        self.growth_rate = GROWTH_CAP
        self.career_break_timer = CAREER_BREAK_DURATION
        # Wage stays at pre_break_wage until timer hits 0; Stage 1 of that
        # future tick applies the 60% reset.
        self.prev_married = True

    def _accept_orange(self, transfer: float) -> None:
        """
        Transition PINK → ORANGE: out of labor force, transfer only.
        """
        self.pre_break_wage = self.wage
        self.state = ORANGE
        self.spousal_transfer = transfer       # overwrites marriage transfer
        self.growth_capped = True
        self.growth_rate = GROWTH_CAP
        # Wage frozen at pre_break_wage until/unless she returns to blue, in
        # which case the 40% reset applies.
        self.prev_married = True

    # Stage 3e: red / orange transfer loss (→ BLUE with penalties)
    def _roll_red_transfer_loss(self) -> None:
        """
        Red agents face a 3% annual transfer-loss risk.

        On loss, she returns to BLUE but retains all wage penalties
        (growth_capped stays True; wage continues from its current value,
        which is either pre_break_wage during break or 0.6×pre_break_wage
        after break ended).
        """
        if self.random.random() < RED_TRANSFER_LOSS:
            self._return_to_blue(orange_wage_reset=False)

    def _roll_orange_transfer_loss(self) -> None:
        """
        Orange agents face a 5% annual transfer-loss risk.

        On loss, return to BLUE with wage reset to 40% of pre_break_wage.
        """
        if self.random.random() < ORANGE_TRANSFER_LOSS:
            self._return_to_blue(orange_wage_reset=True)

    # Shared transition: return to BLUE
    def _return_to_blue(self, orange_wage_reset: bool) -> None:
        """
        Transition any non-blue state back to BLUE.

        orange_wage_reset : bool
            Only true when called from ORANGE transfer loss; applies the 40%
            wage reset against pre_break_wage.

        Per user-confirmed rules:
          • spike_cooldown is reset to 5 (re-integration period; applies even
            if the agent never spiked).
          • growth_rate is recomputed by current (type, age, growth_capped).
          • spousal_transfer is zeroed.
          • career_break_timer is cleared (RED-during-break → BLUE case).
          • prev_married stays True if it was True.
        """
        if orange_wage_reset:
            self.wage = ORANGE_WAGE_RESET * self.pre_break_wage
        self.state = BLUE
        self.spousal_transfer = 0.0
        self.career_break_timer = 0
        self.growth_rate = self._baseline_growth_rate()
        self.spike_cooldown = SPIKE_COOLDOWN_TICKS

    # Decision rule: 5-year discounted forecast
    def _forecast_value(self, scenario: str, offered_transfer: float) -> float:
        """
        Compute V(s) for scenario s.

        V(s) = Σ_{h=1..5} δ^(h-1) · u_s(h)

        where u_s(h) = log(I+1) − γ·max(0, log((R+1)/(I+1))) and I = income_s(h).
        R = self.reference_income is held constant over the 5-year horizon
        (a bounded-rationality simplification, §4.6).
        """
        incomes = self._project_income_stream(scenario, offered_transfer)
        R = self.reference_income
        gamma = self.model.gamma
        total = 0.0
        for h, income in enumerate(incomes):
            # Absolute-utility component (V1 baseline).
            u = math.log(income + 1.0)
            # Asymmetric status-anxiety penalty (V2 addition).
            if gamma > 0.0:
                gap = math.log((R + 1.0) / (income + 1.0))
                if gap > 0.0:           # only fires when income < R
                    u -= gamma * gap
            total += (DISCOUNT ** h) * u
        return total

    def _project_income_stream(
        self, scenario: str, offered_transfer: float
    ) -> List[float]:
        """
        Project a 5-year income stream under the named scenario.

        The forecast is deterministic: no stochastic events (dissolution,
        transfer loss, wage spikes) are simulated inside the forecast.
        Wage spikes in particular are explicitly excluded as unforeseen
        windfalls.
        """
        H = FORECAST_HORIZON
        incomes: List[float] = []

        if scenario == "reject_blue":
            # Continue as BLUE with the agent's CURRENT growth rate.
            # Project forward, NOT crossing age 35 mid-forecast for simplicity
            # (the spec specifies "current growth rate", not piecewise).
            w = self.wage
            r = self.growth_rate
            for _ in range(H):
                w *= 1.0 + r
                incomes.append(w)

        elif scenario == "accept_pink":
            # Accept-pink: growth_rate -> 0.7 * current; receive transfer
            # each year.
            w = self.wage
            r = PINK_GROWTH_FACTOR * self.growth_rate
            for _ in range(H):
                w *= 1.0 + r
                incomes.append(w + offered_transfer)

        elif scenario == "reject_pink":
            # Stay PINK with current growth_rate and existing transfer.
            w = self.wage
            r = self.growth_rate
            t = self.spousal_transfer
            for _ in range(H):
                w *= 1.0 + r
                incomes.append(w + t)

        elif scenario == "accept_red":
            # Year 1–2: zero wage income (just the transfer).
            # Year 3 onward: 0.6 × pre-break wage, growing at 1%.
            pre_break = self.wage
            reset_wage = RED_WAGE_RESET * pre_break
            for h in range(H):
                if h < CAREER_BREAK_DURATION:
                    incomes.append(offered_transfer)
                else:
                    years_since_resume = h - CAREER_BREAK_DURATION
                    w = reset_wage * ((1.0 + GROWTH_CAP) ** years_since_resume)
                    incomes.append(w + offered_transfer)

        elif scenario == "accept_orange":
            # Transfer alone, for 5 years.
            for _ in range(H):
                incomes.append(offered_transfer)

        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")

        return incomes
