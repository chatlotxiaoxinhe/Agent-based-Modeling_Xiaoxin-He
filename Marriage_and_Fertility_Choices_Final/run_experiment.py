"""
Runs the model across multiple γ values × multiple random seeds at full scale,
collects time-series and final-tick data, writes CSV outputs, and generates
two analysis plots:

  1. plot_morans_i.png  — Moran's I_Red trajectories (mean ± 1 std across
                          seeds), one curve per γ. 
  2. plot_h1_type_red.png — Type-stratified fraction in RED at age 60.
                            Tests H1 ("low > median > high entry into red").

Outputs:
  - batch_time_series.csv   : per (γ, seed, step) model reporters
  - batch_final_states.csv  : per (γ, seed, agent) final state + type
  - plot_morans_i.png       : H3/H5 visualization
  - plot_h1_type_red.png    : H1 visualization

Usage:
    python run_experiment.py

Runtime: roughly 5–8 minutes for the default 3 γ × 5 seeds = 15 full-scale
runs (each: 1800 agents over 38 ticks). To shorten for a quick test, edit
the GAMMAS / SEEDS / SCALE_FACTOR constants below.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import MarriageFertilityModel

# Experiment configuration
GAMMAS: List[float] = [0.0, 0.3, 0.6]   # social-comparison strengths to test
SEEDS: List[int] = [0, 1, 2, 3, 4]      # random seeds (one run per seed per γ)
SCALE_FACTOR: float = 1.0               # 1.0 = full 50×40 grid, 1800 agents
N_SEEDS = len(SEEDS)

# Output paths
CSV_TIME_PATH = "batch_time_series.csv"
CSV_FINAL_PATH = "batch_final_states.csv"
PLOT_MORANS_PATH = "plot_morans_i.png"
PLOT_H1_PATH = "plot_h1_type_red.png"
PLOT_WITHIN_TYPE_PATH = "plot_within_type_morans.png"

# Color scheme reused in both plots (consistent with app.py state colors,
# slightly muted for line/bar plots).
GAMMA_COLORS = {
    0.0: "#888888",     # gray — V1 baseline
    0.3: "#ff7f00",     # orange — moderate γ
    0.6: "#e41a1c",     # red — strong γ
}
TYPE_COLORS = {
    "high":   "#1f77b4",
    "median": "#7f7f7f",
    "low":    "#d62728",
}

# Batch runner
def run_batch(
    gammas: List[float],
    seeds: List[int],
    scale_factor: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the model once per (γ, seed) combination.

    Each run is allowed to terminate naturally (38 ticks, until all agents
    reach age 60). Returns two DataFrames:

      df_time  : long-format time series. Columns:
                  step, Blue, Pink, Red, Orange, MeanIncome,
                  MoransI_Orange, MoransI_Red, gamma, seed.
      df_final : per-agent final state. Columns:
                  gamma, seed, type, state, is_red.
    """
    records_time = []
    records_final = []
    total = len(gammas) * len(seeds)

    for i, gamma in enumerate(gammas):
        for j, seed in enumerate(seeds):
            run_id = i * len(seeds) + j + 1
            t0 = time.time()
            print(
                f"  [{run_id:2d}/{total}] γ={gamma}, seed={seed:2d} … ",
                end="", flush=True
            )

            # ----- run -----
            m = MarriageFertilityModel(
                gamma=gamma, scale_factor=scale_factor, seed=seed
            )
            while m.running:
                m.step()
            elapsed = time.time() - t0

            # ----- collect time series -----
            ts = m.datacollector.get_model_vars_dataframe().reset_index(
                names="step"
            )
            ts["gamma"] = gamma
            ts["seed"] = seed
            records_time.append(ts)

            # ----- collect final agent states -----
            for a in m.agents:
                records_final.append({
                    "gamma": gamma,
                    "seed": seed,
                    "type": a.type,
                    "state": a.state,
                })

            n_red = sum(1 for a in m.agents if a.state == "red")
            print(f"done in {elapsed:.1f}s  (red={n_red})")

            # Help garbage-collect the (large) per-agent reporter dataframe.
            del m

    df_time = pd.concat(records_time, ignore_index=True)
    df_final = pd.DataFrame(records_final)
    df_final["is_red"] = (df_final["state"] == "red").astype(int)
    return df_time, df_final

# Plot 1: Moran's I_Red trajectories (one curve per γ, ±1 std band)
def plot_morans_i_trajectories(
    df_time: pd.DataFrame, gammas: List[float], out_path: str
) -> None:
    """
    Mean ± 1 std Moran's I_Red trajectory, one curve per γ.

    A curve significantly above the γ=0 baseline (especially in mean height
    or temporal breadth) is the model-level signature of V2's local
    social-comparison feedback.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for gamma in gammas:
        sub = df_time[df_time["gamma"] == gamma]
        agg = sub.groupby("step")["MoransI_Red"].agg(["mean", "std"]).reset_index()
        color = GAMMA_COLORS.get(gamma, "#000000")
        ax.plot(
            agg["step"], agg["mean"],
            label=f"γ = {gamma}", linewidth=2.0, color=color,
        )
        ax.fill_between(
            agg["step"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.18, color=color, linewidth=0,
        )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Tick (years from age 22)")
    ax.set_ylabel("Moran's I for RED state")
    ax.set_title(
        "Spatial autocorrelation of the working-mother (RED) state over the life cycle\n"
        f"Mean ± 1 std across {N_SEEDS} seeds, full scale (1800 agents on 50×40 grid)"
    )
    ax.legend(title="Social comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# Plot 2: H1 — Type-stratified RED rate at age 60
def plot_h1_type_red(
    df_final: pd.DataFrame, gammas: List[float], out_path: str
) -> None:
    """
    Bar chart of the fraction of each type in RED at the end of life.

    H1 predicts: low > median > high, monotonically. The plot also shows
    how this gradient changes across γ (a secondary, indirect check on
    whether social comparison preserves the type ordering).
    """
    # mean & std across seeds, per (gamma, type)
    by_seed = (
        df_final.groupby(["gamma", "seed", "type"])["is_red"].mean().reset_index()
    )
    summary = (
        by_seed.groupby(["gamma", "type"])["is_red"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    types = ["high", "median", "low"]
    bar_width = 0.25
    x = np.arange(len(gammas))

    for k, type_name in enumerate(types):
        means = [
            summary[(summary["gamma"] == g) & (summary["type"] == type_name)]["mean"].iloc[0]
            for g in gammas
        ]
        stds = [
            summary[(summary["gamma"] == g) & (summary["type"] == type_name)]["std"].iloc[0]
            for g in gammas
        ]
        ax.bar(
            x + (k - 1) * bar_width, means, bar_width,
            yerr=stds, capsize=4,
            label=type_name.capitalize(),
            color=TYPE_COLORS[type_name],
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"γ = {g}" for g in gammas])
    ax.set_ylabel("Fraction of agents in RED at age 60")
    ax.set_title(
        "H1: type-stratified working-mother rate at end of life cycle\n"
        f"Mean ± 1 std across {N_SEEDS} seeds, full scale"
    )
    ax.legend(title="Type")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

# Plot 3: Within-type Moran's I_Red trajectories (one panel per type)
def plot_within_type_morans(
    df_time: pd.DataFrame, gammas: List[float], out_path: str
) -> None:
    """
    Within-type Moran's I_Red for high/median/low, one panel per type.

    The basic Moran's I (Figure 2) is contaminated by type-region structure:
    it picks up the fact that low-type regions have more reds than high-type
    regions, which is true even at γ=0. The within-type variant strips this
    out by asking: *within agents of the same type, are reds clustered more
    than random?*

    Reading the figure:
      - If γ creates within-type clustering, the γ=0.6 trajectory should
        rise above the γ=0 baseline, especially around the red-entry peak
        (steps 10-15).
      - If γ instead operates through across-type contrast amplification
        with no within-type clustering, all three γ trajectories should
        overlap and stay near zero.
    """
    types = ["high", "median", "low"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, tp in zip(axes, types):
        col_name = f"MoransI_Red_within_{tp}"
        for gamma in gammas:
            sub = df_time[df_time["gamma"] == gamma]
            agg = sub.groupby("step")[col_name].agg(
                ["mean", "std"]
            ).reset_index()
            color = GAMMA_COLORS.get(gamma, "#000000")
            ax.plot(
                agg["step"], agg["mean"],
                label=f"γ = {gamma}", linewidth=1.8, color=color,
            )
            ax.fill_between(
                agg["step"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.15, color=color, linewidth=0,
            )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Tick (years from age 22)")
        ax.set_title(f"Within-{tp}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Within-type Moran's I for RED")
    axes[-1].legend(title="Social comparison", loc="upper right")
    fig.suptitle(
        "Within-type Moran's I (RED) — residual spatial clustering after "
        "stripping out type-region background\n"
        f"Mean ± 1 std across {N_SEEDS} seeds, full scale"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

# Summary statistics
def print_summary(df_time: pd.DataFrame, df_final: pd.DataFrame) -> None:
    """Print key numbers from the experiment to stdout."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # H1 check: red rate by gamma × type
    table_h1 = (
        df_final.groupby(["gamma", "type"])["is_red"].mean().unstack(level="type")
    )
    # Reorder columns to high/median/low for readability
    table_h1 = table_h1[["high", "median", "low"]]
    print("\nFraction in RED at age 60 (mean over seeds):")
    print(table_h1.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    # Check H1 ordering
    print("H1 predicts low > median > high. Per-γ check:")
    for g in table_h1.index:
        row = table_h1.loc[g]
        ordered = row["low"] > row["median"] > row["high"]
        ok = "✓" if ordered else "✗"
        print(
            f"  γ = {g}: low={row['low']:.3f} > median={row['median']:.3f} "
            f"> high={row['high']:.3f}   {ok}"
        )

    # Moran's I_Red peak comparison
    print("\nMoran's I_Red peak (mean over seeds):")
    for g in df_time["gamma"].unique():
        sub = df_time[df_time["gamma"] == g]
        agg = sub.groupby("step")["MoransI_Red"].mean()
        print(
            f"  γ = {g}: peak = {agg.max():.3f} at step {agg.idxmax()};   "
            f"final = {agg.iloc[-1]:.3f}"
        )

    # Cross-seed variance of final RED count (H3 says variance should grow with γ)
    print("\nCross-seed variance of final RED count (H3: should grow with γ):")
    by_seed_red = df_final[df_final["state"] == "red"].groupby(["gamma", "seed"]).size()
    by_seed_red = by_seed_red.unstack(level="seed", fill_value=0)
    for g in by_seed_red.index:
        counts = by_seed_red.loc[g].values
        print(
            f"  γ = {g}: mean={counts.mean():6.1f}  std={counts.std():5.1f}  "
            f"range=[{counts.min()}, {counts.max()}]"
        )

    # Within-type Moran's I peaks (should isolate γ's pure spatial effect if any)
    print("\nWithin-type Moran's I (Red) peak by gamma × type (mean over seeds):")
    print(f"  {'gamma':>6}  {'within-high':>12}  {'within-median':>14}  {'within-low':>12}")
    for g in df_time["gamma"].unique():
        sub = df_time[df_time["gamma"] == g]
        peaks = []
        for tp in ["high", "median", "low"]:
            col = f"MoransI_Red_within_{tp}"
            agg = sub.groupby("step")[col].mean()
            # Use absolute peak (positive or negative) for signal magnitude
            peak_val = agg.iloc[agg.abs().idxmax()] if len(agg) > 0 else 0.0
            peaks.append(peak_val)
        print(
            f"  {g:>6}  {peaks[0]:>+12.4f}  {peaks[1]:>+14.4f}  {peaks[2]:>+12.4f}"
        )

# Main
if __name__ == "__main__":
    print("Marriage & Fertility ABM — batch experiment")
    print(f"  γ values:     {GAMMAS}")
    print(f"  seeds:        {SEEDS}")
    print(f"  scale_factor: {SCALE_FACTOR}")
    print(f"  total runs:   {len(GAMMAS) * len(SEEDS)}")
    print()
    print("Running …")

    t_start = time.time()
    df_time, df_final = run_batch(GAMMAS, SEEDS, SCALE_FACTOR)
    elapsed = time.time() - t_start
    print(f"\nAll runs complete in {elapsed:.1f}s.\n")

    # CSV outputs
    df_time.to_csv(CSV_TIME_PATH, index=False)
    df_final.to_csv(CSV_FINAL_PATH, index=False)
    print(f"Saved time-series CSV : {CSV_TIME_PATH}  ({len(df_time):,} rows)")
    print(f"Saved final-state CSV : {CSV_FINAL_PATH}  ({len(df_final):,} rows)")

    # Plots
    plot_morans_i_trajectories(df_time, GAMMAS, PLOT_MORANS_PATH)
    plot_h1_type_red(df_final, GAMMAS, PLOT_H1_PATH)
    plot_within_type_morans(df_time, GAMMAS, PLOT_WITHIN_TYPE_PATH)
    print(f"Saved plot            : {PLOT_MORANS_PATH}")
    print(f"Saved plot            : {PLOT_H1_PATH}")
    print(f"Saved plot            : {PLOT_WITHIN_TYPE_PATH}")

    print_summary(df_time, df_final)
