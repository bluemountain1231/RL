"""Parameter sensitivity analysis for NOA (Section 4.4).

Tests three key parameters with single-factor variation:
  1. Population size:  P = {4, 8, 12, 16, 24}
  2. Cache weight:     gamma_c = {0, 0.15, 0.35, 0.55, 0.75}
  3. Recovery window:  W = {0, 3, 6, 10, 14}

Each configuration runs 3 random seeds. Results are printed as tables
and saved as CSV + plots.

Usage:
  python experiments/noa_param_sensitivity.py -s 100000
  python experiments/noa_param_sensitivity.py -s 100000 --runs 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── SUMO path setup ──────────────────────────────────────────────────────────
def _ensure_sumo_home() -> None:
    if "SUMO_HOME" in os.environ:
        return
    default = "/usr/share/sumo"
    if os.path.isdir(default):
        os.environ["SUMO_HOME"] = default
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")


_ensure_sumo_home()
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sumo_rl import SumoEnvironment
from sumo_rl.agents import NOAAgent
from sumo_rl.environment.observations import NOAObservationFunction


def _repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


# ── Default NOA parameters (Table 4-2) ───────────────────────────────────────
DEFAULT_PARAMS = dict(
    population_size=12,
    elite_size=4,
    cache_size=256,
    alpha=0.6,
    beta=0.3,
    sigma=0.15,
    recovery_window=6,
    recovery_queue_threshold=0.75,
    recovery_delay_threshold=0.7,
    cache_weight=0.35,
    switch_penalty=0.08,
    weight_queue=0.35,
    weight_delay=0.3,
    weight_pressure=0.15,
    weight_throughput=0.2,
    fitness_weight_queue=0.4,
    fitness_weight_delay=0.3,
    fitness_weight_speed=0.15,
    fitness_weight_throughput=0.15,
)

# ── Sensitivity test definitions ─────────────────────────────────────────────
SENSITIVITY_TESTS = {
    "population_size": {
        "param_name": "Population Size (P)",
        "values": [4, 8, 12, 16, 24],
        "default": 12,
        "override_key": "population_size",
        "x_label": "Population Size",
    },
    "cache_weight": {
        "param_name": "Cache Weight (gamma_c)",
        "values": [0.0, 0.15, 0.35, 0.55, 0.75],
        "default": 0.35,
        "override_key": "cache_weight",
        "x_label": "Cache Weight",
    },
    "recovery_window": {
        "param_name": "Recovery Window (W)",
        "values": [0, 3, 6, 10, 14],
        "default": 6,
        "override_key": "recovery_window",
        "x_label": "Recovery Window",
    },
}


def _build_metrics(info: dict, seconds: int, elapsed: float) -> dict:
    """Extract key metrics from simulation info dict."""
    queue_length = info.get("agents_total_stopped", info.get("system_total_stopped", 0))
    avg_delay = info.get("system_mean_waiting_time", 0.0)
    throughput_total = info.get("system_total_arrived", 0)
    mean_speed = info.get("system_mean_speed", 0.0)
    teleported = info.get("system_total_teleported", 0)
    return {
        "queue_length": queue_length,
        "average_delay": avg_delay,
        "throughput_total": throughput_total,
        "mean_speed": mean_speed,
        "teleported": teleported,
        "compute_time": elapsed,
    }


def _run_single(
    param_overrides: dict,
    seconds: int,
    seed: int,
    route: str,
    min_green: int,
    max_green: int,
) -> dict:
    """Run a single simulation with given parameter overrides."""
    env = SumoEnvironment(
        net_file=_repo_path("sumo_rl/nets/single-intersection/single-intersection.net.xml"),
        route_file=_repo_path(route),
        out_csv_name=None,
        use_gui=False,
        num_seconds=seconds,
        min_green=min_green,
        max_green=max_green,
        single_agent=True,
        observation_class=NOAObservationFunction,
        reward_fn="diff-waiting-time",
    )

    agent_params = dict(DEFAULT_PARAMS)
    agent_params.update(param_overrides)
    agent_params["action_space"] = env.action_space
    agent_params["seed"] = seed

    # Handle recovery_window=0: disable recovery by setting thresholds to infinity
    if agent_params.get("recovery_window", 6) == 0:
        agent_params["recovery_window"] = 1  # minimum allowed by agent
        agent_params["recovery_queue_threshold"] = float("inf")
        agent_params["recovery_delay_threshold"] = float("inf")

    agent = NOAAgent(**agent_params)

    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    start_time = time.time()
    while not (terminated or truncated):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.learn(observation, reward=reward, info=info, done=terminated or truncated)
    elapsed = time.time() - start_time

    metrics = _build_metrics(info, seconds, elapsed)
    env.close()
    return metrics


def _format_mean_std(values: list[float], fmt: str = ".2f") -> str:
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:{fmt}} ± {std:{fmt}}"


def _format_sensitivity_table(test_name: str, test_def: dict, results: dict[float, list[dict]]) -> str:
    """Build a markdown-style table for one sensitivity test."""
    param_label = test_def["param_name"]
    headers = [param_label, "Avg Queue (veh)", "Avg Delay (s)", "Throughput (veh)", "Avg Speed (m/s)", "Compute Time (s)"]
    rows = []
    for val in test_def["values"]:
        metrics_list = results[val]
        queues = [m["queue_length"] for m in metrics_list]
        delays = [m["average_delay"] for m in metrics_list]
        throughputs = [m["throughput_total"] for m in metrics_list]
        speeds = [m["mean_speed"] for m in metrics_list]
        times = [m["compute_time"] for m in metrics_list]
        marker = " *" if val == test_def["default"] else ""
        rows.append([
            f"{val}{marker}",
            _format_mean_std(queues),
            _format_mean_std(delays),
            _format_mean_std(throughputs, ".1f"),
            _format_mean_std(speeds),
            _format_mean_std(times, ".1f"),
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    body_lines = ["| " + " | ".join(r[i].ljust(w) for i, w in enumerate(col_widths)) + " |" for r in rows]
    note = "\n(* = default value used in this study)"
    return "\n".join([header_line, sep] + body_lines) + note


def _save_sensitivity_plot(
    test_name: str,
    test_def: dict,
    results: dict[float, list[dict]],
    output_path: str,
) -> None:
    """Generate line plots for one sensitivity test."""
    values = test_def["values"]
    x_label = test_def["x_label"]
    default_val = test_def["default"]

    metrics_keys = [
        ("average_delay", "Avg Delay (s)"),
        ("queue_length", "Avg Queue Length (veh)"),
        ("throughput_total", "Total Throughput (veh)"),
        ("mean_speed", "Avg Speed (m/s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(metrics_keys):
        ax = axes[idx]
        means = [np.mean([m[key] for m in results[v]]) for v in values]
        stds = [np.std([m[key] for m in results[v]]) for v in values]

        ax.errorbar(values, means, yerr=stds, marker="o", capsize=5, linewidth=2, markersize=8, color="#2196F3")
        # Mark the default value
        if default_val in values:
            default_idx = values.index(default_val)
            ax.plot(default_val, means[default_idx], marker="*", markersize=15, color="#F44336", zorder=5,
                    label=f"Default ({default_val})")
            ax.legend(fontsize=10)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.set_xticks(values)

    fig.suptitle(f"NOA Sensitivity Analysis: {test_def['param_name']} (Section 4.4)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NOA Parameter Sensitivity Analysis (Section 4.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--seconds", type=int, default=100000, help="Simulation seconds per run.")
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds per config.")
    parser.add_argument("--route", type=str,
                        default="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
                        help="Route file.")
    parser.add_argument("--min-green", type=int, default=10, help="Minimum green time.")
    parser.add_argument("--max-green", type=int, default=50, help="Maximum green time.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to outputs/sensitivity/.")
    parser.add_argument("--test", type=str, default=None, choices=list(SENSITIVITY_TESTS.keys()),
                        help="Run only one specific test instead of all three.")
    args = parser.parse_args()

    output_dir = args.output_dir or _repo_path("outputs/sensitivity")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tests_to_run = {args.test: SENSITIVITY_TESTS[args.test]} if args.test else SENSITIVITY_TESTS

    all_csv_rows = []

    for test_name, test_def in tests_to_run.items():
        print(f"\n{'='*60}")
        print(f"Sensitivity test: {test_def['param_name']}")
        print(f"  Values: {test_def['values']}")
        print(f"{'='*60}")

        results: dict[float, list[dict]] = {}

        for val in test_def["values"]:
            override = {test_def["override_key"]: val}
            cfg_metrics = []
            print(f"\n  {test_def['override_key']} = {val}")

            for seed in range(args.runs):
                print(f"    Seed {seed}/{args.runs - 1} ...", end=" ", flush=True)
                metrics = _run_single(
                    param_overrides=override,
                    seconds=args.seconds,
                    seed=seed,
                    route=args.route,
                    min_green=args.min_green,
                    max_green=args.max_green,
                )
                cfg_metrics.append(metrics)
                print(f"queue={metrics['queue_length']}, delay={metrics['average_delay']:.2f}s, "
                      f"throughput={metrics['throughput_total']}, speed={metrics['mean_speed']:.2f}m/s, "
                      f"time={metrics['compute_time']:.1f}s")

                all_csv_rows.append({
                    "test": test_name,
                    "param_value": val,
                    "seed": seed,
                    **metrics,
                })

            results[val] = cfg_metrics

        # Print table
        table = _format_sensitivity_table(test_name, test_def, results)
        print(f"\n{table}")

        # Save plot
        plot_path = os.path.join(output_dir, f"sensitivity_{test_name}_{timestamp}.png")
        _save_sensitivity_plot(test_name, test_def, results, plot_path)
        print(f"  Plot: {plot_path}")

    # Save all results to CSV
    csv_path = os.path.join(output_dir, f"sensitivity_all_{timestamp}.csv")
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")
