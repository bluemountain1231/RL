"""Ablation experiment for NOA core mechanisms (Section 4.2.3).

Compares four configurations over 3 random seeds each:
  1. Full NOA (all mechanisms enabled)
  2. No Cache  (cache_weight = 0)
  3. No Recovery (recovery thresholds set to infinity so recovery never triggers)
  4. No Elite (elite_size = 1, minimal elite guidance)

Usage:
  python experiments/noa_ablation.py -s 100000
  python experiments/noa_ablation.py -s 100000 --runs 3
"""

from __future__ import annotations

import argparse
import os
import sys
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


# ── Ablation configurations ──────────────────────────────────────────────────
ABLATION_CONFIGS = {
    "Full NOA": {},
    "No Cache": {
        "cache_weight": 0.0,
    },
    "No Recovery": {
        # Setting thresholds to infinity ensures recovery never triggers
        "recovery_queue_threshold": float("inf"),
        "recovery_delay_threshold": float("inf"),
    },
    "No Elite": {
        "elite_size": 1,
    },
}


def _build_metrics(info: dict, seconds: int) -> dict:
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
    }


def _run_single(
    config_name: str,
    agent_kwargs: dict,
    seconds: int,
    seed: int,
    route: str,
    min_green: int,
    max_green: int,
) -> dict:
    """Run a single simulation with the given config and seed."""
    env = SumoEnvironment(
        net_file=_repo_path("sumo_rl/nets/single-intersection/single-intersection.net.xml"),
        route_file=_repo_path(route),
        out_csv_name=None,  # no CSV output for ablation
        use_gui=False,
        num_seconds=seconds,
        min_green=min_green,
        max_green=max_green,
        single_agent=True,
        observation_class=NOAObservationFunction,
        reward_fn="diff-waiting-time",
    )

    default_kwargs = dict(
        action_space=env.action_space,
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
        seed=seed,
    )
    default_kwargs.update(agent_kwargs)

    agent = NOAAgent(**default_kwargs)

    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.learn(observation, reward=reward, info=info, done=terminated or truncated)

    metrics = _build_metrics(info, seconds)
    env.close()
    return metrics


def _format_mean_std(values: list[float]) -> str:
    """Format as 'mean ± std' string."""
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.2f} ± {std:.2f}"


def _format_table(results: dict[str, list[dict]]) -> str:
    """Build a markdown-style summary table."""
    headers = ["Config", "Avg Queue (veh)", "Avg Delay (s)", "Total Throughput (veh)", "Avg Speed (m/s)"]
    rows = []
    for cfg_name, metrics_list in results.items():
        queues = [m["queue_length"] for m in metrics_list]
        delays = [m["average_delay"] for m in metrics_list]
        throughputs = [m["throughput_total"] for m in metrics_list]
        speeds = [m["mean_speed"] for m in metrics_list]
        rows.append([
            cfg_name,
            _format_mean_std(queues),
            _format_mean_std(delays),
            _format_mean_std(throughputs),
            _format_mean_std(speeds),
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    body_lines = ["| " + " | ".join(r[i].ljust(w) for i, w in enumerate(col_widths)) + " |" for r in rows]

    return "\n".join([header_line, sep] + body_lines)


def _save_comparison_plot(results: dict[str, list[dict]], output_path: str) -> None:
    """Generate bar chart comparing ablation configurations."""
    configs = list(results.keys())
    metrics_keys = [
        ("queue_length", "Avg Queue Length (veh)"),
        ("average_delay", "Avg Delay (s)"),
        ("throughput_total", "Total Throughput (veh)"),
        ("mean_speed", "Avg Speed (m/s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]

    for idx, (key, label) in enumerate(metrics_keys):
        ax = axes[idx]
        means = [np.mean([m[key] for m in results[cfg]]) for cfg in configs]
        stds = [np.std([m[key] for m in results[cfg]]) for cfg in configs]
        bars = ax.bar(configs, means, yerr=stds, capsize=5, color=colors[:len(configs)], alpha=0.85)
        ax.set_title(label, fontsize=13)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=15)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(means),
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("NOA Ablation Experiment (Section 4.2.3)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NOA Ablation Experiment (Section 4.2.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--seconds", type=int, default=100000, help="Simulation seconds per run.")
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds (runs) per config.")
    parser.add_argument("--route", type=str,
                        default="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
                        help="Route file.")
    parser.add_argument("--min-green", type=int, default=10, help="Minimum green time.")
    parser.add_argument("--max-green", type=int, default=50, help="Maximum green time.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results. Defaults to outputs/ablation/.")
    args = parser.parse_args()

    output_dir = args.output_dir or _repo_path("outputs/ablation")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: dict[str, list[dict]] = {}

    for cfg_name, overrides in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running ablation: {cfg_name}")
        print(f"  Overrides: {overrides if overrides else 'None (full NOA)'}")
        print(f"{'='*60}")

        cfg_metrics = []
        for seed in range(args.runs):
            print(f"  Seed {seed}/{args.runs - 1} ...", end=" ", flush=True)
            metrics = _run_single(
                config_name=cfg_name,
                agent_kwargs=overrides,
                seconds=args.seconds,
                seed=seed,
                route=args.route,
                min_green=args.min_green,
                max_green=args.max_green,
            )
            cfg_metrics.append(metrics)
            print(f"queue={metrics['queue_length']}, delay={metrics['average_delay']:.2f}s, "
                  f"throughput={metrics['throughput_total']}, speed={metrics['mean_speed']:.2f}m/s")

        all_results[cfg_name] = cfg_metrics

    # ── Output results ───────────────────────────────────────────────────────
    table = _format_table(all_results)
    print(f"\n{'='*60}")
    print("Ablation Experiment Results (Section 4.2.3)")
    print(f"{'='*60}")
    print(table)

    # Save table to text file
    table_path = os.path.join(output_dir, f"ablation_results_{timestamp}.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("NOA Ablation Experiment Results (Section 4.2.3)\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Simulation seconds: {args.seconds}\n")
        f.write(f"Runs per config: {args.runs}\n\n")
        f.write(table + "\n")

    # Save raw metrics to CSV
    csv_path = os.path.join(output_dir, f"ablation_results_{timestamp}.csv")
    rows = []
    for cfg_name, metrics_list in all_results.items():
        for seed_idx, m in enumerate(metrics_list):
            rows.append({"config": cfg_name, "seed": seed_idx, **m})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Save comparison plot
    plot_path = os.path.join(output_dir, f"ablation_comparison_{timestamp}.png")
    _save_comparison_plot(all_results, plot_path)

    print(f"\nResults saved to:")
    print(f"  Table: {table_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Plot:  {plot_path}")
