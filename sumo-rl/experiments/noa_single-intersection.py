import argparse
import glob
import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


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


def _repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


def _build_metrics(info: dict, seconds: int, total_reward: float) -> dict:
    throughput_total = info.get("system_total_arrived", 0)
    throughput_per_second = throughput_total / max(1, seconds)
    queue_length = info.get("agents_total_stopped", info.get("system_total_stopped", 0))
    avg_delay = info.get("system_mean_waiting_time", 0.0)
    total_delay = info.get("system_total_waiting_time", 0.0)
    mean_speed = info.get("system_mean_speed", 0.0)
    backlog = info.get("system_total_backlogged", 0)
    running = info.get("system_total_running", 0)
    teleported = info.get("system_total_teleported", 0)
    departed = info.get("system_total_departed", 0)

    return {
        "total_reward": total_reward,
        "queue_length": queue_length,
        "average_delay": avg_delay,
        "total_delay": total_delay,
        "throughput_total": throughput_total,
        "throughput_per_second": throughput_per_second,
        "mean_speed": mean_speed,
        "backlog": backlog,
        "running": running,
        "departed": departed,
        "teleported": teleported,
    }


def _evaluate_metrics(metrics: dict) -> str:
    notes = []

    if metrics["queue_length"] <= 5:
        notes.append("排队长度很低")
    elif metrics["queue_length"] <= 15:
        notes.append("排队长度中等")
    else:
        notes.append("排队长度偏高")

    if metrics["average_delay"] <= 5:
        notes.append("平均延误较低")
    elif metrics["average_delay"] <= 15:
        notes.append("平均延误中等")
    else:
        notes.append("平均延误偏高")

    if metrics["throughput_per_second"] >= 0.25:
        notes.append("通行能力较好")
    elif metrics["throughput_per_second"] >= 0.12:
        notes.append("通行能力一般")
    else:
        notes.append("通行能力偏弱")

    if metrics["teleported"] > 0:
        notes.append("出现车辆 teleport，说明有拥堵风险")
    if metrics["backlog"] > 0:
        notes.append("仍有待发车辆积压")

    return "；".join(notes)


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if window <= 1:
        return numeric
    return numeric.rolling(window=window, min_periods=1, center=True).mean()


def _incremental_throughput(df: pd.DataFrame) -> pd.Series:
    cumulative_col = None
    if "system_total_arrived" in df.columns:
        cumulative_col = "system_total_arrived"
    elif "system_total_departed" in df.columns:
        cumulative_col = "system_total_departed"

    if cumulative_col is None:
        return pd.Series([0.0] * len(df), index=df.index, dtype=float)

    cumulative = pd.to_numeric(df[cumulative_col], errors="coerce")
    step = pd.to_numeric(df["step"], errors="coerce")
    delta_count = cumulative.diff().fillna(0).clip(lower=0)
    delta_step = step.diff().replace(0, pd.NA)
    throughput = (delta_count / delta_step).fillna(0)
    return throughput


def _prepare_plot_df(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    prepared = df.copy()
    prepared["queue_smoothed"] = _moving_average(prepared["agents_total_stopped"], smooth_window)
    prepared["waiting_smoothed"] = _moving_average(prepared["system_total_waiting_time"], smooth_window)
    prepared["delay_smoothed"] = _moving_average(prepared["system_mean_waiting_time"], smooth_window)
    prepared["speed_smoothed"] = _moving_average(prepared["system_mean_speed"], smooth_window)
    prepared["throughput_increment"] = _incremental_throughput(prepared)
    prepared["throughput_increment_smoothed"] = _moving_average(prepared["throughput_increment"], smooth_window)
    return prepared


def _plot_line(ax, x: pd.Series, y: pd.Series, title: str, ylabel: str, label: Optional[str] = None) -> None:
    ax.plot(x, y, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time Step (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if label is not None:
        ax.legend()


def _align_plot_range(noa_df: pd.DataFrame, ql_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    noa_max_step = pd.to_numeric(noa_df["step"], errors="coerce").max()
    ql_aligned = ql_df[pd.to_numeric(ql_df["step"], errors="coerce") <= noa_max_step].copy()
    return noa_df, ql_aligned


def _save_result_plot(csv_path: str, image_path: str, smooth_window: int) -> None:
    df = _prepare_plot_df(pd.read_csv(csv_path), smooth_window)
    x = pd.to_numeric(df["step"], errors="coerce")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    _plot_line(axes[0], x, df["waiting_smoothed"], "Smoothed Total Waiting Time", "Waiting Time (s)")
    _plot_line(axes[1], x, df["queue_smoothed"], "Smoothed Queue Length", "Queued Vehicles")
    _plot_line(axes[2], x, df["delay_smoothed"], "Smoothed Average Delay", "Delay (s)")
    _plot_line(
        axes[3],
        x,
        df["throughput_increment_smoothed"],
        "Incremental Throughput",
        "Vehicles per Second",
    )

    fig.suptitle("NOA Traffic Signal Control Metrics", fontsize=18)
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _auto_find_ql_csv() -> Optional[str]:
    matches = glob.glob(_repo_path("outputs/single-intersection/*_alpha*_gamma*_eps*_decay*_conn*_ep*.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _save_comparison_plot(noa_csv_path: str, ql_csv_path: str, image_path: str, smooth_window: int) -> None:
    noa_df = _prepare_plot_df(pd.read_csv(noa_csv_path), smooth_window)
    ql_df = _prepare_plot_df(pd.read_csv(ql_csv_path), smooth_window)
    noa_df, ql_df = _align_plot_range(noa_df, ql_df)

    noa_x = pd.to_numeric(noa_df["step"], errors="coerce")
    ql_x = pd.to_numeric(ql_df["step"], errors="coerce")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(ql_x, ql_df["waiting_smoothed"], linewidth=1.8, color="tab:blue", alpha=0.9, label="QL", zorder=2)
    axes[0].plot(noa_x, noa_df["waiting_smoothed"], linewidth=2.8, color="tab:red", alpha=0.95, label="NOA", zorder=3)
    axes[0].set_title("Smoothed Total Waiting Time")
    axes[0].set_xlabel("Time Step (s)")
    axes[0].set_ylabel("Waiting Time (s)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    axes[1].plot(ql_x, ql_df["queue_smoothed"], linewidth=1.8, color="tab:blue", alpha=0.9, label="QL", zorder=2)
    axes[1].plot(noa_x, noa_df["queue_smoothed"], linewidth=2.8, color="tab:red", alpha=0.95, label="NOA", zorder=3)
    axes[1].set_title("Smoothed Queue Length")
    axes[1].set_xlabel("Time Step (s)")
    axes[1].set_ylabel("Queued Vehicles")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    axes[2].plot(ql_x, ql_df["delay_smoothed"], linewidth=1.8, color="tab:blue", alpha=0.9, label="QL", zorder=2)
    axes[2].plot(noa_x, noa_df["delay_smoothed"], linewidth=2.8, color="tab:red", alpha=0.95, label="NOA", zorder=3)
    axes[2].set_title("Smoothed Average Delay")
    axes[2].set_xlabel("Time Step (s)")
    axes[2].set_ylabel("Delay (s)")
    axes[2].set_ylim(bottom=0)
    axes[2].legend()

    if "system_total_arrived" in ql_df.columns or "system_total_departed" in ql_df.columns:
        axes[3].plot(
            ql_x,
            ql_df["throughput_increment_smoothed"],
            linewidth=1.8,
            color="tab:blue",
            alpha=0.9,
            label="QL",
            zorder=2,
        )
        axes[3].plot(
            noa_x,
            noa_df["throughput_increment_smoothed"],
            linewidth=2.8,
            color="tab:red",
            alpha=0.95,
            label="NOA",
            zorder=3,
        )
        axes[3].set_title("Incremental Throughput")
        axes[3].set_ylabel("Vehicles per Second")
    else:
        axes[3].plot(ql_x, ql_df["speed_smoothed"], linewidth=1.8, color="tab:blue", alpha=0.9, label="QL", zorder=2)
        axes[3].plot(noa_x, noa_df["speed_smoothed"], linewidth=2.8, color="tab:red", alpha=0.95, label="NOA", zorder=3)
        axes[3].set_title("Smoothed Mean Speed")
        axes[3].set_ylabel("Speed (m/s)")
    axes[3].set_xlabel("Time Step (s)")
    axes[3].set_ylim(bottom=0)
    axes[3].legend()

    fig.suptitle("NOA vs QL Traffic Signal Control Comparison", fontsize=18)
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _resolve_plot_path(path: Optional[str], fallback: str, run: int, multi_run: bool) -> str:
    if path is None:
        resolved = fallback
    else:
        resolved = _repo_path(path)

    if not multi_run:
        return resolved

    root, ext = os.path.splitext(resolved)
    ext = ext or ".png"
    return f"{root}_run{run}{ext}"


from sumo_rl import SumoEnvironment
from sumo_rl.agents import NOAAgent
from sumo_rl.environment.observations import NOAObservationFunction


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Nutcracker Optimizer Single-Intersection""",
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("--forever", action="store_true", default=False, help="Keep launching new episodes until manually stopped.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("--population-size", type=int, default=12, help="NOA candidate population size.\n")
    prs.add_argument("--elite-size", type=int, default=4, help="Number of elite candidates to retain.\n")
    prs.add_argument("--cache-size", type=int, default=256, help="State-action cache capacity.\n")
    prs.add_argument("--alpha", type=float, default=0.6, help="Search attraction coefficient.\n")
    prs.add_argument("--beta", type=float, default=0.3, help="Search differential coefficient.\n")
    prs.add_argument("--sigma", type=float, default=0.15, help="Gaussian perturbation scale.\n")
    prs.add_argument("--recovery-window", type=int, default=6, help="Recovery duration in decision steps.\n")
    prs.add_argument("--recovery-queue-threshold", type=float, default=0.75, help="Queue threshold for triggering recovery.\n")
    prs.add_argument("--recovery-delay-threshold", type=float, default=0.7, help="Delay threshold for triggering recovery.\n")
    prs.add_argument("--cache-weight", type=float, default=0.35, help="Weight of cache reuse in phase scoring.\n")
    prs.add_argument("--w-queue", type=float, default=0.35, help="Weight for phase demand / queue term.\n")
    prs.add_argument("--w-delay", type=float, default=0.3, help="Weight for phase delay term.\n")
    prs.add_argument("--w-pressure", type=float, default=0.15, help="Weight for pressure term.\n")
    prs.add_argument("--w-throughput", type=float, default=0.2, help="Weight for throughput proxy term.\n")
    prs.add_argument("--w-switch", type=float, default=0.08, help="Penalty for keeping the same action under search.\n")
    prs.add_argument("--fitness-w-queue", type=float, default=0.4, help="Weight for queue term in realized fitness.\n")
    prs.add_argument("--fitness-w-delay", type=float, default=0.3, help="Weight for delay term in realized fitness.\n")
    prs.add_argument("--fitness-w-speed", type=float, default=0.15, help="Weight for speed term in realized fitness.\n")
    prs.add_argument("--fitness-w-throughput", type=float, default=0.15, help="Weight for throughput term in realized fitness.\n")
    prs.add_argument("--plot-output", type=str, default=None, help="Optional PNG output path for the metrics plot.\n")
    prs.add_argument("--compare-output", type=str, default=None, help="Optional PNG output path for the NOA vs QL comparison plot.\n")
    prs.add_argument("--ql-csv", type=str, default=None, help="QL CSV path used for comparison plotting.\n")
    prs.add_argument("--smooth-window", type=int, default=9, help="Moving average window for waiting, queue, delay and throughput curves.\n")
    args = prs.parse_args()

    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = _repo_path(
        f"outputs/single-intersection/{experiment_time}_noa_pop{args.population_size}"
        f"_alpha{args.alpha}_beta{args.beta}_sigma{args.sigma}"
    )

    multi_run = args.forever or args.runs > 1
    ql_csv_for_compare = _repo_path(args.ql_csv) if args.ql_csv is not None else _auto_find_ql_csv()

    run = 1
    while args.forever or run <= args.runs:
        env = SumoEnvironment(
            net_file=_repo_path("sumo_rl/nets/single-intersection/single-intersection.net.xml"),
            route_file=_repo_path(args.route),
            out_csv_name=out_csv,
            use_gui=args.gui,
            num_seconds=args.seconds,
            min_green=args.min_green,
            max_green=args.max_green,
            single_agent=True,
            observation_class=NOAObservationFunction,
            reward_fn="diff-waiting-time",
        )

        agent = NOAAgent(
            action_space=env.action_space,
            population_size=args.population_size,
            elite_size=args.elite_size,
            cache_size=args.cache_size,
            alpha=args.alpha,
            beta=args.beta,
            sigma=args.sigma,
            recovery_window=args.recovery_window,
            recovery_queue_threshold=args.recovery_queue_threshold,
            recovery_delay_threshold=args.recovery_delay_threshold,
            cache_weight=args.cache_weight,
            switch_penalty=args.w_switch,
            weight_queue=args.w_queue,
            weight_delay=args.w_delay,
            weight_pressure=args.w_pressure,
            weight_throughput=args.w_throughput,
            fitness_weight_queue=args.fitness_w_queue,
            fitness_weight_delay=args.fitness_w_delay,
            fitness_weight_speed=args.fitness_w_speed,
            fitness_weight_throughput=args.fitness_w_throughput,
            seed=run - 1,
        )

        observation, info = env.reset(seed=run - 1)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            agent.learn(observation, reward=reward, info=info, done=terminated or truncated)

        env.save_csv(out_csv, run)
        csv_path = f"{out_csv}_conn{env.label}_ep{run}.csv"
        plot_path = _resolve_plot_path(args.plot_output, f"{out_csv}_conn{env.label}_ep{run}.png", run, multi_run)
        _save_result_plot(csv_path, plot_path, args.smooth_window)

        compare_path = None
        if ql_csv_for_compare is not None and os.path.exists(ql_csv_for_compare):
            compare_path = _resolve_plot_path(
                args.compare_output,
                f"{out_csv}_conn{env.label}_ep{run}_compare_ql.png",
                run,
                multi_run,
            )
            _save_comparison_plot(csv_path, ql_csv_for_compare, compare_path, args.smooth_window)

        metrics = _build_metrics(info, args.seconds, total_reward)
        assessment = _evaluate_metrics(metrics)
        env.close()
        print(f"run={run}")
        print(f"  total_reward: {metrics['total_reward']:.4f}")
        print(f"  车辆排队长度(queue_length): {metrics['queue_length']}")
        print(f"  平均延误时间(average_delay): {metrics['average_delay']:.2f}s")
        print(f"  总延误时间(total_delay): {metrics['total_delay']:.2f}s")
        print(f"  通行能力(throughput_total): {metrics['throughput_total']}")
        print(f"  单位时间通行能力(throughput_per_second): {metrics['throughput_per_second']:.4f} veh/s")
        print(f"  平均速度(mean_speed): {metrics['mean_speed']:.4f} m/s")
        print(f"  运行中车辆(running): {metrics['running']}")
        print(f"  已发车(departed): {metrics['departed']}")
        print(f"  待发积压(backlog): {metrics['backlog']}")
        print(f"  瞬移车辆(teleported): {metrics['teleported']}")
        print(f"  评估: {assessment}")
        print(f"  NOA结果图: {plot_path}")
        if compare_path is not None:
            print(f"  NOA vs QL 对比图: {compare_path}")
        elif args.ql_csv is not None or ql_csv_for_compare is None:
            print("  NOA vs QL 对比图: 未生成（未找到可用的 QL CSV）")
        run += 1
