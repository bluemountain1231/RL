#!/usr/bin/env python3
"""一键跑通 SUMO + TraCI + sumo-rl（默认无头；--gui 使用 sumo-gui）。

无 DISPLAY 时（常见云主机）若已安装 xvfb-run，会自动用虚拟显示启动 sumo-gui；
本机有桌面时可加 --no-xvfb 禁止该行为。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _maybe_reexec_under_xvfb_for_gui() -> None:
    if "--gui" not in sys.argv:
        return
    if "--no-xvfb" in sys.argv:
        return
    if os.environ.get("DISPLAY"):
        return
    if os.environ.get("_SUMO_RL_XVFB_WRAPPED"):
        return
    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        print(
            "警告: 未设置 DISPLAY 且未找到 xvfb-run；sumo-gui 可能无法启动。"
            "可安装: apt-get install -y xvfb",
            file=sys.stderr,
        )
        return
    script = os.path.abspath(__file__)
    newenv = os.environ.copy()
    newenv["_SUMO_RL_XVFB_WRAPPED"] = "1"
    argv = [xvfb, "-a", sys.executable, script, *sys.argv[1:]]
    os.execve(xvfb, argv, newenv)


def main() -> int:
    if "SUMO_HOME" not in os.environ:
        default = "/usr/share/sumo"
        if os.path.isdir(default):
            os.environ["SUMO_HOME"] = default
        else:
            print("请设置 SUMO_HOME，例如: export SUMO_HOME=/usr/share/sumo", file=sys.stderr)
            return 1

    # 保证能 import sumo_rl（从仓库根运行）
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from sumo_rl import SumoEnvironment

    p = argparse.ArgumentParser(description="SUMO + TraCI + sumo-rl 演示")
    p.add_argument("--gui", action="store_true", help="使用 sumo-gui（use_gui=True）")
    p.add_argument(
        "--no-xvfb",
        action="store_true",
        help="无 DISPLAY 时也不要自动套 xvfb-run（用于本机 X11 等自行提供显示）",
    )
    p.add_argument("--seconds", type=int, default=300, help="仿真时长（秒）")
    p.add_argument("--steps", type=int, default=40, help="环境 step 次数上限")
    p.add_argument(
        "--delay-ms",
        type=int,
        default=200,
        help="GUI 每个仿真 step 的显示延迟（毫秒，仅 --gui 时生效；0 表示不额外减速）",
    )
    args = p.parse_args()

    additional_sumo_cmd = None
    if args.gui and args.delay_ms > 0:
        additional_sumo_cmd = f"--delay {args.delay_ms}"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
        use_gui=args.gui,
        single_agent=True,
        num_seconds=args.seconds,
        sumo_warnings=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    obs, info = env.reset(seed=0)
    print(f"SUMO_HOME={os.environ['SUMO_HOME']}")
    print(f"use_gui={args.gui}  DISPLAY={os.environ.get('DISPLAY', '')!r}")
    print(f"reset OK, obs_dim={getattr(obs, 'shape', [len(obs)])[0] if hasattr(obs, '__len__') else obs}")

    for i in range(args.steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        print(f"step {i + 1:3d}  t={env.sim_step:6.1f}s  reward={r:8.4f}  truncated={trunc}")
        if term or trunc:
            print("episode 结束（时间到或终止）")
            break

    env.close()
    print("TraCI 已关闭，演示结束。")
    return 0


if __name__ == "__main__":
    _maybe_reexec_under_xvfb_for_gui()
    raise SystemExit(main())
