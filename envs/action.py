from __future__ import annotations


def decode_dqn_action(action_index: int, action_delta_seconds: int) -> int:
    mapping = {
        0: -action_delta_seconds,
        1: 0,
        2: action_delta_seconds,
    }
    return mapping[action_index]


def decode_ddpg_action(raw_action: float, action_delta_seconds: int) -> int:
    clipped = max(-1.0, min(1.0, float(raw_action)))
    scaled = clipped * action_delta_seconds
    return int(scaled + 0.5) if scaled >= 0 else int(scaled - 0.5)


def apply_green_constraints(
    proposed_green: int,
    min_green: int,
    max_green: int,
) -> int:
    return max(min_green, min(max_green, proposed_green))
