from __future__ import annotations

import numpy as np

from envs.observation_space import FEATURES_PER_LANE


def build_state(
    lane_metrics: dict[str, dict[str, float]],
    lane_order: list[str],
    current_phase: int,
    phase_count: int,
    phase_elapsed: float,
) -> np.ndarray:
    values: list[float] = []
    for lane_id in lane_order:
        metrics = lane_metrics[lane_id]
        for feature_name in FEATURES_PER_LANE:
            values.append(float(metrics[feature_name]))

    phase_one_hot = [0.0] * phase_count
    phase_one_hot[current_phase] = 1.0
    values.extend(phase_one_hot)
    values.append(float(phase_elapsed))
    return np.asarray(values, dtype=np.float32)
