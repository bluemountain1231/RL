from envs.action import apply_green_constraints, decode_ddpg_action, decode_dqn_action


def test_decode_dqn_action_maps_index_to_green_delta():
    assert decode_dqn_action(action_index=0, action_delta_seconds=5) == -5
    assert decode_dqn_action(action_index=1, action_delta_seconds=5) == 0
    assert decode_dqn_action(action_index=2, action_delta_seconds=5) == 5


def test_decode_ddpg_action_clips_and_scales_to_green_delta():
    assert decode_ddpg_action(raw_action=-2.0, action_delta_seconds=5) == -5
    assert decode_ddpg_action(raw_action=0.0, action_delta_seconds=5) == 0
    assert decode_ddpg_action(raw_action=0.6, action_delta_seconds=5) == 3
    assert decode_ddpg_action(raw_action=2.0, action_delta_seconds=5) == 5


def test_apply_green_constraints_clamps_to_supported_range():
    assert apply_green_constraints(proposed_green=15, min_green=20, max_green=60) == 20
    assert apply_green_constraints(proposed_green=40, min_green=20, max_green=60) == 40
    assert apply_green_constraints(proposed_green=75, min_green=20, max_green=60) == 60
