from envs.reward import compute_reward


def test_compute_reward_combines_waiting_queue_throughput_and_fairness():
    reward = compute_reward(
        total_waiting=10.0,
        total_queue=4.0,
        throughput=3.0,
        directional_waiting_times=[2.0, 4.0, 6.0, 8.0],
        waiting_weight=1.0,
        queue_weight=0.5,
        throughput_weight=0.2,
        fairness_weight=0.1,
    )

    assert round(reward, 2) == -11.10
