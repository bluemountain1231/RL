import numpy as np

from agents.replay_buffer import ReplayBuffer


def test_replay_buffer_stores_and_samples_batches():
    buffer = ReplayBuffer(capacity=3)
    for index in range(3):
        buffer.push(
            state=np.array([index], dtype=np.float32),
            action=np.array([index], dtype=np.float32),
            reward=float(index),
            next_state=np.array([index + 1], dtype=np.float32),
            done=bool(index % 2),
        )

    batch = buffer.sample(batch_size=2)

    assert batch["state"].shape == (2, 1)
    assert batch["action"].shape == (2, 1)
    assert batch["reward"].shape == (2,)
    assert batch["next_state"].shape == (2, 1)
    assert batch["done"].shape == (2,)


def test_replay_buffer_samples_scalar_actions_without_losing_batch_dimension():
    buffer = ReplayBuffer(capacity=3)
    for index in range(3):
        buffer.push(
            state=np.array([index], dtype=np.float32),
            action=index,
            reward=float(index),
            next_state=np.array([index + 1], dtype=np.float32),
            done=False,
        )

    batch = buffer.sample(batch_size=2)

    assert batch["action"].shape == (2,)
    assert batch["action"].dtype == np.int64
