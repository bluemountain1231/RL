from __future__ import annotations


class FixedTimeBaselinePolicy:
    def __init__(self, schedule: list[int]) -> None:
        self.schedule = schedule

    def act(self, step_index: int) -> int:
        return self.schedule[step_index % len(self.schedule)]
