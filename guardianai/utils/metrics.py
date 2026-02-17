from collections import defaultdict


class Metrics:
    """
    Lightweight in-memory metrics store.
    """

    def __init__(self):
        self.counters = defaultdict(int)

    def inc(self, key: str, value: int = 1):
        self.counters[key] += value

    def snapshot(self) -> dict:
        return dict(self.counters)
