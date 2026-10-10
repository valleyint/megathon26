"""Timer utility for latency profiling across pipeline stages."""

import time
from contextlib import contextmanager
from typing import Generator, Dict


class PipelineTimer:
    """Tracks latency across multiple named pipeline steps."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, step_name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings[step_name] = elapsed_ms

    def get_summary(self) -> Dict[str, float]:
        summary = dict(self.timings)
        summary["total_ms"] = sum(self.timings.values())
        return summary
