"""SIGTERM as a request to stop at the next step boundary, not mid-step.

homeserv's `gpu-run` sends SIGTERM when a recording starts on the box (the GPU
goes to the interactive user). The default action would kill the process and
cost every step since the last rolling checkpoint; instead the trainer saves
its rolling checkpoint at the next step boundary and exits PREEMPTED.

Weights only: the trainers have no resume path yet (no optimizer state or step
is saved), so a preempted run is restarted, and the checkpoint is a salvage.
"""
from __future__ import annotations

import signal

# EX_TEMPFAIL: "try again later", the code gpu-run and worker-drain read as busy, not failed.
PREEMPTED = 75


class Preempt:
    """Install with `Preempt()` before the loop; test `if pre.requested:` once per step."""

    def __init__(self, signum: int = signal.SIGTERM):
        self.requested = False
        signal.signal(signum, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        if self.requested:  # a second SIGTERM means "now": restore the default and re-raise
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        self.requested = True
        print("preempt requested: stopping at the next step boundary", flush=True)
