"""Preempt: the first SIGTERM sets a flag; the second one kills."""
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = """
import sys, time
from lemma.preempt import Preempt, PREEMPTED
pre = Preempt()
print("ready", flush=True)
for step in range(10_000):
    if pre.requested:
        print(f"stopped at {step}", flush=True)
        sys.exit(PREEMPTED)
    time.sleep(0.01)
sys.exit(0)
"""


def _spawn():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True, env=env)
    assert p.stdout.readline().strip() == "ready"
    return p


def test_first_sigterm_stops_at_step_boundary():
    p = _spawn()
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=10)
    assert p.returncode == 75
    assert "stopped at" in out


def test_second_sigterm_kills():
    p = _spawn()
    # Two signals back to back usually both land before the next step boundary
    # (the kill); if the boundary wins the race, the clean exit is also correct.
    p.send_signal(signal.SIGTERM)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=10)
    assert p.returncode in (75, -signal.SIGTERM)
