"""Search runtime identity, objective diagnostics and sampled process memory."""
import importlib.metadata
import platform
import threading
from pathlib import Path
import psutil
import numpy as np

def environment():
    lock = Path(__file__).resolve().parents[3] / "requirements-phase-f.lock"
    pins = dict(line.strip().split("==", 1) for line in lock.read_text(encoding="utf-8").splitlines() if "==" in line)
    values = {p: importlib.metadata.version(p) for p in (*pins, "numpy")}
    if any(values[p] != version for p, version in pins.items()):
        raise ValueError("Search environment differs from requirements-phase-f.lock")
    return {"packages": values, "platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version()}

def correlations(candidates):
    valid = [list(vars(c.objectives).values()) for c in candidates if c.feasible]
    matrix = []
    for i in range(4):
        row = []
        for j in range(4):
            if len(valid) < 2:
                row.append({"value": None, "reason": "insufficient_unique_candidates"})
            else:
                a = np.asarray(valid)[:, i]
                b = np.asarray(valid)[:, j]
                if np.ptp(a) == 0 or np.ptp(b) == 0:
                    row.append({"value": None, "reason": "constant_objective"})
                else:
                    row.append({"value": float(np.corrcoef(a, b)[0, 1]), "reason": None})
        matrix.append(row)
    return matrix

class MemoryMonitor:
    def __init__(self, interval=.01):
        self.interval, self.peak = interval, 0
        self.stop = threading.Event()

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
