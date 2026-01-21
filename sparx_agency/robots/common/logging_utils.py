import time
import numpy as np

class TicToc:
    def __init__(self, name="Task", logger=None):
        self.name = name
        self.logger = logger
        self.t_start = None

    def __enter__(self):
        self.t_start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        elapsed_ms = (time.perf_counter() - self.t_start) * 1000
        msg = f"[{self.name}] {elapsed_ms:.2f} ms"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

def log_distribution(logger, label, data):
    """
    Counts unique values in a numpy array and logs them cleanly.
    Useful for: Grid values distribution: {-1: 111205, 20: 270, 100: 81}
    """
    unique, counts = np.unique(data, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"{label} distribution: {dist}")

def throttle_log(logger, msg, last_log_time, interval=1.0):
    """Logs only if 'interval' seconds have passed since last_log_time."""
    current_time = time.time()
    if current_time - last_log_time > interval:
        logger.info(msg)
        return current_time
    return last_log_time