"""The start gate on free RAM: the first pending job that fits is chosen, and budgeted while it loads."""
from __future__ import annotations

from pathlib import Path

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.ram_gate import GIB, RamGate, mem_available_bytes
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.shard_pool import PoolJob, run_pool


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def job(key):
    return PoolJob(key=key, command=["true"], log_path=Path("/dev/null"))


def test_the_gate_starts_the_first_pending_scene_that_fits_and_budgets_it_while_it_loads():
    """Free RAM decides, not a job count (six scenes at once took a 30 GB machine into swap): a scene starts when
    the memory available, less the needs of the scenes still loading, less its own, leaves the reserve."""
    clock, free = Clock(), [16.0]
    need = {"large": 6.2, "middling": 5.7, "small": 2.0}
    gate = RamGate(lambda j: need[j.key], reserve_gib=4.5, spacing_s=15.0, settle_s=80.0,
                   available=lambda: free[0] * GIB, clock=clock)
    pending = [job("large"), job("middling"), job("small")]                          # 6.2, 5.7 and 2.0 GiB
    assert gate.choose(pending) == 0 and gate.last_reason is None
    gate.started()
    pending.pop(0)
    assert gate.choose(pending) is None and "spacing" in gate.last_reason
    clock.now = 20.0
    # The machine does not show the first scene yet (16 GiB still "available"): it is budgeted at its need.
    assert gate.choose(pending) == 1, "5.7 GiB does not fit beside the 6.2 still loading; the 2.0 GiB scene does"
    gate.started()
    pending.pop(1)
    clock.now = 40.0
    assert gate.choose(pending) is None and "RAM" in gate.last_reason and "5.7 GiB the smallest" in gate.last_reason
    clock.now, free[0] = 200.0, 7.8                                                  # both loaded, and showing
    assert gate.choose(pending) is None, "7.8 - 5.7 leaves 2.1 GiB, under the reserve"
    free[0] = 10.3
    assert gate.choose(pending) == 0
    with pytest.raises(ValueError):
        RamGate(lambda j: 1.0, reserve_gib=-1.0)
    with pytest.raises(RuntimeError):
        RamGate(lambda j: 1.0).started()
    with pytest.raises(NotImplementedError):
        RamGate(lambda j: 1.0).ready()


def test_the_memory_available_is_read_from_the_kernels_own_line(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\nMemAvailable:   12345678 kB\n")
    assert mem_available_bytes(str(meminfo)) == 12345678 * 1024
    meminfo.write_text("MemTotal:       32000000 kB\n")
    with pytest.raises(RuntimeError, match="no MemAvailable line"):
        mem_available_bytes(str(meminfo))
    assert mem_available_bytes() > 0


class Finished:
    """A process that has already ended."""

    pid = 4242

    def __init__(self, code=0):
        self.code = code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        pass

    def kill(self):
        pass


def test_the_gate_runs_under_the_campaigns_own_pool_out_of_queue_order(tmp_path):
    started, logs, free = [], [], [9.0]
    gate = RamGate(lambda j: {"big": 6.0, "small": 2.0}[j.key], reserve_gib=4.0, spacing_s=0.0, settle_s=0.0,
                   available=lambda: free[0] * GIB)
    jobs = [PoolJob(key="big", command=["run", "big"], log_path=tmp_path / "big.log"),
            PoolJob(key="small", command=["run", "small"], log_path=tmp_path / "small.log")]

    def popen(command, **kwargs):
        started.append(command[1])
        free[0] = 12.0                                    # the small scene ended at once: the big one fits now
        return Finished()

    codes, unstarted = run_pool(jobs, cap=lambda: 2, gate=gate, popen=popen, poll_s=0.0, sleep=lambda s: None,
                                log=logs.append)
    assert started == ["small", "big"], "first fit: the scene behind the head of the queue started first"
    assert codes == {"small": 0, "big": 0} and unstarted == []
