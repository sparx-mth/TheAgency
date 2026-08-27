"""``_say`` dispatches each severity to its own logger method.

Two bugs live here if it does not, and both were real. ``rclpy`` caches a
severity per logging *call site*, so one shared ``getattr(logger, level)(msg)``
line raises ``ValueError: Logger severity cannot be changed between calls`` the
first time the client logs at a second severity -- out of a timer callback,
killing the node, on the first dropped request of a flight. And ``rclpy``'s
logger has ``warning``, not ``warn``, so the same ``getattr`` quietly demoted
every warning to info.

The call-site caching itself needs a real ``rclpy`` logger to observe and is not
reproducible here; what this file pins is the dispatch, which is what fixes it.
"""
import threading
import time

import pytest

from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient


class _Logger:
    """Records which method each call landed on -- and has no ``warn``, like rclpy."""

    def __init__(self):
        self.calls = []

    def debug(self, msg):
        self.calls.append(("debug", msg))

    def info(self, msg):
        self.calls.append(("info", msg))

    def warning(self, msg):
        self.calls.append(("warning", msg))

    def error(self, msg):
        self.calls.append(("error", msg))


@pytest.mark.parametrize("level, expected", [
    ("info", "info"),
    ("error", "error"),
    ("debug", "debug"),
    ("warn", "warning"),        # the alias rclpy does not have
    ("warning", "warning"),
    ("nonsense", "info"),       # unknown severity still says something
])
def test_each_severity_reaches_its_own_logger_method(level, expected):
    logger = _Logger()
    ModelClient(logger=logger)._say(level, "hello")
    assert logger.calls == [(expected, "hello")]


def test_mixed_severities_all_get_through():
    logger = _Logger()
    client = ModelClient(logger=logger)
    for level in ("info", "warn", "error", "debug", "info"):
        client._say(level, level)
    assert [c[0] for c in logger.calls] == ["info", "warning", "error", "debug", "info"]


def test_without_a_logger_it_prints_instead_of_raising(capsys):
    ModelClient()._say("error", "no logger here")
    assert "[ERROR] no logger here" in capsys.readouterr().out


# ── re-init replays the arguments it was given ───────────────────────────
class _Resp:
    def __init__(self, code, text=""):
        self.status_code = code
        self.text = text

    def json(self):
        return {"agent_name": "internvla_n1"}


class _Session:
    """Records every POST; /agent/init succeeds only for the right model name."""

    def __init__(self, good_name):
        self.good_name = good_name
        self.init_names = []

    def post(self, url, json=None, timeout=None, headers=None):
        if url.endswith("/agent/init"):
            name = json["agent_config"]["model_name"]
            self.init_names.append(name)
            return _Resp(201 if name == self.good_name else 500, "KeyError")
        return _Resp(404)      # the agent does not exist yet

    def get(self, url, timeout=None):
        return _Resp(200)


def _client_with(session):
    client = ModelClient()
    client._session = session
    return client


def test_step_reinitialises_with_the_name_it_was_given_not_the_default():
    # InternNav registers `internvla_n1`; the signature default is `InternVLA-N1`,
    # which is a 500 forever. A flight where reset() lost the race to a loading
    # server then retried under the wrong name on every frame.
    session = _Session(good_name="internvla_n1")
    client = _client_with(session)

    assert client.init_agent(model_name="internvla_n1") is True
    client.initialized = False          # the server lost its agent

    import numpy as np
    client.step(np.zeros((4, 4, 3), np.uint8), "go")

    assert session.init_names == ["internvla_n1", "internvla_n1"]


def test_reinit_without_a_prior_init_asks_for_the_name_that_exists():
    # There is nothing to replay before the first successful init, so this falls
    # through to the signature default -- which therefore has to be the name
    # InternNav registers. `InternVLA-N1` is a server-side KeyError, so the old
    # default made every cold-start step a 500 until reset() happened to win.
    session = _Session(good_name="internvla_n1")
    client = _client_with(session)
    assert client._reinit() is True
    assert session.init_names == ["internvla_n1"]


# ── one checkpoint load at a time ────────────────────────────────────────
class _SlowLoadingSession:
    """A server whose first /agent/init takes a while, like a real one.

    It also refuses to hold two agents at once -- which is what an 8 GB card
    does when a 7B checkpoint is asked for twice: the second load OOMs and the
    server stops answering anything at all.
    """

    def __init__(self, load_s=0.2):
        self.load_s = load_s
        self.init_calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self.agent_exists = False

    def post(self, url, json=None, timeout=None, headers=None):
        if url.endswith("/agent/init"):
            self.init_calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            try:
                time.sleep(self.load_s)
                if self.agent_exists:
                    return _Resp(500, "torch.OutOfMemoryError")
                self.agent_exists = True
                return _Resp(201)
            finally:
                self.concurrent -= 1
        if url.endswith("/reset"):
            return _Resp(200 if self.agent_exists else 404)
        return _Resp(404)

    def get(self, url, timeout=None):
        return _Resp(200)


def test_two_threads_asking_to_init_load_the_checkpoint_once():
    """The failure this prevents cost a whole five-minute recording.

    The policy node initialises the agent at start-up AND re-initialises from
    any step that finds the client uninitialised, on a different thread. Both
    reaching a cold server means the second one asks for a second copy of the
    checkpoint while the first is still loading; on an 8 GB card that is a CUDA
    OOM twenty-seven seconds in, after which the server answers nothing and the
    aircraft sits still for the rest of the flight.
    """
    session = _SlowLoadingSession()
    client = _client_with(session)
    results = []

    def go():
        results.append(client.init_agent(model_name="internvla_n1",
                                         model_settings={"width": 600}))

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results), "every caller should end up initialised"
    assert session.max_concurrent == 1, "two inits overlapped"
    assert session.init_calls == 1, "the checkpoint was asked for more than once"


def test_the_client_does_not_warn_about_settings_it_applied_itself():
    """The "NOT applied" warning is for a stale server, not for our own wait.

    A second caller of a serialised init finds the agent the FIRST caller just
    created with these very settings; telling the operator they were ignored
    would send them off restarting a correctly configured server.
    """
    logger = _Logger()
    client = ModelClient(logger=logger)
    client._session = _SlowLoadingSession(load_s=0.0)

    assert client.init_agent(model_name="internvla_n1", model_settings={"width": 600})
    assert client.init_agent(model_name="internvla_n1", model_settings={"width": 600})
    assert not [c for c in logger.calls if c[0] == "warning"], logger.calls
    # ...and the logger really is wired, or the line above proves nothing.
    assert any("already initialized" in c[1] for c in logger.calls), logger.calls


def test_it_DOES_warn_when_someone_else_created_the_agent():
    """The stale-server case the warning exists for: settings silently ignored."""
    logger = _Logger()
    client = ModelClient(logger=logger)
    session = _SlowLoadingSession(load_s=0.0)
    session.agent_exists = True          # a server that was up before we started
    client._session = session

    assert client.init_agent(model_name="internvla_n1", model_settings={"width": 600})
    assert session.init_calls == 0, "an existing agent must not be re-created"
    assert [c for c in logger.calls if c[0] == "warning"], logger.calls
