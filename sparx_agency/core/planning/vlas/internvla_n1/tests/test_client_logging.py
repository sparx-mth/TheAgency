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


def test_reinit_without_a_prior_init_still_tries():
    session = _Session(good_name="InternVLA-N1")
    client = _client_with(session)
    assert client._reinit() is True
    assert session.init_names == ["InternVLA-N1"]
