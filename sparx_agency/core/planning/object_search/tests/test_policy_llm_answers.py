"""The policy keys its LLM client's answer store by the episode it is running, and counts the episode's answers."""
from sparx_agency.core.planning.object_search.tests.fixtures import FakeLLM, make_policy


class ScopedLLM(FakeLLM):
    """A client with an answer store's two hooks."""

    def __init__(self):
        super().__init__()
        self.scopes, self.stats = [], {"replayed": 0, "repeated": 0, "asked": 0, "recorded": 0}

    def set_answer_scope(self, scope):
        self.scopes.append(scope)

    def answer_stats(self):
        return dict(self.stats)


def test_reset_scopes_the_store_to_the_episode_and_the_record_counts_only_this_episodes_answers():
    llm = ScopedLLM()
    llm.stats["asked"] = 5                                   # answers of earlier episodes
    policy, ep = make_policy(llm=llm)
    assert llm.scopes == [ep.episode_id]
    llm.stats["asked"], llm.stats["replayed"] = 8, 2
    assert policy.episode_info()["llm_answers"] == {"replayed": 2, "repeated": 0, "asked": 3, "recorded": 0}


def test_a_client_without_a_store_adds_nothing_to_the_record():
    policy, _ = make_policy(llm=FakeLLM())
    assert "llm_answers" not in policy.episode_info()
