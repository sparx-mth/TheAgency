import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.room_names import majority_name, normalise_name


@pytest.mark.parametrize("text, expected", [("Living Room", "living room"), ("  bedroom. ", "bedroom"), ("the Kitchen", "kitchen"),
                                            ("a bathroom", "bathroom"), ("", ""), (None, "")])
def test_a_room_name_is_lowercase_without_article_or_punctuation(text, expected):
    assert normalise_name(text) == expected


def test_the_majority_name_wins_with_its_share():
    assert majority_name(["Kitchen", "kitchen", "dining room", "kitchen"]) == ("kitchen", 0.75)


def test_a_tie_goes_to_the_name_that_comes_first_alphabetically():
    assert majority_name(["bedroom", "office"])[0] == "bedroom"


def test_empty_answers_are_not_votes():
    assert majority_name(["", None, "hallway"]) == ("hallway", 1.0)
    assert majority_name(["", None]) == ("", 0.0)
