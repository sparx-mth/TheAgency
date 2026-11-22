from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import PERCENT_PROMPTS, STYLES, YES_NO


def test_every_style_is_a_question_about_one_target():
    for style in STYLES:
        assert (PERCENT_PROMPTS.get(style) or YES_NO[style]).count("%s") == 1


def test_the_new_styles_are_offered_and_the_original_percent_is_unchanged():
    assert {"percent", "unseen", "nearby", "keep"} <= set(STYLES)
    assert PERCENT_PROMPTS["percent"].startswith("This photo was taken inside one room of a house. How likely is it that this room contains a %s?")
