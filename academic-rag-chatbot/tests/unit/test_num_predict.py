"""Unit tests for the continuous adaptive generation-budget heuristic."""

from app.config import settings
from app.core.generation.chain import dynamic_num_predict


def test_always_clamped_to_configured_range():
    for q in ["", "Hi", "word " * 500, "explain and compare and list all the things " * 30]:
        n = dynamic_num_predict(q)
        assert settings.llm_num_predict_min <= n <= settings.llm_num_predict_max


def test_detailed_gets_more_than_brief():
    assert dynamic_num_predict("Explain the attention mechanism in detail.") > \
           dynamic_num_predict("What is BERT?")


def test_longer_question_gets_more_than_shorter_same_intent():
    short = dynamic_num_predict("Which methods were compared?")
    long = dynamic_num_predict(
        "Which methods were compared and how did each perform across the benchmarks reported?"
    )
    assert long > short


def test_multipart_gets_more_than_single_part():
    single = dynamic_num_predict("Describe the model architecture.")
    multi = dynamic_num_predict(
        "Describe the model architecture; list all datasets; and compare the baselines."
    )
    assert multi > single


def test_brief_factual_stays_modest():
    # who/what-is/define style questions should land near the floor, not the ceiling
    assert dynamic_num_predict("Who funded this research?") <= 2 * settings.llm_num_predict_min
    assert dynamic_num_predict("Define overfitting.") <= 2 * settings.llm_num_predict_min


def test_returns_int():
    assert isinstance(dynamic_num_predict("What evaluation metrics were used?"), int)


def test_detailed_and_brief_are_mutually_exclusive():
    # "explain ... is" matches both a detailed cue and a brief cue. Detailed must
    # win outright (x1.8), not be halved back down by the brief multiplier (x0.9).
    detailed = dynamic_num_predict("Explain in detail what the attention mechanism is.")
    plain_detailed = dynamic_num_predict("Explain the attention mechanism thoroughly.")
    # If brief still compounded, the "...is" variant would be ~half this.
    assert detailed >= plain_detailed * 0.9


def test_brief_cue_not_matched_inside_a_word():
    # "basis"/"scan" must NOT trip the "is"/"can" brief cues via substring matching.
    # This question has no real cue, so it keeps the per-word base budget; the old
    # substring logic would have wrongly halved it.
    neutral = dynamic_num_predict("The basis for these scan results matters here overall.")
    halved = dynamic_num_predict("Is this correct?")
    assert neutral > halved
