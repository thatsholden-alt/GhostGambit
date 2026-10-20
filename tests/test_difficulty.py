"""Tests for the difficulty ladder.

Runs on a bare laptop -- no engine, no hardware, nothing installed.
    python3 -m pytest tests/
"""

import pytest

from game import difficulty


def test_ladder_has_at_least_ten_levels():
    # The requirement says "10+ difficulty levels". If someone trims the ladder
    # to make a demo shorter, this is the test that stops it reaching the panel.
    assert len(difficulty.LEVELS) >= 10


def test_levels_are_numbered_one_upwards_with_no_gaps():
    numbers = [row["level"] for row in difficulty.LEVELS]
    assert numbers == list(range(1, len(difficulty.LEVELS) + 1))


def test_ladder_is_sorted_by_strength():
    # This is the whole point of the module. The previous ladder failed exactly
    # here: level 3 was stronger than level 4 and nobody noticed until we
    # measured it over 40 games.
    elos = [row["elo"] for row in difficulty.LEVELS]
    assert elos == sorted(elos)
    assert len(set(elos)) == len(elos), "two levels share an Elo, so they are the same level"


def test_every_level_switches_limit_strength_on():
    # UCI_Elo does nothing unless UCI_LimitStrength is true. Forgetting it makes
    # every level play at full strength while looking correct in the code.
    for row in difficulty.LEVELS:
        options = difficulty.uci_options_for(row["level"])
        assert options["UCI_LimitStrength"] is True


def test_options_carry_the_elo_for_that_level():
    assert difficulty.uci_options_for(1)["UCI_Elo"] == 500
    assert difficulty.uci_options_for(10)["UCI_Elo"] == 2850


def test_a_level_that_does_not_exist_raises():
    # Raising rather than clamping: a bad level here means the caller is broken.
    for bad in (0, 11, -3, 99):
        with pytest.raises(ValueError):
            difficulty.uci_options_for(bad)


def test_clamp_level_pulls_stray_knob_readings_into_range():
    # The knob is analog, so out-of-range readings are expected noise, not bugs.
    assert difficulty.clamp_level(0) == difficulty.MIN_LEVEL
    assert difficulty.clamp_level(-5) == difficulty.MIN_LEVEL
    assert difficulty.clamp_level(11) == difficulty.MAX_LEVEL
    assert difficulty.clamp_level(500) == difficulty.MAX_LEVEL
    assert difficulty.clamp_level(4) == 4


def test_clamped_levels_are_always_usable():
    # Whatever the ADC hands us, the result must be a level the engine accepts.
    for raw in range(-20, 40):
        difficulty.uci_options_for(difficulty.clamp_level(raw))


def test_describe_mentions_the_level_and_its_elo():
    text = difficulty.describe(4)
    assert "4" in text
    assert "1250" in text


def test_stock_stockfish_is_rejected_because_its_floor_is_too_high():
    # Measured from the binary: Stockfish 19 reports UCI_Elo min 1320. Our three
    # easiest rungs are below that, so it cannot play them.
    problems = difficulty.check_engine_supports_ladder(1320, 3190)
    assert len(problems) == 1
    assert "1, 2, 3" in problems[0]


def test_fairy_stockfish_can_play_the_whole_ladder():
    # Measured from the binary: Fairy-Stockfish 14 reports UCI_Elo 500..2850.
    assert difficulty.check_engine_supports_ladder(500, 2850) == []
