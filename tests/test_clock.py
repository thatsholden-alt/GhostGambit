"""Tests for game/clock.py, run on a fake clock so a 10-minute game takes microseconds."""

import pytest

import config
from game.clock import ChessClock, format_time

WHITE, BLACK = True, False   # the same booleans python-chess uses


class FakeTime(object):
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t


def make_clock(base_s=600.0, increment_s=5.0):
    fake = FakeTime()
    return ChessClock(base_s, increment_s, time_fn=fake.now), fake


def test_defaults_come_from_config():
    clock = ChessClock(time_fn=FakeTime().now)
    assert clock.remaining(WHITE) == config.DEFAULT_BASE_TIME_S
    assert clock.increment_s == config.DEFAULT_INCREMENT_S


def test_nothing_runs_before_the_game_starts():
    clock, fake = make_clock()
    fake.t += 30
    assert clock.remaining(WHITE) == 600.0
    assert clock.remaining(BLACK) == 600.0


def test_only_the_side_to_move_loses_time():
    clock, fake = make_clock()
    clock.start(WHITE)
    fake.t += 12.5
    assert clock.remaining(WHITE) == pytest.approx(587.5)
    assert clock.remaining(BLACK) == 600.0


def test_ending_a_turn_adds_the_increment_and_starts_the_other_clock():
    clock, fake = make_clock()
    clock.start(WHITE)
    fake.t += 20
    clock.end_turn(WHITE)
    assert clock.remaining(WHITE) == pytest.approx(585.0)   # 600 - 20 + 5
    assert clock.to_move == BLACK
    fake.t += 3
    assert clock.remaining(BLACK) == pytest.approx(597.0)
    assert clock.remaining(WHITE) == pytest.approx(585.0)   # white's clock stopped


def test_robot_motion_is_charged_to_nobody():
    # Engine (black) thinks for 0.5 s, then the gantry spends 9 s on a capture.
    clock, fake = make_clock()
    clock.start(BLACK)
    fake.t += 0.5
    clock.pause()                 # gantry starts moving
    clock.end_turn(BLACK)         # the engine's move is decided
    fake.t += 9.0                 # dragging pieces
    assert clock.remaining(BLACK) == pytest.approx(604.5)   # 600 - 0.5 + 5
    assert clock.remaining(WHITE) == 600.0                  # not charged yet
    clock.resume()                # gantry parked: now it's the human's time
    fake.t += 4.0
    assert clock.remaining(WHITE) == pytest.approx(596.0)


def test_running_out_of_time_loses_and_stays_lost():
    clock, fake = make_clock(base_s=60.0)
    clock.start(WHITE)
    fake.t += 59.9
    assert clock.flag_fallen() is None
    fake.t += 0.2
    assert clock.flag_fallen() == WHITE
    assert clock.remaining(WHITE) == 0.0
    # Pressing the button after the flag fell is too late: no increment, still lost.
    clock.end_turn(WHITE)
    assert clock.flag_fallen() == WHITE
    assert clock.remaining(WHITE) == 0.0
    clock.resume()
    assert clock.remaining(BLACK) == 60.0   # black's clock never starts


def test_a_paused_clock_cannot_run_out():
    clock, fake = make_clock(base_s=10.0)
    clock.start(WHITE)
    clock.pause()
    fake.t += 3600
    assert clock.flag_fallen() is None
    assert clock.remaining(WHITE) == 10.0


def test_ending_the_wrong_sides_turn_is_a_bug():
    clock, _fake = make_clock()
    clock.start(WHITE)
    with pytest.raises(ValueError):
        clock.end_turn(BLACK)


def test_an_untimed_game_never_runs_out():
    clock, fake = make_clock(base_s=None)
    clock.start(WHITE)
    fake.t += 10 ** 6
    assert clock.remaining(WHITE) is None
    assert clock.flag_fallen() is None
    clock.end_turn(WHITE)
    assert clock.to_move == BLACK


def test_display_format():
    assert format_time(600.0) == "10:00"
    assert format_time(598.7) == "9:58"
    assert format_time(10.0) == "0:10"
    assert format_time(7.49) == "0:07.4"    # tenths, rounded down, near the end
    assert format_time(0.0) == "0:00.0"
    assert format_time(None) == "--:--"
