"""Tests for motion/driver.py.

Runs on a bare laptop: no gantry, no controller, no pyserial. Nothing in here
sleeps in real time -- the GRBL tests hand the driver a fake clock whose only
way to advance is for the driver to ask it to.

    python -m pytest tests/test_driver.py -q
"""

import time

import pytest

import config
from motion import driver


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeClock(object):
    """Time that only moves when the code under test sleeps.

    This is how the timeout tests finish instantly. A real 30 s timeout would
    take 30 s to prove; this one takes microseconds and proves the same thing.
    """

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class FakeSerialPort(object):
    """Just enough of a GRBL controller to answer back.

    Line in -> "ok" out. A bare "?" byte -> a status report out. The report
    states come from self.states, and the last one repeats forever, so
    states=["Run", "Run", "Idle"] simulates a move that takes two polls.
    """

    def __init__(self, port=None, baudrate=None, timeout=None, states=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.states = list(states) if states else ["Idle"]
        self.lines = []          # complete G-code lines we were sent
        self.raw_writes = []     # every write() as text, realtime bytes included
        self.closed = False

        # A line that starts with this string gets "error:2" instead of "ok".
        self.reject_prefix = None
        # A line that starts with this string gets no reply at all -- a
        # controller that accepted the bytes and then wedged.
        self.silent_prefix = None
        # False = ignore "?" entirely, the deepest kind of wedge.
        self.answer_status = True
        # Appended to every status report, e.g. "MPos:10.000,20.000,0.000".
        self.extra_status_fields = []

        self._outbox = []
        self._partial = ""

    # -- the pyserial surface we actually use -------------------------------

    def write(self, data):
        text = data.decode("ascii")
        self.raw_writes.append(text)

        if text in ("?", "!", "~", "\x18"):
            if text == "?" and self.answer_status:
                self._outbox.append(self._status_line())
            return len(data)

        self._partial += text
        while "\n" in self._partial:
            line, _, self._partial = self._partial.partition("\n")
            line = line.strip()
            self.lines.append(line)
            if self.silent_prefix is not None and line.startswith(self.silent_prefix):
                continue
            if self.reject_prefix is not None and line.startswith(self.reject_prefix):
                self._outbox.append("error:2")
            else:
                self._outbox.append("ok")
        return len(data)

    def readline(self):
        if not self._outbox:
            return b""      # a real port returns empty on timeout
        return (self._outbox.pop(0) + "\n").encode("ascii")

    def close(self):
        self.closed = True

    # -- helpers ------------------------------------------------------------

    def _status_line(self):
        state = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        fields = [state] + list(self.extra_status_fields)
        return "<" + "|".join(fields) + ">"


def make_grbl(states=None):
    """Build a GrblMotionDriver wired to a fake port and a fake clock."""
    fake_port = FakeSerialPort(states=states)
    clock = FakeClock()

    # The driver calls the factory with keyword arguments to open the port. We
    # ignore them and hand back the port the test already built, so the test
    # can read back everything the driver wrote to it.
    def factory(port=None, baudrate=None, timeout=None):
        return fake_port

    grbl = driver.GrblMotionDriver(
        serial_factory=factory,
        time_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    return grbl, fake_port, clock


def travel(points, why="test travel"):
    return {
        "kind": "travel",
        "points": points,
        "speed_mms": config.TRAVEL_SPEED_MMS,
        "why": why,
    }


def drag(points, why="test drag"):
    return {
        "kind": "drag",
        "points": points,
        "speed_mms": config.DRAG_SPEED_MMS,
        "why": why,
    }


def park_point():
    return (config.PARK_X_MM, config.PARK_Y_MM)


# ---------------------------------------------------------------------------
# The module must import and work with no hardware and no pyserial
# ---------------------------------------------------------------------------

def test_module_imports_without_pyserial_at_top_level():
    # pyserial is not installed on the dev laptops. If anyone moves the import
    # to the top of driver.py, "import motion.driver" starts failing for the
    # whole team and every other module's tests go red at once.
    assert not hasattr(driver, "serial")


def test_make_motion_driver_returns_the_fake_when_the_flag_is_set(monkeypatch):
    # monkeypatch is a pytest fixture: ask for it as an argument and pytest
    # hands you an object that can change a value and put it back afterwards,
    # so a test cannot leak a setting into the next test.
    monkeypatch.setattr(config, "USE_FAKE_HARDWARE", True)
    assert isinstance(driver.make_motion_driver(), driver.FakeMotionDriver)


def test_make_motion_driver_returns_the_real_one_when_the_flag_is_clear(monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_HARDWARE", False)
    clock = FakeClock()
    made = driver.make_motion_driver(
        serial_factory=FakeSerialPort,
        time_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    assert isinstance(made, driver.GrblMotionDriver)


def test_the_flag_the_project_ships_with_selects_the_fake():
    # Until December this must stay true, or play_text.py stops running on a
    # laptop.
    assert config.USE_FAKE_HARDWARE is True


# ---------------------------------------------------------------------------
# The abstract base promises nothing and implements nothing
# ---------------------------------------------------------------------------

def test_base_class_refuses_every_method():
    base = driver.MotionDriver()
    # pytest.raises passes only if the block raises what you named.
    with pytest.raises(NotImplementedError):
        base.home()
    with pytest.raises(NotImplementedError):
        base.run_segment(travel([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(NotImplementedError):
        base.wait_until_idle()
    with pytest.raises(NotImplementedError):
        base.position()
    with pytest.raises(NotImplementedError):
        base.set_magnet(True)
    with pytest.raises(NotImplementedError):
        base.is_homed()


def test_both_drivers_are_motion_drivers():
    assert issubclass(driver.FakeMotionDriver, driver.MotionDriver)
    assert issubclass(driver.GrblMotionDriver, driver.MotionDriver)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_segment_length_of_a_single_point_is_zero():
    assert driver.segment_length_mm([(10.0, 10.0)]) == 0.0


def test_segment_length_adds_up_the_legs():
    points = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    assert driver.segment_length_mm(points) == pytest.approx(7.0)


def test_segment_length_measures_diagonals_properly():
    assert driver.segment_length_mm([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0)


def test_check_segment_fills_in_a_missing_speed_from_the_kind():
    _kind, _points, speed, _why = driver.check_segment(
        {"kind": "drag", "points": [(0.0, 0.0), (1.0, 1.0)]}
    )
    assert speed == config.DRAG_SPEED_MMS


def test_check_segment_rejects_a_bad_kind():
    with pytest.raises(ValueError):
        driver.check_segment({"kind": "teleport", "points": [(0.0, 0.0)]})


def test_check_segment_rejects_empty_points():
    with pytest.raises(ValueError):
        driver.check_segment({"kind": "travel", "points": []})


def test_check_segment_rejects_a_non_positive_speed():
    with pytest.raises(ValueError):
        driver.check_segment(
            {"kind": "travel", "points": [(0.0, 0.0)], "speed_mms": 0.0}
        )


def test_parse_status_reads_the_state_word():
    report = driver.parse_status("<Idle|MPos:0.000,0.000,0.000|FS:0,0>")
    assert report["state"] == "Idle"
    assert report["mpos"] == (0.0, 0.0)


def test_parse_status_keeps_only_x_and_y():
    report = driver.parse_status("<Run|MPos:114.300,57.150,2.500|FS:300,0>")
    assert report["mpos"] == (114.3, 57.15)


def test_parse_status_reads_the_work_offset():
    report = driver.parse_status(
        "<Idle|MPos:10.000,20.000,0.000|WCO:1.500,2.500,0.000>"
    )
    assert report["wco"] == (1.5, 2.5)


def test_parse_status_drops_the_hold_substate():
    # "Hold:0" and "Hold:1" are both Hold as far as we care.
    assert driver.parse_status("<Hold:0|MPos:0.000,0.000,0.000>")["state"] == "Hold"


def test_parse_status_ignores_anything_that_is_not_a_report():
    assert driver.parse_status("ok") is None
    assert driver.parse_status("Grbl 1.1f ['$' for help]") is None
    assert driver.parse_status("") is None
    assert driver.parse_status(None) is None


# ---------------------------------------------------------------------------
# FakeMotionDriver
# ---------------------------------------------------------------------------

def test_fake_starts_parked_off_the_board_and_unhomed():
    fake = driver.FakeMotionDriver()
    assert fake.position() == park_point()
    assert fake.is_homed() is False
    assert fake.magnet_engaged is False
    assert fake.history == []
    # Physical rule 3: parking on a square would make that square read occupied
    # forever, because our own magnet looks like a piece to the sensor grid.
    assert config.PARK_Y_MM < 0.0


def test_fake_home_sets_homed_and_lands_on_the_homing_offset():
    fake = driver.FakeMotionDriver()
    fake.home()
    assert fake.is_homed() is True
    assert fake.position() == (config.HOMING_OFFSET_X_MM, config.HOMING_OFFSET_Y_MM)
    assert fake.home_count == 1


def test_fake_running_a_segment_ends_at_its_last_point():
    fake = driver.FakeMotionDriver()
    fake.home()
    fake.run_segment(travel([fake.position(), (100.0, 50.0), (200.0, 75.0)]))
    assert fake.position() == (200.0, 75.0)


def test_fake_records_every_segment_in_order():
    fake = driver.FakeMotionDriver()
    first = travel([park_point(), (0.0, 0.0)], why="go to a1")
    second = drag([(0.0, 0.0), (0.0, 57.15)], why="a1 to a2")
    fake.run_segment(first)
    fake.run_segment(second)

    assert len(fake.history) == 2
    assert [row["why"] for row in fake.history] == ["go to a1", "a1 to a2"]
    assert [row["kind"] for row in fake.history] == ["travel", "drag"]


def test_fake_history_is_not_a_live_reference_to_the_callers_dict():
    fake = driver.FakeMotionDriver()
    segment = travel([park_point(), (0.0, 0.0)], why="original")
    fake.run_segment(segment)
    segment["why"] = "someone reused the dict"
    assert fake.history[0]["why"] == "original"


def test_fake_engages_the_magnet_for_a_drag_and_drops_it_for_a_travel():
    fake = driver.FakeMotionDriver()
    fake.run_segment(travel([park_point(), (0.0, 0.0)]))
    assert fake.magnet_engaged is False

    fake.run_segment(drag([(0.0, 0.0), (57.15, 0.0)]))
    assert fake.magnet_engaged is True

    fake.run_segment(travel([(57.15, 0.0), park_point()]))
    assert fake.magnet_engaged is False


def test_fake_magnet_can_also_be_driven_by_hand():
    fake = driver.FakeMotionDriver()
    fake.set_magnet(True)
    assert fake.magnet_engaged is True
    fake.set_magnet(False)
    assert fake.magnet_engaged is False


def test_fake_ignores_a_magnet_command_that_changes_nothing():
    fake = driver.FakeMotionDriver()
    fake.set_magnet(False)
    fake.set_magnet(False)
    assert fake.events == []


def test_fake_events_interleave_magnet_changes_with_segments():
    # A capture is removal first, and the ordering of magnet-on / drag /
    # magnet-off is the part that goes wrong on real hardware, so the fake has
    # to make that ordering visible.
    fake = driver.FakeMotionDriver()
    fake.run_segment(travel([park_point(), (0.0, 0.0)]))
    fake.run_segment(drag([(0.0, 0.0), (57.15, 0.0)]))
    kinds = [name for name, _payload in fake.events]
    assert kinds == ["segment", "magnet", "segment"]
    assert fake.events[1][1] is True


def test_fake_wait_until_idle_returns_at_once():
    fake = driver.FakeMotionDriver()
    fake.run_segment(travel([park_point(), (400.0, 400.0)]))
    started = time.monotonic()
    assert fake.wait_until_idle() is True
    assert fake.wait_until_idle(timeout_s=0.001) is True
    # The whole point of the fake: a thousand-move game must not take a
    # thousand moves' worth of seconds.
    assert time.monotonic() - started < 0.05


def test_fake_adds_up_the_distance_it_travelled():
    fake = driver.FakeMotionDriver()
    fake.run_segment(travel([(0.0, 0.0), (30.0, 40.0)]))
    fake.run_segment(travel([(30.0, 40.0), (30.0, 50.0)]))
    assert fake.distance_mm == pytest.approx(60.0)


def test_fake_charges_simulated_time_at_the_segments_own_speed():
    fake = driver.FakeMotionDriver()
    fake.set_magnet(False)      # no-op, so no settle time creeps into the sum
    fake.run_segment(travel([(0.0, 0.0), (config.TRAVEL_SPEED_MMS, 0.0)]))
    assert fake.sim_time_s == pytest.approx(1.0)


def test_fake_charges_the_magnet_settle_time_too():
    fake = driver.FakeMotionDriver()
    fake.set_magnet(True)
    assert fake.sim_time_s == pytest.approx(config.MAGNET_ENGAGE_SETTLE_S)


def test_a_drag_is_slower_than_the_same_path_travelled_empty():
    # If these two ever come out equal, someone has collapsed the two speeds
    # into one and the pieces will start slipping off the magnet.
    path = [(0.0, 0.0), (200.0, 0.0)]
    empty = driver.FakeMotionDriver()
    empty.run_segment(travel(path))
    loaded = driver.FakeMotionDriver()
    loaded.set_magnet(True)     # settle already paid, so we compare motion only
    loaded.run_segment(drag(path))
    assert loaded.sim_time_s - config.MAGNET_ENGAGE_SETTLE_S > empty.sim_time_s


def test_fake_stays_quiet_when_segments_chain_head_to_tail():
    fake = driver.FakeMotionDriver()
    fake.run_segment(travel([park_point(), (0.0, 0.0)]))
    fake.run_segment(drag([(0.0, 0.0), (57.15, 57.15)]))
    fake.run_segment(travel([(57.15, 57.15), park_point()]))
    assert fake.warnings == []


def test_fake_complains_when_a_segment_starts_somewhere_else():
    fake = driver.FakeMotionDriver()
    fake.run_segment(travel([park_point(), (0.0, 0.0)]))
    fake.run_segment(drag([(228.6, 228.6), (285.75, 228.6)], why="planned from stale"))
    assert len(fake.warnings) == 1
    assert "stale" in fake.warnings[0]


def test_fake_refuses_a_malformed_segment():
    fake = driver.FakeMotionDriver()
    with pytest.raises(ValueError):
        fake.run_segment({"kind": "drag"})
    with pytest.raises(ValueError):
        fake.run_segment({"kind": "shuffle", "points": [(0.0, 0.0)]})
    with pytest.raises(ValueError):
        fake.run_segment("G0 X10")


def test_fake_reset_history_forgets_the_log_but_not_the_position():
    fake = driver.FakeMotionDriver()
    fake.home()
    fake.run_segment(travel([fake.position(), (150.0, 25.0)]))
    fake.reset_history()
    assert fake.history == []
    assert fake.events == []
    assert fake.distance_mm == 0.0
    assert fake.position() == (150.0, 25.0)
    assert fake.is_homed() is True


def test_fake_can_be_started_somewhere_other_than_park():
    fake = driver.FakeMotionDriver(x_mm=10.0, y_mm=20.0)
    assert fake.position() == (10.0, 20.0)


# ---------------------------------------------------------------------------
# GrblMotionDriver -- driven through a fake port, no hardware, no pyserial
# ---------------------------------------------------------------------------

def test_grbl_sends_no_gcode_on_connect():
    # A controller with homing enabled boots into Alarm and answers every G-code
    # line with error:9 until it has homed, so connecting must stay silent.
    _grbl, port, _clock = make_grbl()
    assert port.lines == []


def test_grbl_home_sends_the_homing_command_then_the_preamble_then_the_offset():
    grbl, port, _clock = make_grbl()
    port.lines = []
    grbl.home()

    # Units and modes go out after $H (the controller only accepts G-code once
    # it is out of Alarm) and before G92 (whose numbers are read in the current units).
    assert port.lines[0] == "$H"
    assert port.lines[1:4] == list(driver.GRBL_PREAMBLE)
    assert port.lines[4].startswith("G92 ")
    assert grbl.is_homed() is True
    assert grbl.position() == pytest.approx(
        (config.HOMING_OFFSET_X_MM, config.HOMING_OFFSET_Y_MM)
    )


def test_grbl_home_waits_the_homing_deadline_not_the_serial_one():
    # "$H" is only answered after the whole homing crawl, so the driver must keep
    # waiting far longer than the 2 s it allows an ordinary line. A controller
    # that never answers still gives up -- at the homing deadline.
    grbl, port, clock = make_grbl()
    port.silent_prefix = "$H"
    with pytest.raises(driver.MotionTimeout):
        grbl.home()
    assert clock.t >= driver.HOMING_TIMEOUT_S
    assert grbl.is_homed() is False


def test_grbl_asks_for_status_with_a_bare_question_mark_and_no_newline():
    # A newline after a realtime byte pushes an empty line into the buffer,
    # GRBL answers it with an extra "ok", and every reply after that belongs to
    # the wrong command.
    grbl, port, _clock = make_grbl()
    grbl.wait_until_idle()
    assert "?" in port.raw_writes
    assert "?\n" not in port.raw_writes


def test_grbl_waits_through_several_run_reports_before_calling_it_idle():
    grbl, port, clock = make_grbl(states=["Run", "Run", "Run", "Idle"])
    grbl.wait_until_idle()
    assert port.raw_writes.count("?") == 4
    # It polled, so it must have waited between polls rather than spinning.
    assert clock.t == pytest.approx(3 * config.MOTION_POLL_INTERVAL_S)


def test_grbl_gives_up_on_a_controller_that_never_goes_idle():
    grbl, _port, clock = make_grbl(states=["Run"])
    with pytest.raises(driver.MotionTimeout):
        grbl.wait_until_idle()
    # The timeout came from config, and the fake clock proves no real seconds
    # were burned proving it.
    assert clock.t >= config.MOTION_TIMEOUT_S


def test_grbl_honours_a_shorter_timeout_when_asked():
    grbl, _port, clock = make_grbl(states=["Run"])
    with pytest.raises(driver.MotionTimeout):
        grbl.wait_until_idle(timeout_s=0.5)
    assert clock.t < config.MOTION_TIMEOUT_S


def test_grbl_treats_an_alarm_as_an_error_not_as_arrival():
    grbl, _port, _clock = make_grbl(states=["Alarm"])
    with pytest.raises(driver.ControllerError):
        grbl.wait_until_idle()


def test_grbl_gives_up_on_a_line_that_is_never_acknowledged():
    # One reference build called a blocking readline() with no timeout. When
    # its controller stopped answering, the whole program froze with no
    # message. Every wait in this module has a deadline, including this one.
    grbl, port, clock = make_grbl()
    port.silent_prefix = "G0"
    with pytest.raises(driver.MotionTimeout):
        grbl.run_segment(travel([(0.0, 0.0), (57.15, 0.0)]))
    assert clock.t >= config.SERIAL_TIMEOUT_S


def test_grbl_gives_up_on_a_controller_that_ignores_the_status_request():
    grbl, port, clock = make_grbl()
    port.answer_status = False
    with pytest.raises(driver.MotionTimeout):
        grbl.wait_until_idle()
    assert clock.t >= config.MOTION_TIMEOUT_S


def test_grbl_raises_when_the_controller_rejects_a_line():
    grbl, port, _clock = make_grbl()
    port.reject_prefix = "G0"
    with pytest.raises(driver.ControllerError):
        grbl.run_segment(travel([(0.0, 0.0), (57.15, 0.0)]))


def test_grbl_emits_g0_rapids_for_a_travel_segment():
    grbl, port, _clock = make_grbl()
    port.lines = []
    grbl.run_segment(travel([(0.0, 0.0), (57.15, 0.0), (57.15, 57.15)]))

    assert "G0 X57.150 Y0.000" in port.lines
    assert "G0 X57.150 Y57.150" in port.lines
    # A rapid has no feed word: the controller uses its own maximum rate.
    assert not any(line.startswith("G0") and " F" in line for line in port.lines)


def test_grbl_emits_g1_with_a_feed_rate_for_a_drag_segment():
    grbl, port, _clock = make_grbl()
    port.lines = []
    grbl.run_segment(drag([(0.0, 0.0), (0.0, 57.15)]))

    expected_feed = config.DRAG_SPEED_MMS * 60.0
    assert "G1 X0.000 Y57.150 F%.1f" % (expected_feed,) in port.lines


def test_grbl_feed_rate_is_per_minute_not_per_second():
    # GRBL's F word is mm/min. Sending mm/s makes every drag run 60x too slow,
    # which on the bench looks exactly like a stuck controller.
    grbl, port, _clock = make_grbl()
    port.lines = []
    grbl.run_segment(drag([(0.0, 0.0), (100.0, 0.0)]))
    feeds = [line for line in port.lines if " F" in line]
    assert len(feeds) == 1
    feed_value = float(feeds[0].split(" F")[1])
    assert feed_value == pytest.approx(config.DRAG_SPEED_MMS * 60.0)
    assert feed_value != pytest.approx(config.DRAG_SPEED_MMS)


def test_grbl_turns_the_magnet_on_before_the_drag_and_dwells_to_let_it_settle():
    grbl, port, _clock = make_grbl()
    port.lines = []
    grbl.run_segment(drag([(0.0, 0.0), (57.15, 0.0)]))

    assert port.lines[0] == driver.MAGNET_ON_GCODE
    assert port.lines[1] == "G4 P%.3f" % (config.MAGNET_ENGAGE_SETTLE_S,)
    assert port.lines[2].startswith("G1 ")
    assert grbl.magnet_engaged is True


def test_grbl_releases_the_magnet_before_travelling_away():
    grbl, port, _clock = make_grbl()
    grbl.run_segment(drag([(0.0, 0.0), (57.15, 0.0)]))
    port.lines = []
    grbl.run_segment(travel([(57.15, 0.0), park_point()]))

    assert port.lines[0] == driver.MAGNET_OFF_GCODE
    assert grbl.magnet_engaged is False


def test_grbl_does_not_re_send_a_magnet_state_it_is_already_in():
    grbl, port, _clock = make_grbl()
    grbl.run_segment(travel([(0.0, 0.0), (57.15, 0.0)]))
    port.lines = []
    grbl.run_segment(travel([(57.15, 0.0), (114.3, 0.0)]))
    assert driver.MAGNET_OFF_GCODE not in port.lines


def test_grbl_waits_for_the_previous_segment_before_starting_the_next():
    # This is the bug the module exists to prevent: "ok" means queued, not
    # arrived. Each run_segment must poll for Idle before it touches anything.
    grbl, port, _clock = make_grbl()
    polls_after_connect = port.raw_writes.count("?")
    grbl.run_segment(travel([(0.0, 0.0), (57.15, 0.0)]))
    assert port.raw_writes.count("?") > polls_after_connect


def test_grbl_believes_the_controller_about_where_it_is():
    grbl, port, _clock = make_grbl()
    port.extra_status_fields = ["MPos:123.400,56.700,0.000"]
    grbl.wait_until_idle()
    assert grbl.position() == pytest.approx((123.4, 56.7))


def test_grbl_corrects_a_machine_position_by_the_work_offset():
    # WPos = MPos - WCO. Skipping the correction puts every reported position
    # out by the homing offset.
    grbl, port, _clock = make_grbl()
    port.extra_status_fields = ["MPos:100.000,100.000,0.000", "WCO:10.000,20.000,0.000"]
    grbl.wait_until_idle()
    assert grbl.position() == pytest.approx((90.0, 80.0))


def test_grbl_remembers_a_work_offset_it_was_told_about_earlier():
    # GRBL sends WCO only every few reports, not every one.
    grbl, port, _clock = make_grbl()
    port.extra_status_fields = ["MPos:0.000,0.000,0.000", "WCO:5.000,5.000,0.000"]
    grbl.wait_until_idle()
    port.extra_status_fields = ["MPos:100.000,50.000,0.000"]
    grbl.wait_until_idle()
    assert grbl.position() == pytest.approx((95.0, 45.0))


def test_grbl_realtime_helpers_send_single_bytes():
    grbl, port, _clock = make_grbl()
    grbl.feed_hold()
    grbl.cycle_start()
    grbl.soft_reset()
    assert port.raw_writes[-3:] == ["!", "~", "\x18"]
    # A soft reset throws away the machine's idea of where it is.
    assert grbl.is_homed() is False


def test_grbl_close_drops_the_piece_before_closing_the_port():
    # Exiting mid-drag with the magnet live leaves a piece welded to a dead
    # carriage under the board.
    grbl, port, _clock = make_grbl()
    grbl.set_magnet(True)
    grbl.close()
    assert port.lines[-2] == driver.MAGNET_OFF_GCODE
    assert port.closed is True
