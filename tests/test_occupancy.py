"""Tests for the occupancy grid and its debounce filter.

Runs on a bare laptop: no I2C, no smbus, no sensor board, and -- importantly --
no sleeping. The debounce tests drive a hand-cranked clock instead of waiting,
so the whole file finishes in milliseconds.

    python -m pytest tests/test_occupancy.py -q
"""

import time

import chess
import pytest

import config
from sensing import occupancy


class FakeClock(object):
    """A stopwatch we turn by hand, standing in for time.monotonic.

    Defining __call__ makes an instance of this class usable as if it were a
    function: writing clock() runs the __call__ method below. That is what lets
    us pass the object straight into Debouncer(time_fn=clock), which expects
    something callable that returns a number of seconds.
    """

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


# A mask that is easy to eyeball: the two back ranks and the two pawn ranks.
START_MASK = 0xFFFF00000000FFFF


# ---------------------------------------------------------------------------
# occupancy_from_board
# ---------------------------------------------------------------------------

def test_start_position_fills_ranks_one_two_seven_and_eight():
    mask = occupancy.occupancy_from_board(chess.Board())
    assert mask == START_MASK, "\n" + occupancy.format_mask(mask)


def test_empty_board_is_zero():
    assert occupancy.occupancy_from_board(chess.Board(fen=None)) == 0


def test_bit_zero_is_a1_and_bit_sixtythree_is_h8():
    # The single most important thing in this file. If this is backwards, move
    # inference silently matches the wrong move.
    board = chess.Board(fen=None)
    board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    assert occupancy.occupancy_from_board(board) == 1

    board = chess.Board(fen=None)
    board.set_piece_at(chess.H8, chess.Piece(chess.ROOK, chess.BLACK))
    assert occupancy.occupancy_from_board(board) == 1 << 63


def test_every_one_of_the_sixtyfour_squares_owns_its_own_bit():
    # Checking a1 and h8 pins the two ends but leaves 62 squares unexamined; a
    # file/rank transposition, for instance, fixes both corners and scrambles
    # everything between them.
    for square in chess.SQUARES:
        board = chess.Board(fen=None)
        board.set_piece_at(square, chess.Piece(chess.PAWN, chess.WHITE))
        assert occupancy.occupancy_from_board(board) == 1 << square, (
            chess.square_name(square)
        )


def test_colour_does_not_matter_only_presence():
    white = chess.Board(fen=None)
    white.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
    black = chess.Board(fen=None)
    black.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.BLACK))
    assert occupancy.occupancy_from_board(white) == occupancy.occupancy_from_board(black)


def test_a_quiet_move_clears_the_from_square_and_sets_the_to_square():
    board = chess.Board()
    before = occupancy.occupancy_from_board(board)
    board.push(chess.Move.from_uci("e2e4"))
    after = occupancy.occupancy_from_board(board)

    assert occupancy.is_occupied(before, chess.E2)
    assert not occupancy.is_occupied(after, chess.E2)
    assert occupancy.is_occupied(after, chess.E4)
    # Nothing else moved.
    assert before ^ after == (1 << chess.E2) | (1 << chess.E4)


def test_a_capture_leaves_the_piece_count_one_lower():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    before = occupancy.occupancy_from_board(board)
    board.push(chess.Move.from_uci("e4d5"))
    after = occupancy.occupancy_from_board(board)
    assert bin(after).count("1") == bin(before).count("1") - 1


def test_en_passant_empties_a_square_that_is_not_the_destination():
    # The physical case that catches everyone: after exd6 e.p. the removed pawn
    # was on d5, not on d6.
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    board.push(chess.Move.from_uci("e5d6"))
    mask = occupancy.occupancy_from_board(board)
    assert not occupancy.is_occupied(mask, chess.D5)
    assert occupancy.is_occupied(mask, chess.D6)


def test_castling_shows_both_king_and_rook_moved():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    board.push(chess.Move.from_uci("e1g1"))
    mask = occupancy.occupancy_from_board(board)
    assert occupancy.is_occupied(mask, chess.G1)
    assert occupancy.is_occupied(mask, chess.F1)
    assert not occupancy.is_occupied(mask, chess.E1)
    assert not occupancy.is_occupied(mask, chess.H1)


def test_queenside_castling_moves_the_rook_three_squares_not_two():
    # Only the kingside case was covered, and the two sides are not mirror
    # images: O-O-O leaves a1 and e1 empty and fills c1 and d1, so a1 must clear
    # while b1 stays empty throughout.
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
    board.push(chess.Move.from_uci("e1c1"))
    mask = occupancy.occupancy_from_board(board)
    assert occupancy.squares_in_mask(mask) == sorted([chess.C1, chess.D1, chess.E8])


def test_black_castling_is_read_the_same_way_as_white():
    # Black-vs-white asymmetry is where square arithmetic usually goes wrong:
    # the same castle happens 56 squares higher up.
    board = chess.Board("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")
    board.push(chess.Move.from_uci("e8g8"))
    mask = occupancy.occupancy_from_board(board)
    assert occupancy.squares_in_mask(mask) == sorted([chess.E1, chess.F8, chess.G8])


def test_promotion_is_invisible_to_the_occupancy_grid():
    # A fact the whole sensing stack rests on: reed switches see magnets, not
    # piece types, so queening and knighting the same pawn are the same reading.
    # Move inference cannot tell them apart and the UI has to ask.
    queened = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    knighted = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    queened.push(chess.Move.from_uci("a7a8q"))
    knighted.push(chess.Move.from_uci("a7a8n"))
    assert occupancy.occupancy_from_board(queened) == occupancy.occupancy_from_board(knighted)
    assert occupancy.is_occupied(occupancy.occupancy_from_board(queened), chess.A8)


def test_agrees_with_python_chess_own_occupancy_bitboard():
    # A cross-check against the library, which numbers squares the same way.
    # If these ever disagree, our bit order drifted.
    board = chess.Board()
    for uci in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"]:
        board.push(chess.Move.from_uci(uci))
        assert occupancy.occupancy_from_board(board) == board.occupied


def test_mask_helpers_round_trip():
    squares = [chess.A1, chess.E4, chess.H8]
    mask = occupancy.mask_from_squares(squares)
    assert occupancy.squares_in_mask(mask) == sorted(squares)


def test_format_mask_draws_rank_eight_first():
    lines = occupancy.format_mask(START_MASK).split("\n")
    assert len(lines) == config.BOARD_RANKS
    assert lines[0] == "# # # # # # # #"      # rank 8
    assert lines[3] == ". . . . . . . ."      # rank 5
    assert lines[7] == "# # # # # # # #"      # rank 1


def test_format_mask_puts_the_a_file_on_the_left():
    # START_MASK is symmetric left-to-right, so the test above cannot tell an
    # a-file-first drawing from an h-file-first one. b1 can.
    lines = occupancy.format_mask(1 << chess.B1).split("\n")
    assert lines[7] == ". # . . . . . ."
    assert lines[0] == ". . . . . . . ."


# ---------------------------------------------------------------------------
# The sensors
# ---------------------------------------------------------------------------

def test_base_class_refuses_to_be_read():
    # pytest.raises is a context manager: the "with" block passes only if the
    # code inside it raises that exact exception type. It is how you test that
    # something fails on purpose.
    with pytest.raises(NotImplementedError):
        occupancy.OccupancySensor().read_raw()


def test_fake_sensor_can_be_set_from_a_board():
    sensor = occupancy.FakeOccupancySensor()
    assert sensor.read_raw() == 0
    sensor.set_from_board(chess.Board())
    assert sensor.read_raw() == START_MASK


def test_fake_sensor_can_be_set_from_a_raw_mask():
    sensor = occupancy.FakeOccupancySensor()
    sensor.set_mask(1 << chess.E4)
    assert sensor.read_raw() == 1 << chess.E4


def test_fake_sensor_rejects_a_mask_that_does_not_fit_the_board():
    sensor = occupancy.FakeOccupancySensor()
    with pytest.raises(ValueError):
        sensor.set_mask(1 << 64)
    with pytest.raises(ValueError):
        sensor.set_mask(-1)


def test_factory_returns_the_fake_when_the_flag_is_on(monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_HARDWARE", True)
    assert isinstance(occupancy.make_occupancy_sensor(), occupancy.FakeOccupancySensor)


def test_smbus_is_never_imported_at_module_scope():
    # The entire project has to import on a Windows laptop with no I2C bus. If
    # smbus ever creeps up to the top of sensing/occupancy.py it would land as a
    # module-level name here, and every test in the repo would stop collecting.
    assert not hasattr(occupancy, "smbus")
    assert not hasattr(occupancy, "smbus2")


class FakeI2CBus(object):
    """Stands in for smbus. Registers are a dict keyed by (address, register)."""

    def __init__(self, default=0xFF):
        self.default = default          # all pins high == all switches open
        self.registers = {}
        self.writes = []
        self.closed = False

    def read_byte_data(self, address, register):
        return self.registers.get((address, register), self.default)

    def write_byte_data(self, address, register, value):
        self.writes.append((address, register, value))
        self.registers[(address, register)] = value

    def close(self):
        self.closed = True


def test_factory_builds_the_real_sensor_when_the_flag_is_off(monkeypatch):
    # The previous version of this test wrapped the call in
    # "except Exception: return", so it passed no matter what went wrong --
    # including the real class having been renamed out from under the factory.
    # Handing in a stand-in bus lets the real branch run all the way through on
    # a laptop, so the assertion below actually means something.
    monkeypatch.setattr(config, "USE_FAKE_HARDWARE", False)
    sensor = occupancy.make_occupancy_sensor(bus=FakeI2CBus())
    assert isinstance(sensor, occupancy.Mcp23017OccupancySensor)
    assert not isinstance(sensor, occupancy.FakeOccupancySensor)


def test_real_sensor_configures_every_expander_as_pulled_up_inputs():
    bus = FakeI2CBus()
    occupancy.Mcp23017OccupancySensor(bus=bus)
    for address in occupancy.EXPANDER_ADDRESSES:
        written = dict((reg, val) for (addr, reg, val) in bus.writes if addr == address)
        assert written[occupancy._REG_IODIRA] == 0xFF
        assert written[occupancy._REG_IODIRB] == 0xFF
        assert written[occupancy._REG_GPPUA] == 0xFF
        assert written[occupancy._REG_GPPUB] == 0xFF


def test_real_sensor_reads_an_empty_board_as_zero():
    # Every switch open, every pin pulled high, active low -> nothing occupied.
    sensor = occupancy.Mcp23017OccupancySensor(bus=FakeI2CBus(default=0xFF))
    assert sensor.read_raw() == 0


def test_real_sensor_reads_all_switches_closed_as_a_full_board():
    sensor = occupancy.Mcp23017OccupancySensor(bus=FakeI2CBus(default=0x00))
    assert sensor.read_raw() == occupancy.FULL_BOARD_MASK


def test_real_sensor_puts_expander_zero_port_a_bit_zero_on_square_a1():
    bus = FakeI2CBus(default=0xFF)
    # Pull bit 0 of the first expander's port A low: switch closed on a1.
    bus.registers[(occupancy.EXPANDER_ADDRESSES[0], occupancy._REG_GPIOA)] = 0xFE
    sensor = occupancy.Mcp23017OccupancySensor(bus=bus)
    assert sensor.read_raw() == 1 << chess.A1


def test_real_sensor_puts_the_last_expander_port_b_bit_seven_on_square_h8():
    bus = FakeI2CBus(default=0xFF)
    bus.registers[(occupancy.EXPANDER_ADDRESSES[-1], occupancy._REG_GPIOB)] = 0x7F
    sensor = occupancy.Mcp23017OccupancySensor(bus=bus)
    assert sensor.read_raw() == 1 << chess.H8


def test_every_expander_pin_lands_on_its_own_square():
    # Two spot checks are not enough for a 64-way mapping: a swapped port A/B,
    # or expanders enumerated backwards, would sail past a1-and-h8-only test.
    # Drive exactly one pin low at a time and insist on exactly one bit out.
    for square in range(64):
        expander, within = divmod(square, occupancy.EXPANDER_SQUARES)
        register = occupancy._REG_GPIOA if within < 8 else occupancy._REG_GPIOB
        bus = FakeI2CBus(default=0xFF)
        bus.registers[(occupancy.EXPANDER_ADDRESSES[expander], register)] = (
            0xFF & ~(1 << (within % 8))
        )
        sensor = occupancy.Mcp23017OccupancySensor(bus=bus)
        got = sensor.read_raw()
        assert got == 1 << square, (
            "square %d came back as\n%s" % (square, occupancy.format_mask(got))
        )


def test_real_sensor_can_be_wired_active_high_instead():
    bus = FakeI2CBus(default=0x00)
    sensor = occupancy.Mcp23017OccupancySensor(bus=bus, active_low=False)
    assert sensor.read_raw() == 0


def test_active_high_wiring_reads_a_pin_that_is_high_as_occupied():
    # The other half of the active-high case. Testing only the all-zero
    # direction would still pass if read_raw() had stopped reading the bus.
    bus = FakeI2CBus(default=0xFF)
    sensor = occupancy.Mcp23017OccupancySensor(bus=bus, active_low=False)
    assert sensor.read_raw() == occupancy.FULL_BOARD_MASK


def test_active_high_wiring_leaves_the_internal_pullups_switched_off():
    # Regression guard. The class used to turn the pull-ups on unconditionally.
    # The MCP23017's internal resistor pulls toward VDD and the chip has no
    # pull-down, so on an active-high board that holds every pin high forever
    # and read_raw() reports a permanently full board -- a fault that would
    # look like a dead sensor grid and take a day to find on the bench.
    bus = FakeI2CBus(default=0xFF)
    occupancy.Mcp23017OccupancySensor(bus=bus, active_low=False)
    for address in occupancy.EXPANDER_ADDRESSES:
        written = dict((reg, val) for (addr, reg, val) in bus.writes if addr == address)
        assert written[occupancy._REG_GPPUA] == 0x00
        assert written[occupancy._REG_GPPUB] == 0x00
        # Still inputs, though.
        assert written[occupancy._REG_IODIRA] == 0xFF
        assert written[occupancy._REG_IODIRB] == 0xFF


def test_a_port_read_wider_than_a_byte_cannot_spill_onto_other_squares():
    # A bus that hands back more than eight bits must not corrupt the squares
    # belonging to the next expander, and must never produce a mask that does
    # not fit on a chessboard.
    bus = FakeI2CBus(default=0x00)
    bus.registers[(occupancy.EXPANDER_ADDRESSES[0], occupancy._REG_GPIOB)] = 0xFFFF
    sensor = occupancy.Mcp23017OccupancySensor(bus=bus, active_low=False)
    mask = sensor.read_raw()
    assert mask <= occupancy.FULL_BOARD_MASK
    assert mask == 0xFF00        # squares 8..15 only, nothing above them


def test_closing_leaves_a_bus_somebody_else_owns_open():
    # A bus handed in from outside may be shared with other devices, so the
    # sensor must not close it. Only a bus this object opened is its to close.
    bus = FakeI2CBus()
    sensor = occupancy.Mcp23017OccupancySensor(bus=bus)
    sensor.close()
    assert not bus.closed


def test_real_sensor_rejects_a_set_of_expanders_that_cannot_cover_the_board():
    with pytest.raises(ValueError):
        occupancy.Mcp23017OccupancySensor(bus=FakeI2CBus(), addresses=(0x20, 0x21))


# ---------------------------------------------------------------------------
# The debouncer -- the substantive part
# ---------------------------------------------------------------------------

def test_default_hold_time_comes_from_config():
    assert occupancy.Debouncer().hold_s == config.SENSOR_DEBOUNCE_S


def test_default_clock_is_monotonic_not_wall_clock():
    # Not a style preference. The Pi has no battery-backed clock, so time.time()
    # can jump by years when NTP syncs and would break the debounce outright.
    assert occupancy.Debouncer().time_fn is time.monotonic


def test_a_mask_that_changes_then_holds_is_reported_exactly_once():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)

    assert debouncer.update(START_MASK, now=0.0) is None
    assert debouncer.update(START_MASK, now=hold) == START_MASK
    # Same board, later polls: nothing new to say.
    assert debouncer.update(START_MASK, now=hold + 1.0) is None
    assert debouncer.update(START_MASK, now=hold + 2.0) is None


def test_a_second_change_is_reported_too():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    moved = START_MASK & ~(1 << chess.E2) | (1 << chess.E4)

    debouncer.update(START_MASK, now=0.0)
    assert debouncer.update(START_MASK, now=hold) == START_MASK
    assert debouncer.update(moved, now=1.0) is None
    assert debouncer.update(moved, now=1.0 + hold) == moved


def test_a_flickering_square_is_never_reported():
    # A hand hovering over one square makes it chatter. Poll at the real rate
    # for a full second and check nothing is ever believed.
    hold = config.SENSOR_DEBOUNCE_S
    poll_s = 1.0 / config.SENSOR_POLL_HZ
    assert poll_s < hold, "this test only means something if we poll faster than we hold"

    debouncer = occupancy.Debouncer(hold_s=hold)
    flicker = START_MASK ^ (1 << chess.E4)
    now = 0.0
    for i in range(int(1.0 / poll_s)):
        mask = START_MASK if i % 2 == 0 else flicker
        assert debouncer.update(mask, now=now) is None
        now += poll_s


def test_a_change_before_the_hold_expires_restarts_the_timer():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    other = START_MASK ^ (1 << chess.D2)

    debouncer.update(START_MASK, now=0.0)
    # Changes with 10% of the hold left to run.
    assert debouncer.update(other, now=hold * 0.9) is None
    # The moment the FIRST mask would have qualified: still nothing, because the
    # clock restarted.
    assert debouncer.update(other, now=hold) is None
    assert debouncer.update(other, now=hold * 0.9 + hold * 0.99) is None
    assert debouncer.update(other, now=hold * 0.9 + hold) == other


def test_just_before_the_threshold_nothing_is_reported():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    debouncer.update(START_MASK, now=0.0)
    assert debouncer.update(START_MASK, now=hold - 1e-9) is None


def test_exactly_at_the_threshold_the_mask_is_reported():
    # Held for exactly the hold time counts as held long enough: the comparison
    # is ">=", and at 20 Hz a sample lands right on the boundary often enough
    # that the choice has to be deliberate.
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    debouncer.update(START_MASK, now=0.0)
    assert debouncer.update(START_MASK, now=hold) == START_MASK


def test_just_after_the_threshold_the_mask_is_reported():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    debouncer.update(START_MASK, now=0.0)
    assert debouncer.update(START_MASK, now=hold + 1e-9) == START_MASK


def test_the_boundary_survives_a_clock_that_did_not_start_at_zero():
    # Regression guard. Written as "now - started < hold_s" this fails: with a
    # start time of 10.0, (10.0 + 0.1) - 10.0 comes out as 0.09999999999999964
    # in binary floating point, a hair under the hold time, and the sample that
    # landed exactly on the boundary gets thrown away. Comparing against a
    # deadline computed once avoids it.
    hold = config.SENSOR_DEBOUNCE_S
    for started in [0.0, 10.0, 1234.5, 86400.0]:
        debouncer = occupancy.Debouncer(hold_s=hold)
        assert debouncer.update(START_MASK, now=started) is None
        assert debouncer.update(START_MASK, now=started + hold) == START_MASK


def test_the_clock_is_taken_from_time_fn_when_now_is_left_out():
    clock = FakeClock()
    debouncer = occupancy.Debouncer(hold_s=config.SENSOR_DEBOUNCE_S, time_fn=clock)

    assert debouncer.update(START_MASK) is None
    clock.advance(config.SENSOR_DEBOUNCE_S / 2.0)
    assert debouncer.update(START_MASK) is None
    clock.advance(config.SENSOR_DEBOUNCE_S)
    assert debouncer.update(START_MASK) == START_MASK


def test_a_piece_slid_across_the_board_reports_only_where_it_stops():
    # Physical rule 1 in miniature: dragging a rook from a1 to a4 by hand trips
    # a2 and a3 on the way. Only the final resting mask should ever come out.
    clock = FakeClock()
    debouncer = occupancy.Debouncer(hold_s=config.SENSOR_DEBOUNCE_S, time_fn=clock)
    poll_s = 1.0 / config.SENSOR_POLL_HZ

    base = occupancy.mask_from_squares([chess.E1, chess.E8])
    reports = []
    for square in [chess.A1, chess.A2, chess.A3]:
        reports.append(debouncer.update(base | (1 << square)))
        clock.advance(poll_s)

    settled = base | (1 << chess.A4)
    for _ in range(10):
        reports.append(debouncer.update(settled))
        clock.advance(poll_s)

    assert [r for r in reports if r is not None] == [settled]


def test_reset_throws_away_a_reading_taken_while_the_gantry_was_moving():
    # Physical rule 3: our own carriage magnet reads as a piece, so anything
    # sampled mid-move is garbage and must not count toward the hold time.
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    garbage = START_MASK | (1 << chess.E4)

    debouncer.update(garbage, now=0.0)
    debouncer.reset()
    # Without the reset this poll would have satisfied the hold time.
    assert debouncer.update(garbage, now=hold) is None
    assert debouncer.update(garbage, now=hold * 2) == garbage


def test_reset_lets_an_unchanged_board_be_reported_again():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    debouncer.update(START_MASK, now=0.0)
    assert debouncer.update(START_MASK, now=hold) == START_MASK

    debouncer.reset()
    debouncer.update(START_MASK, now=10.0)
    assert debouncer.update(START_MASK, now=10.0 + hold) == START_MASK


def test_settling_says_whether_a_reading_is_still_in_the_air():
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    assert not debouncer.settling()
    debouncer.update(START_MASK, now=0.0)
    assert debouncer.settling()
    debouncer.update(START_MASK, now=hold)
    assert not debouncer.settling()


def test_the_debouncer_never_sleeps_so_a_whole_game_of_polling_is_instant():
    # Guards two design properties at once. 20 Hz for ten minutes is 12000
    # polls: update() must do arithmetic and nothing else (the wall-clock
    # assertion), and an unchanging board must be announced ONCE and then stay
    # quiet, or the state machine drowns in repeats of news it already has.
    clock = FakeClock()
    debouncer = occupancy.Debouncer(time_fn=clock)
    started = time.monotonic()
    reports = []
    for _ in range(12000):
        report = debouncer.update(START_MASK)
        if report is not None:
            reports.append(report)
        clock.advance(1.0 / config.SENSOR_POLL_HZ)
    assert reports == [START_MASK]
    assert time.monotonic() - started < 1.0


def test_an_empty_board_is_reported_rather_than_swallowed():
    # Mask 0 is falsy, so anything written as "if self._candidate:" instead of
    # "if self._candidate is not None:" would silently refuse to ever report an
    # empty board -- exactly the reading SETUP_CHECK gets when the pieces are
    # still in the box.
    hold = config.SENSOR_DEBOUNCE_S
    debouncer = occupancy.Debouncer(hold_s=hold)
    assert debouncer.update(0, now=0.0) is None
    assert debouncer.settling()
    assert debouncer.update(0, now=hold) == 0
    assert not debouncer.settling()
