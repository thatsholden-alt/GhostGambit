"""Tests for the game state machine.

Runs on a bare laptop: no hardware, no engine, no sleeping, and -- on purpose --
no import of anybody else's module. The driver, the sensor, the planner, the
inference and the engine are all tiny stubs defined right here. That is not
laziness: if the machine can be driven by four twenty-line stubs then it really
does depend only on the interface contract, which is the thing we want to prove
before four people try to plug their modules together in November.

    python -m pytest tests/test_state_machine.py -q
"""

import chess
import pytest

import config
from state_machine import GameStateMachine, State


# --- geometry the stubs need -------------------------------------------------
#
# motion/corexy.py owns the real square_to_xy(). These four lines are the same
# arithmetic, kept local so the tests import nothing of anyone else's.

PARK_XY = (config.PARK_X_MM, config.PARK_Y_MM)


def square_xy(square):
    """Centre of a python-chess square, in mm, in the board coordinate system."""
    return (chess.square_file(square) * config.SQUARE_MM,
            chess.square_rank(square) * config.SQUARE_MM)


# --- stubs -------------------------------------------------------------------


class StubDriver:
    """Stands in for motion/driver.py. Tracks position, never sleeps."""

    def __init__(self):
        self.homed = False
        self.x = 0.0
        self.y = 0.0
        self.magnet = False
        self.segments_run = []
        self.home_calls = 0
        # Set by a test to make the next segment blow up like a stalled motor.
        self.break_next_segment = False
        # Set by a test to make wait_until_idle report a timeout.
        self.time_out = False

    def home(self):
        self.home_calls += 1
        self.homed = True
        self.x = config.HOMING_OFFSET_X_MM
        self.y = config.HOMING_OFFSET_Y_MM

    def run_segment(self, segment):
        if self.break_next_segment:
            self.break_next_segment = False
            raise IOError("motor controller stopped answering")
        self.segments_run.append(segment)
        self.x, self.y = segment["points"][-1]

    def wait_until_idle(self, timeout_s=None):
        return not self.time_out

    def position(self):
        return (self.x, self.y)

    def set_magnet(self, engaged):
        self.magnet = engaged

    def is_homed(self):
        return self.homed


class StubSensor:
    """Stands in for sensing/occupancy.py, and polices when it is read.

    The interesting part is reads_while_moving. Rule 3 of the build says the
    grid must never be consulted unless the carriage is parked off the 64
    squares, so the sensor itself counts every time that rule is broken and the
    tests assert the count is zero.
    """

    def __init__(self, driver, mask=0):
        self.driver = driver
        self.mask = mask
        self.reads = 0
        self.reads_while_moving = 0

    def read_raw(self):
        self.reads += 1
        x, y = self.driver.position()
        if (abs(x - config.PARK_X_MM) > 0.5 or abs(y - config.PARK_Y_MM) > 0.5):
            self.reads_while_moving += 1
        return self.mask


HOME_XY = (config.HOMING_OFFSET_X_MM, config.HOMING_OFFSET_Y_MM)


def travel(points, why):
    return {"kind": "travel", "points": points,
            "speed_mms": config.TRAVEL_SPEED_MMS, "why": why}


def drag(points, why):
    return {"kind": "drag", "points": points,
            "speed_mms": config.DRAG_SPEED_MMS, "why": why}


class StubPlanner:
    """Stands in for motion/planner.py.

    Real lane routing is somebody else's job; all the state machine cares about
    is that it gets segment dicts in execution order and that running the last
    one leaves the carriage where that segment says it does. Like the real
    planner, every plan starts and ends at the park spot.
    """

    def __init__(self):
        self.moves_planned = []

    def plan_move(self, board, move):
        self.moves_planned.append(move)
        start = square_xy(move.from_square)
        end = square_xy(move.to_square)
        return [
            travel([PARK_XY, start], "go and get it"),
            drag([start, end], "drag the piece"),
            travel([end, PARK_XY], "park clear of the squares"),
        ]

    def plan_park(self):
        # The real one takes no argument and therefore assumes it is already
        # standing on the park spot, so this stub tells the same lie.
        return [travel([PARK_XY, PARK_XY], "park off the board")]

    def plan_home(self):
        return [travel([HOME_XY, PARK_XY], "retreat from the home corner")]


class ForgetfulPlanner(StubPlanner):
    """A planner whose plans stop dead on the destination square.

    Nothing in the interface contract actually promises that a plan ends
    parked. If one ever does not, the machine has to notice and add the hop
    itself, or the next occupancy read happens with the carriage magnet sitting
    under a square and the game quietly desynchronises.
    """

    def plan_move(self, board, move):
        self.moves_planned.append(move)
        start = square_xy(move.from_square)
        end = square_xy(move.to_square)
        return [travel([PARK_XY, start], "go and get it"),
                drag([start, end], "drag the piece and stop right there")]

    def plan_home(self):
        return []


class StubInference:
    """Stands in for sensing/inference.py, using the same robust formulation.

    Push each legal move, compare the resulting occupancy against what the grid
    saw, pop. Exactly one match is the move. It is four lines here because
    python-chess does the work; the real module also has to explain itself to
    the player, which is why it is a whole file.
    """

    def infer_move(self, board, observed_mask):
        matches = []
        for move in board.legal_moves:
            board.push(move)
            hit = board.occupied == observed_mask
            board.pop()
            if hit:
                matches.append(move)
        if len(matches) == 1:
            return matches[0], ""
        if not matches:
            return None, "no legal move leaves the board looking like that"
        return None, "ambiguous: %s" % " or ".join(m.uci() for m in matches)

    def describe_mismatch(self, board, observed_mask):
        return "%d squares disagree" % bin(board.occupied ^ observed_mask).count("1")


class StubEngine:
    """Plays the first legal move it is offered. Deterministic, which is all a
    test needs -- move quality is game/difficulty.py's problem, not ours."""

    def __init__(self):
        self.calls = 0

    def select_move(self, board):
        self.calls += 1
        return next(iter(board.legal_moves))


class ExplodingEngine:
    def select_move(self, board):
        raise RuntimeError("engine process died")


# --- helpers -----------------------------------------------------------------


def build(engine=None, board=None, planner=None):
    """A machine sitting in WAIT_HUMAN_MOVE with the pieces set up correctly."""
    board = board if board is not None else chess.Board()
    driver = StubDriver()
    sensor = StubSensor(driver, mask=board.occupied)
    planner = planner if planner is not None else StubPlanner()
    machine = GameStateMachine(driver, sensor, engine=engine, board=board,
                               planner=planner, inference=StubInference())
    machine.run_until(State.WAIT_HUMAN_MOVE)
    return machine, driver, sensor, planner


def mask_after(board, uci):
    """The occupancy the grid would report once this move has been played."""
    scratch = board.copy(stack=False)
    scratch.push(chess.Move.from_uci(uci))
    return scratch.occupied


def human_plays(machine, sensor, uci, until=State.WAIT_HUMAN_MOVE):
    """Move the wood, press the button, and run to the next resting state."""
    assert machine.state is State.WAIT_HUMAN_MOVE
    sensor.mask = mask_after(machine.board, uci)
    assert machine.press_end_turn() is True
    visited = [machine.step()]
    visited.extend(machine.run_until(until))
    return visited


# --- boot and homing ---------------------------------------------------------


def test_every_state_has_a_handler():
    # The dispatch dict is the whole design. A state with no entry would only
    # be discovered on the day the machine reached it.
    machine, _, _, _ = build()
    for state in State:
        assert state in machine._handlers


def test_power_up_homes_before_it_trusts_anything():
    driver = StubDriver()
    sensor = StubSensor(driver, mask=chess.Board().occupied)
    machine = GameStateMachine(driver, sensor, planner=StubPlanner(),
                               inference=StubInference())

    assert machine.state is State.BOOT
    visited = machine.run_until(State.WAIT_HUMAN_MOVE)

    assert visited == [State.HOMING, State.MENU, State.SETUP_CHECK,
                       State.WAIT_HUMAN_MOVE]
    assert driver.home_calls == 1
    # And it parked off the 64 squares before the first grid read.
    assert driver.position() == PARK_XY
    assert sensor.reads_while_moving == 0


def test_setup_check_holds_until_the_pieces_are_right():
    driver = StubDriver()
    sensor = StubSensor(driver, mask=0)  # empty board: nobody has set up yet
    machine = GameStateMachine(driver, sensor, planner=StubPlanner(),
                               inference=StubInference())

    machine.run_until(State.SETUP_CHECK)
    for _ in range(5):
        assert machine.step() is State.SETUP_CHECK
    assert "set the pieces up" in machine.last_error

    sensor.mask = chess.Board().occupied
    assert machine.step() is State.WAIT_HUMAN_MOVE
    assert machine.last_error == ""


# --- the happy path ----------------------------------------------------------


def test_full_happy_path_over_several_moves():
    engine = StubEngine()
    machine, driver, sensor, planner = build(engine=engine)

    for uci in ("e2e4", "d2d4", "c2c4"):
        visited = human_plays(machine, sensor, uci)
        assert visited[0] is State.VALIDATE_MOVE
        assert State.ENGINE_THINKING in visited
        assert State.EXECUTING_MOVE in visited
        assert visited[-1] is State.WAIT_HUMAN_MOVE
        assert State.ILLEGAL_RECOVERY not in visited
        assert State.ERROR not in visited

    # Three human moves and three engine replies really happened.
    assert len(machine.board.move_stack) == 6
    assert engine.calls == 3
    assert machine.turn == "white"
    assert machine.last_error == ""
    # The magnet is never left on, and the carriage always ends up parked.
    assert driver.magnet is False
    assert driver.position() == PARK_XY
    assert len(planner.moves_planned) == 3


def test_a_plan_that_forgets_to_park_still_ends_parked():
    # Rule 3 is not left to the planner's good manners: the state machine is
    # the one that gets burned if the carriage is still on a square.
    machine, driver, sensor, _ = build(engine=StubEngine(),
                                       planner=ForgetfulPlanner())
    assert driver.position() == PARK_XY          # even after a homing plan of []

    human_plays(machine, sensor, "e2e4")
    assert driver.position() == PARK_XY
    assert sensor.reads_while_moving == 0
    assert driver.magnet is False


def test_the_grid_is_never_read_while_the_carriage_is_moving():
    # Rule 3. Reading one square early is how the model silently desynchronises
    # from the wood, and every move after that is wrong.
    machine, driver, sensor, _ = build(engine=StubEngine())

    for uci in ("e2e4", "d2d4"):
        human_plays(machine, sensor, uci)

    assert sensor.reads > 0
    assert sensor.reads_while_moving == 0


def test_executing_a_move_runs_one_segment_per_step():
    # Bounded work: a capture is two full trips across the board, and the UI has
    # to keep repainting while it happens.
    machine, driver, sensor, _ = build(engine=StubEngine())
    sensor.mask = mask_after(machine.board, "e2e4")
    machine.press_end_turn()

    assert machine.step() is State.VALIDATE_MOVE
    assert machine.step() is State.ENGINE_THINKING
    assert machine.step() is State.EXECUTING_MOVE      # planning only
    assert len(driver.segments_run) == 1               # homing's park hop

    machine.step()
    assert len(driver.segments_run) == 2
    machine.step()
    assert len(driver.segments_run) == 3


def test_the_magnet_is_on_for_drags_and_off_for_travel():
    machine, driver, sensor, _ = build(engine=StubEngine())
    sensor.mask = mask_after(machine.board, "e2e4")
    machine.press_end_turn()
    machine.run_until(State.EXECUTING_MOVE)

    machine.step()                       # travel to the piece
    assert driver.magnet is False
    machine.step()                       # drag it
    assert driver.magnet is True
    machine.run_until(State.WAIT_HUMAN_MOVE)
    assert driver.magnet is False


# --- friend vs friend --------------------------------------------------------


def test_engine_none_is_friend_vs_friend_and_never_crashes():
    machine, driver, sensor, planner = build(engine=None)
    assert machine.mode == "friend"

    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        visited = human_plays(machine, sensor, uci)
        assert State.ENGINE_THINKING not in visited
        assert State.EXECUTING_MOVE not in visited
        assert State.ERROR not in visited

    assert len(machine.board.move_stack) == 4
    assert machine.turn == "white"
    assert machine.engine_to_move() is False
    # Nobody asked the gantry to move a thing beyond the initial park hop.
    assert len(planner.moves_planned) == 0
    assert len(driver.segments_run) == 1


def test_turn_flips_with_the_board():
    machine, _, sensor, _ = build(engine=None)
    assert machine.turn == "white"
    human_plays(machine, sensor, "e2e4")
    assert machine.turn == "black"


def test_the_game_ends_when_the_position_ends():
    # Fool's mate, played by two humans.
    machine, _, sensor, _ = build(engine=None)
    for uci in ("f2f3", "e7e5", "g2g4"):
        human_plays(machine, sensor, uci)

    human_plays(machine, sensor, "d8h4", until=State.GAME_OVER)
    assert machine.state is State.GAME_OVER
    assert "checkmate" in machine.result
    assert "0-1" in machine.result

    # And it stays there until somebody starts another game.
    for _ in range(3):
        assert machine.step() is State.GAME_OVER


def test_a_new_game_re_homes():
    machine, driver, sensor, _ = build(engine=None)
    driver.homed = False        # power was cut to the steppers, say

    assert machine.new_game() is State.BOOT
    assert len(machine.board.move_stack) == 0
    visited = machine.run_until(State.WAIT_HUMAN_MOVE)
    assert State.HOMING in visited
    assert driver.home_calls == 2
    assert driver.is_homed() is True


# --- illegal moves -----------------------------------------------------------


def test_an_illegal_move_is_dragged_back_and_play_resumes():
    machine, driver, sensor, planner = build(engine=StubEngine())

    # The human slides the e-pawn three squares, to e5.
    sensor.mask = (machine.board.occupied & ~(1 << chess.E2)) | (1 << chess.E5)
    machine.press_end_turn()
    visited = [machine.step()]
    visited.extend(machine.run_until(State.WAIT_HUMAN_MOVE))

    assert visited[0] is State.VALIDATE_MOVE
    assert State.ILLEGAL_RECOVERY in visited
    assert State.ENGINE_THINKING not in visited
    assert visited[-1] is State.WAIT_HUMAN_MOVE

    # The position is untouched and it is still the same player's turn.
    assert len(machine.board.move_stack) == 0
    assert machine.turn == "white"
    assert machine.last_error != ""

    # The gantry was asked to put the pawn back on e2, and then to park.
    assert planner.moves_planned == [chess.Move(chess.E5, chess.E2)]
    drags = [s for s in driver.segments_run if s["kind"] == "drag"]
    assert drags[-1]["points"][-1] == square_xy(chess.E2)
    assert driver.position() == PARK_XY
    assert driver.magnet is False


def test_recovery_leaves_the_grid_alone_until_the_carriage_parks():
    machine, driver, sensor, _ = build(engine=None)
    sensor.mask = (machine.board.occupied & ~(1 << chess.B1)) | (1 << chess.B5)
    machine.press_end_turn()
    machine.step()
    machine.run_until(State.WAIT_HUMAN_MOVE)
    assert sensor.reads_while_moving == 0


def test_a_mess_the_gantry_cannot_fix_is_handed_back_to_the_human():
    # Two pieces lifted and not put back. There is no single drag that undoes
    # that, and guessing with a magnet under the board makes it worse.
    machine, driver, sensor, planner = build(engine=None)
    sensor.mask = machine.board.occupied & ~(1 << chess.E2) & ~(1 << chess.D2)
    before = len(driver.segments_run)
    machine.press_end_turn()
    machine.step()
    machine.run_until(State.WAIT_HUMAN_MOVE)

    assert planner.moves_planned == []
    assert driver.segments_run[before:] == []
    assert "by hand" in machine.last_error
    assert len(machine.board.move_stack) == 0


def test_play_carries_on_normally_after_a_recovery():
    machine, driver, sensor, _ = build(engine=StubEngine())
    sensor.mask = (machine.board.occupied & ~(1 << chess.E2)) | (1 << chess.E5)
    machine.press_end_turn()
    machine.step()
    machine.run_until(State.WAIT_HUMAN_MOVE)

    visited = human_plays(machine, sensor, "e2e4")
    assert State.EXECUTING_MOVE in visited
    assert len(machine.board.move_stack) == 2
    assert machine.last_error == ""


# --- faults and recovery -----------------------------------------------------


def test_a_hardware_fault_while_executing_lands_in_error_and_recovers():
    machine, driver, sensor, _ = build(engine=StubEngine())
    sensor.mask = mask_after(machine.board, "e2e4")
    machine.press_end_turn()
    machine.run_until(State.EXECUTING_MOVE)

    driver.break_next_segment = True
    assert machine.step() is State.ERROR
    assert "motor controller stopped answering" in machine.last_error
    assert driver.magnet is False

    # ERROR is sticky until somebody acknowledges it.
    for _ in range(3):
        assert machine.step() is State.ERROR

    driver.homed = False
    machine.acknowledge_error()
    visited = machine.run_until(State.WAIT_HUMAN_MOVE)
    assert visited[0] is State.HOMING
    assert driver.home_calls == 2
    assert driver.is_homed() is True

    # The engine's move never happened physically, so it was not recorded; the
    # machine re-decided it after homing and finished it.
    assert len(machine.board.move_stack) == 2
    assert machine.state is State.WAIT_HUMAN_MOVE


def test_an_injected_fault_reaches_error_from_any_state():
    for target in (State.WAIT_HUMAN_MOVE, State.EXECUTING_MOVE):
        machine, driver, sensor, _ = build(engine=StubEngine())
        if target is State.EXECUTING_MOVE:
            sensor.mask = mask_after(machine.board, "e2e4")
            machine.press_end_turn()
            machine.run_until(State.EXECUTING_MOVE)

        machine.inject_fault("e-stop pressed")
        assert machine.step() is State.ERROR
        assert machine.last_error == "e-stop pressed"
        assert driver.magnet is False


def test_a_motion_timeout_is_a_fault():
    machine, driver, sensor, _ = build(engine=StubEngine())
    sensor.mask = mask_after(machine.board, "e2e4")
    machine.press_end_turn()
    machine.run_until(State.EXECUTING_MOVE)

    driver.time_out = True
    assert machine.step() is State.ERROR
    assert "did not finish" in machine.last_error


def test_an_engine_that_dies_is_a_fault_not_a_crash():
    machine, driver, sensor, _ = build(engine=ExplodingEngine())
    sensor.mask = mask_after(machine.board, "e2e4")
    machine.press_end_turn()
    machine.run_until(State.ENGINE_THINKING)
    assert machine.step() is State.ERROR
    assert "engine process died" in machine.last_error


def test_reading_the_grid_off_park_becomes_an_error():
    # The interlock that keeps rule 3 honest even if a planner forgets to park.
    machine, driver, sensor, _ = build(engine=None)
    machine.press_end_turn()
    assert machine.step() is State.VALIDATE_MOVE

    driver.x, driver.y = square_xy(chess.D4)   # stranded mid-board
    reads_before = sensor.reads
    assert machine.step() is State.ERROR
    assert "not parked" in machine.last_error
    assert sensor.reads == reads_before        # it refused, it did not peek


# --- inputs ------------------------------------------------------------------


def test_the_end_turn_button_is_ignored_unless_it_is_your_turn_to_press_it():
    machine, driver, sensor, _ = build(engine=StubEngine())
    sensor.mask = mask_after(machine.board, "e2e4")
    machine.press_end_turn()
    machine.run_until(State.EXECUTING_MOVE)

    # An impatient jab while the gantry is still dragging must not be saved up
    # and used to end the next turn before the human has touched a piece.
    assert machine.press_end_turn() is False
    machine.run_until(State.WAIT_HUMAN_MOVE)
    for _ in range(3):
        assert machine.step() is State.WAIT_HUMAN_MOVE


def test_run_until_gives_up_loudly_rather_than_quietly():
    machine, _, sensor, _ = build(engine=None)
    with pytest.raises(RuntimeError):
        machine.run_until(State.GAME_OVER, max_steps=20)


def test_run_until_returns_immediately_if_already_there():
    machine, _, _, _ = build(engine=None)
    assert machine.run_until(State.WAIT_HUMAN_MOVE) == []
