"""The state machine that runs a whole game.

One function per state, one dispatch dict, nothing else. The board is always in
exactly one state and you can print it: self.state. Every reference build we
studied did this with a pile of loose booleans instead -- one of them has a
Game_Init() that hands back a fourteen-element tuple of flags, duplicated
verbatim in two files -- and none of them can answer "what is the board doing
right now?" with a single value.

How it is driven: something outside calls step() over and over. step() does a
small, BOUNDED amount of work and returns the new state. It never blocks waiting
for a human and never loops forever, because there is no event loop, no threads
and no async anywhere in this project. The UI polls; that is the whole design.

    BOOT -> HOMING -> MENU -> SETUP_CHECK -> WAIT_HUMAN_MOVE
    WAIT_HUMAN_MOVE --(end-turn button)--> VALIDATE_MOVE
        VALIDATE_MOVE --legal--> ENGINE_THINKING -> EXECUTING_MOVE -> WAIT_HUMAN_MOVE
        VALIDATE_MOVE --illegal--> ILLEGAL_RECOVERY -> WAIT_HUMAN_MOVE
    any state --fault / timeout / e-stop--> ERROR --acknowledged--> HOMING
    -> GAME_OVER as soon as the position is finished

Everything the outside world can do to the machine is a method call, never a
poke at an attribute: press_end_turn(), inject_fault(), acknowledge_error(),
new_game(). That is what makes the whole thing testable with no hardware.

The machine does not import motion/driver.py or sensing/occupancy.py: it is
handed a driver and a sensor and only ever calls the methods in the interface
contract. It does need the planner and the move inference, and those are
imported late (see _get_planner) so that this file imports on a checkout where
they do not exist yet.
"""

import enum

import chess

import config


# How far from the park point still counts as "parked", in mm. Belt stretch and
# the driver's own rounding mean the reported position is never exactly the park
# point. One millimetre is far tighter than the ~28 mm it would take for the
# carriage magnet to reach under a real square, so this is a safe fence.
# THIS SHOULD MOVE TO config.py once someone measures the real repeatability.
PARK_TOLERANCE_MM = 1.0


def _is_parked(xy):
    """True when a point is the park spot, allowing for belt slop."""
    x, y = xy
    return (abs(x - config.PARK_X_MM) <= PARK_TOLERANCE_MM
            and abs(y - config.PARK_Y_MM) <= PARK_TOLERANCE_MM)


class State(enum.Enum):
    """Every situation the board can be in. Exactly one is true at a time.

    enum.Enum is a fixed, named list of values. State.MENU is not the string
    "MENU" and not the number 3 -- it is its own object, so a typo like
    State.MENUE fails immediately with an AttributeError instead of quietly
    matching nothing. Compare them with `is`.
    """

    BOOT = "boot"
    HOMING = "homing"
    MENU = "menu"
    SETUP_CHECK = "setup_check"
    WAIT_HUMAN_MOVE = "wait_human_move"
    VALIDATE_MOVE = "validate_move"
    ENGINE_THINKING = "engine_thinking"
    EXECUTING_MOVE = "executing_move"
    ILLEGAL_RECOVERY = "illegal_recovery"
    GAME_OVER = "game_over"
    ERROR = "error"


class GameStateMachine:
    """Runs one game, one bounded step at a time.

    driver  -- anything with the motion/driver.py MotionDriver interface
    sensor  -- anything with the sensing/occupancy.py OccupancySensor interface
    engine  -- an object with select_move(board), or None for friend-vs-friend
    board   -- a chess.Board, or None for a fresh game
    """

    def __init__(self, driver, sensor, engine=None, board=None,
                 planner=None, inference=None, engine_color=chess.BLACK):
        self.driver = driver
        self.sensor = sensor
        self.engine = engine
        self.board = board if board is not None else chess.Board()

        # Which colour the robot plays. The human is White by default, so the
        # human moves first and the engine answers, which is the sequence the
        # state diagram describes.
        self.engine_color = engine_color

        # Everything the UI is allowed to read.
        self.state = State.BOOT
        self.last_error = ""
        self.result = ""
        self.mode = ""

        # The two collaborators we cannot be handed by the caller. Left as None
        # they are imported on first use; tests pass their own stubs in.
        self._planner = planner
        self._inference = inference

        # Latched inputs from the physical world. These are not state -- they
        # are buttons, and the state that cares about one consumes it.
        self._end_turn_pressed = False
        self._pending_fault = None
        self._error_acknowledged = False

        # Work in progress: planned segments still to run, and the move that
        # becomes real once the last of them has run.
        self._segments = []
        self._pending_move = None

        # The dispatch table. One entry per state, no exceptions, and the check
        # below proves it -- a missing handler would otherwise only show up on
        # the day the machine happened to reach that state.
        self._handlers = {
            State.BOOT: self._on_boot,
            State.HOMING: self._on_homing,
            State.MENU: self._on_menu,
            State.SETUP_CHECK: self._on_setup_check,
            State.WAIT_HUMAN_MOVE: self._on_wait_human_move,
            State.VALIDATE_MOVE: self._on_validate_move,
            State.ENGINE_THINKING: self._on_engine_thinking,
            State.EXECUTING_MOVE: self._on_executing_move,
            State.ILLEGAL_RECOVERY: self._on_illegal_recovery,
            State.GAME_OVER: self._on_game_over,
            State.ERROR: self._on_error,
        }
        missing = [s.name for s in State if s not in self._handlers]
        if missing:
            raise RuntimeError("no handler for state(s): %s" % ", ".join(missing))

    # -- what the UI reads ---------------------------------------------------

    @property
    def turn(self):
        """"white" or "black" -- whose move it is, for the screen.

        A @property is a method that is read like a plain attribute: you write
        machine.turn, not machine.turn(). It is used here so the UI treats
        .state, .last_error and .turn the same way.
        """
        return "white" if self.board.turn == chess.WHITE else "black"

    def engine_to_move(self):
        """True when the robot owes us a move. Always False with engine=None."""
        return self.engine is not None and self.board.turn == self.engine_color

    # -- what the outside world does to the machine --------------------------

    def press_end_turn(self):
        """The human pressed the end-turn button. True if the press counted.

        A press outside WAIT_HUMAN_MOVE is dropped rather than remembered.
        Someone jabbing the button while the gantry is still dragging must not
        end their turn a moment after they get the board back, before they have
        touched a piece.
        """
        if self.state is not State.WAIT_HUMAN_MOVE:
            return False
        self._end_turn_pressed = True
        return True

    def inject_fault(self, message):
        """Report a hardware fault, a motion timeout or an e-stop.

        The next step() drops the magnet and goes to ERROR from whatever state
        it was in. The e-stop wiring and the serial layer call this; tests call
        it directly, which is the point -- nobody reaches into attributes.
        """
        self._pending_fault = str(message)

    def acknowledge_error(self):
        """The operator cleared the fault, so the next step() re-homes."""
        self._error_acknowledged = True

    def new_game(self, board=None):
        """Throw the position away and start again, homing on the way.

        We re-home between games because a game's worth of drags is a game's
        worth of chances to have lost a step without noticing.
        """
        self.board = board if board is not None else chess.Board()
        self.last_error = ""
        self.result = ""
        self._segments = []
        self._pending_move = None
        self._end_turn_pressed = False
        self._pending_fault = None
        self._error_acknowledged = False
        self.state = State.BOOT
        return self.state

    # -- the pump ------------------------------------------------------------

    def step(self):
        """Do one bounded chunk of work and return the new state."""
        if self._pending_fault is not None:
            message = self._pending_fault
            self._pending_fault = None
            self.state = self._fail(message)
            return self.state

        handler = self._handlers[self.state]
        try:
            next_state = handler()
        except Exception as exc:
            # A hardware fault arrives as an exception out of the driver, and a
            # traceback on the screen in front of the review panel is worse than
            # an ERROR state someone can acknowledge. The message is kept, so
            # nothing is swallowed silently.
            next_state = self._fail("%s: %s" % (type(exc).__name__, exc))
        self.state = next_state
        return self.state

    def run_until(self, state, max_steps=1000):
        """Step until the machine reaches `state`; return the states it entered.

        Returns [] if the machine is already there -- it does not step first.
        Raises if `state` is not reached within max_steps, because a machine
        that is stuck should be loud rather than quietly give up.
        """
        visited = []
        steps = 0
        while self.state is not state:
            if steps >= max_steps:
                raise RuntimeError(
                    "did not reach %s in %d steps (stuck in %s; last error: %r)"
                    % (state.name, max_steps, self.state.name, self.last_error))
            visited.append(self.step())
            steps += 1
        return visited

    # -- one function per state ----------------------------------------------

    def _on_boot(self):
        """Power-up. Nothing is known about the carriage, so go and find out."""
        self.last_error = ""
        return State.HOMING

    def _on_homing(self):
        """Run the homing cycle, then get the carriage off the 64 squares."""
        self.driver.home()
        self._wait_for_carriage("homing cycle")
        if not self.driver.is_homed():
            return self._fail("driver still reports not homed after homing")

        # The homing cycle leaves the carriage on the home reference point,
        # which is on top of a1 -- exactly where rule 3 says it must not sit,
        # because our own magnet reads as a piece. plan_home() is the retreat
        # from there to the park spot. It rides along inside this step rather
        # than getting steps of its own: the homing cycle it follows is one
        # blocking hardware move anyway, so there is no responsiveness left here
        # to protect.
        for segment in self._plan_ending_parked(self._get_planner().plan_home()):
            self._run_segment(segment)
        self.driver.set_magnet(False)
        return State.MENU

    def _on_menu(self):
        """Note what kind of game this is and move on.

        There is no waiting here. The machine is polled, so while the player is
        still reading the menu the UI simply does not call step(). Note what is
        NOT done: the position is not reset. HOMING comes through here after
        every error, and wiping the board mid-game because a belt slipped would
        be far worse than the belt slipping.
        """
        self.mode = "friend" if self.engine is None else "engine"
        return State.SETUP_CHECK

    def _on_setup_check(self):
        """Hold here until the wood on the board matches the position we hold."""
        observed = self._read_occupancy()

        # python-chess keeps board.occupied as a 64-bit int with bit i set when
        # square i has a piece on it, a1 as bit 0 -- exactly the shape the sensor
        # grid returns, so the two compare directly.
        if observed == self.board.occupied:
            self.last_error = ""
            return self._after_position_changed()

        self.last_error = "set the pieces up: %s" % self._describe_mismatch(observed)
        return State.SETUP_CHECK

    def _on_wait_human_move(self):
        """Wait for the end-turn button.

        Deliberately does not read the grid. A hand is over the board and pieces
        are in mid-air; the grid says nothing worth having until the human tells
        us they are finished.
        """
        if not self._end_turn_pressed:
            return State.WAIT_HUMAN_MOVE
        self._end_turn_pressed = False
        return State.VALIDATE_MOVE

    def _on_validate_move(self):
        """Work out what the human did, and accept it or undo it."""
        observed = self._read_occupancy()
        move, reason = self._get_inference().infer_move(self.board, observed)

        if move is None:
            self.last_error = reason or self._describe_mismatch(observed)
            # Plan the drag-back now, while the board still holds the position
            # the human started from and we still have the observed mask.
            self._segments = self._plan_put_it_back(observed)
            return State.ILLEGAL_RECOVERY

        # The human already moved the wood; the model is just catching up.
        self.board.push(move)
        self.last_error = ""
        return self._after_position_changed()

    def _on_engine_thinking(self):
        """Ask the engine for a move and plan the motion that plays it."""
        move = self._ask_engine()
        if move is None or move not in self.board.legal_moves:
            return self._fail("engine did not return a legal move (%r)" % (move,))

        # Plan before pushing: the planner needs the position the pieces are
        # physically standing in. The push happens once the wood has moved.
        self._segments = self._plan_ending_parked(
            self._get_planner().plan_move(self.board, move))
        self._pending_move = move
        return State.EXECUTING_MOVE

    def _on_executing_move(self):
        """Run exactly one planned segment per step.

        One per step, not all of them in a loop, so the UI keeps repainting and
        the e-stop keeps being noticed during a capture, which is two full trips
        across the board. The grid is not read anywhere in here: the carriage is
        under the squares and would read as a piece on every one it crosses.
        """
        if self._segments:
            self._run_segment(self._segments.pop(0))
            return State.EXECUTING_MOVE

        self.driver.set_magnet(False)
        if self._pending_move is not None:
            self.board.push(self._pending_move)
            self._pending_move = None
        return self._after_position_changed()

    def _on_illegal_recovery(self):
        """Drag the human's piece back where it came from, then let them retry.

        No lights and no beeps: both reference builds just put the piece back,
        and a piece sliding home on its own is a clearer message than an LED.
        Same rule as EXECUTING_MOVE -- one segment per step, grid untouched.
        """
        if self._segments:
            self._run_segment(self._segments.pop(0))
            return State.ILLEGAL_RECOVERY

        self.driver.set_magnet(False)
        return State.WAIT_HUMAN_MOVE

    def _on_game_over(self):
        """Sit still. Only new_game() leaves this state, and it re-homes."""
        return State.GAME_OVER

    def _on_error(self):
        """Wait for a human to acknowledge, then re-home.

        Always re-home: after a fault nobody knows where the carriage is, and
        the position the driver reports is a guess left over from before.
        """
        if not self._error_acknowledged:
            return State.ERROR
        self._error_acknowledged = False
        return State.HOMING

    # -- helpers -------------------------------------------------------------

    def _after_position_changed(self):
        """Pick the next state now that the board holds a new position."""
        outcome = self.board.outcome()
        if outcome is not None:
            # game/rules.py has is_game_over() saying the same thing for the UI.
            # This asks python-chess directly so the machine has no import-time
            # dependency on a file someone else is still writing.
            self.result = "%s (%s)" % (
                outcome.termination.name.replace("_", " ").lower(),
                outcome.result())
            return State.GAME_OVER

        if self.engine_to_move():
            return State.ENGINE_THINKING
        return State.WAIT_HUMAN_MOVE

    def _run_segment(self, segment):
        """Send one planned segment to the driver and wait for it to finish."""
        # "drag" means magnet on with a piece following; "travel" means the
        # carriage is empty. Getting this backwards drops a piece mid-board or
        # ploughs one across four squares.
        self.driver.set_magnet(segment["kind"] == "drag")
        self.driver.run_segment(segment)
        self._wait_for_carriage(segment.get("why", segment["kind"]))

    def _wait_for_carriage(self, what):
        """Block until the carriage stops, with a timeout that cannot be lost."""
        finished = self.driver.wait_until_idle(timeout_s=config.MOTION_TIMEOUT_S)
        # Drivers signal a timeout one of two ways: raising, or returning False.
        # Only an actual False counts -- returning None on success is the common
        # case and must not be mistaken for a timeout.
        if finished is False:
            raise RuntimeError(
                "carriage did not finish %r within %.0f s"
                % (what, config.MOTION_TIMEOUT_S))

    def _plan_ending_parked(self, segments):
        """Return a plan that is guaranteed to finish off the 64 squares.

        motion/planner.py already parks at the end of every plan it makes, so
        normally this hands the plan straight back. It checks anyway because of
        rule 3: a plan that stops with the carriage under a square makes the
        very next occupancy read a lie, and a lie there poisons the rest of the
        game. Cheap check, expensive failure.
        """
        segments = list(segments)
        if segments and _is_parked(segments[-1]["points"][-1]):
            return segments
        segments.extend(self._get_planner().plan_park())
        return segments

    def _read_occupancy(self):
        """Read the sensor grid, but only with the carriage parked off the board.

        Rule 3 of this build: the grid is ignored while the gantry is moving,
        because our own carriage magnet passing underneath reads as a piece.
        A single early read desynchronises the model from the wood and every
        move after it is wrong, so this refuses loudly instead of returning a
        perfectly plausible lie.
        """
        x, y = self.driver.position()
        if not _is_parked((x, y)):
            raise RuntimeError(
                "refusing to read the sensor grid: carriage is at (%.1f, %.1f), "
                "not parked at (%.1f, %.1f)"
                % (x, y, config.PARK_X_MM, config.PARK_Y_MM))
        return self.sensor.read_raw()

    def _plan_put_it_back(self, observed):
        """Segments that drag a misplaced piece back to the square it left.

        Only the simple case is fixable by machine: exactly one square emptied
        and exactly one square filled, so there is no doubt which piece went
        where. Anything messier -- a piece lifted and pocketed, two pieces
        moved, a capture attempted -- is handed back to the human, because
        guessing with a magnet under the board turns one mistake into four.
        """
        emptied = self.board.occupied & ~observed
        filled = observed & ~self.board.occupied
        if bin(emptied).count("1") != 1 or bin(filled).count("1") != 1:
            self.last_error += " -- put the pieces back by hand"
            return []

        # A mask with a single bit set: the bit's position IS the square number,
        # because a1 is bit 0. bit_length() gives the position of the top bit.
        origin = emptied.bit_length() - 1
        wrong = filled.bit_length() - 1

        piece = self.board.piece_at(origin)
        if piece is None:
            self.last_error += " -- put the pieces back by hand"
            return []

        if (piece.piece_type == chess.KING
                and abs(chess.square_file(origin) - chess.square_file(wrong)) > 1):
            # A king standing two or more files from home looks exactly like a
            # castle to python-chess whatever the rooks are doing, so asking the
            # planner to move it would very likely swing a rook across as well.
            self.last_error += " -- put the king back by hand"
            return []

        # The planner plans from a position, so hand it the position the board
        # is physically in right now: the piece on the wrong square, its own
        # colour to move. Then the move to plan is simply wrong -> origin.
        scratch = self.board.copy(stack=False)
        scratch.remove_piece_at(origin)
        scratch.set_piece_at(wrong, piece)
        scratch.turn = piece.color

        return self._plan_ending_parked(
            self._get_planner().plan_move(scratch, chess.Move(wrong, origin)))

    def _ask_engine(self):
        """Ask the engine for a move and return a chess.Move.

        game/engine.py is not finished, so accept either name it is likely to
        use. getattr(obj, "name", None) looks an attribute up by its string name
        and hands back None instead of raising when there is not one.
        """
        for name in ("select_move", "play"):
            ask = getattr(self.engine, name, None)
            if ask is not None:
                answer = ask(self.board)
                # python-chess's own engine wrapper answers with a PlayResult
                # that carries .move; a plainer object just answers with a move.
                return getattr(answer, "move", answer)
        raise RuntimeError("engine object has no select_move(board) method")

    def _describe_mismatch(self, observed):
        """One line for the screen explaining how the wood and the model differ."""
        return self._get_inference().describe_mismatch(self.board, observed)

    def _fail(self, message):
        """Record a fault, make the gantry safe, and hand back ERROR."""
        self.last_error = message
        self._segments = []
        self._pending_move = None
        try:
            self.driver.set_magnet(False)
        except Exception:
            # We are already handling a fault. If the thing that broke is the
            # link to the motor controller then dropping the magnet fails too,
            # and a crash here would lose the message we are trying to show.
            pass
        return State.ERROR

    def _get_planner(self):
        """The thing that turns a move into gantry segments (motion/planner.py).

        Imported here rather than at the top of the file on purpose: the planner
        is someone else's file and may not exist on a given checkout, and this
        module has to import -- and be unit-tested -- without it. Tests pass a
        stub planner to __init__ instead.
        """
        if self._planner is None:
            from motion import planner
            self._planner = planner
        return self._planner

    def _get_inference(self):
        """The thing that names the human's move (sensing/inference.py).

        Late-imported for the same reason as the planner.
        """
        if self._inference is None:
            from sensing import inference
            self._inference = inference
        return self._inference
