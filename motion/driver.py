"""The boundary between the Pi and the motor controller.

Everything above this file thinks in millimetres and segments. Everything below
it thinks in G-code and serial bytes. This is the only file that knows both.

Three classes:

    MotionDriver        the abstract base -- the promise, no implementation
    FakeMotionDriver    pure software. Tracks position, tracks the magnet,
                        remembers every segment, finishes instantly, never
                        sleeps, needs no hardware at all
    GrblMotionDriver    the real one. Talks G-code over USB serial

WHY THE SPLIT EXISTS: the hardware does not exist until December and the
software cannot wait for it. Every module above this one is written and tested
against FakeMotionDriver, so when the gantry finally exists the only new code
in the system is GrblMotionDriver. make_motion_driver() reads
config.USE_FAKE_HARDWARE and hands back whichever one is right.

THE MOST IMPORTANT THING IN THIS FILE
-------------------------------------
A GRBL controller answers "ok" to every line it accepts. "ok" means "I parsed
that and put it in my queue". It does NOT mean "the carriage has arrived".
GRBL buffers a whole string of moves ahead of the one it is actually running.

If you treat "ok" as arrival you will start the next segment while the previous
one is still executing: the magnet turns off mid-drag, or turns on while the
carriage is under some innocent bystander piece, and pieces end up in the wrong
squares with no error message anywhere.

Arrival is a SEPARATE question. You ask it by sending the single character "?"
and reading the status report until the state word is Idle. Reports look like

    <Idle|MPos:0.000,0.000,0.000|FS:0,0>
    <Run|MPos:114.300,57.150,0.000|FS:300,0|WCO:0.000,0.000,0.000>

Verified in grbl's own source: the state words are printed in
report_realtime_status() and the realtime command bytes are #defined in
config.h (CMD_STATUS_REPORT '?', CMD_FEED_HOLD '!', CMD_CYCLE_START '~',
CMD_RESET 0x18).
    https://github.com/gnea/grbl   (grbl/report.c, grbl/protocol.c, grbl/config.h)
"""

import math
import time

import config


# ---------------------------------------------------------------------------
# Constants that ought to live in config.py
# ---------------------------------------------------------------------------
# These are real physical / protocol facts, but config.py does not have them
# yet and this module is not allowed to edit config.py. They are flagged in the
# hand-off report; move them across when someone owns config.py again.

# Homing crawls at the homing feed rate across the whole travel, so it takes far
# longer than any normal move. Timing it out at MOTION_TIMEOUT_S would abort a
# perfectly healthy home. PROVISIONAL -- measure once the frame exists.
HOMING_TIMEOUT_S = 60.0

# How the magnet is switched. GRBL has no "magnet" concept, so we borrow an
# output it does have. M8/M9 is the coolant flood pin: a plain on/off digital
# output. The spindle (M3/M5) would also work but it is a PWM pin with a
# configurable spin-up delay attached to it, which is a second thing to get
# wrong for no benefit.
MAGNET_ON_GCODE = "M8"
MAGNET_OFF_GCODE = "M9"

# How far the carriage may be from where a segment claims it starts before we
# call it a planning bug. Segments are supposed to chain head to tail; a gap
# means somebody generated a plan from a stale position.
POSITION_TOLERANCE_MM = 0.5

# GRBL feed rates are millimetres per MINUTE. Our whole codebase is millimetres
# per second. Forgetting this factor makes every drag run 60x too slow, which
# looks exactly like a stuck controller.
SECONDS_PER_MINUTE = 60.0

# Sent once when the link opens: millimetres, absolute coordinates, feed rate
# interpreted as units-per-minute. Cheap insurance against a controller left in
# inches or in relative mode by whoever was poking at it last.
GRBL_PREAMBLE = ("G21", "G90", "G94")

# A controller that will not stop talking must not spin us forever while we are
# only trying to clear its startup banner.
MAX_BANNER_LINES = 20


# The two segment kinds, spelled once so a typo is an AttributeError instead of
# a silently-never-matching string.
TRAVEL = "travel"
DRAG = "drag"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MotionError(Exception):
    """Anything the motion layer refuses to do."""


class MotionTimeout(MotionError):
    """The controller stopped answering, or never reached Idle in time.

    One reference build called a blocking readline() with no timeout. When its
    controller wedged, the whole program froze with no message and no way out
    except pulling the USB cable. Every wait in this file has a deadline.
    """


class ControllerError(MotionError):
    """The controller answered, and what it said was 'error:N' or 'ALARM'."""


# ---------------------------------------------------------------------------
# Small helpers -- pure functions, no hardware, easy to test
# ---------------------------------------------------------------------------

def segment_length_mm(points):
    """Total path length through a list of (x, y) points, in millimetres.

    motion/corexy.py has a path_length_mm() that does the same thing. This file
    deliberately does not import it: the driver must keep working even if the
    rest of motion/ is half-written, and a driver that cannot be imported is a
    driver nobody can test.
    """
    total = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def check_segment(segment):
    """Validate a segment dict and fill in the optional bits.

    Returns (kind, points, speed_mms, why). Raises ValueError on a segment that
    could not possibly be executed, because a malformed segment is a bug in the
    planner and the loudest possible failure during a test run is the cheapest
    place to find it.
    """
    if not isinstance(segment, dict):
        raise ValueError("segment must be a dict, got %r" % (type(segment).__name__,))

    kind = segment.get("kind")
    if kind not in (TRAVEL, DRAG):
        raise ValueError("segment kind must be %r or %r, got %r" % (TRAVEL, DRAG, kind))

    raw_points = segment.get("points")
    if not raw_points:
        raise ValueError("segment has no points")

    points = []
    for point in raw_points:
        if len(point) != 2:
            raise ValueError("segment point must be (x, y), got %r" % (point,))
        points.append((float(point[0]), float(point[1])))

    # A missing speed is not worth refusing to move over: the kind already says
    # which of the two speeds it must be.
    speed_mms = segment.get("speed_mms")
    if speed_mms is None:
        speed_mms = config.DRAG_SPEED_MMS if kind == DRAG else config.TRAVEL_SPEED_MMS
    speed_mms = float(speed_mms)
    if speed_mms <= 0.0:
        raise ValueError("segment speed_mms must be positive, got %r" % (speed_mms,))

    return kind, points, speed_mms, segment.get("why", "")


def parse_status(line):
    """Pull a GRBL status report apart. Returns a dict, or None if not a report.

    Input looks like  <Idle|MPos:0.000,0.000,0.000|FS:0,0|WCO:1.000,2.000,0.000>
    Output keys: "state" (str), "mpos"/"wpos"/"wco" (tuple of floats, or None).

    Only the state word is guaranteed. Which position fields appear depends on
    the controller's $10 status report mask, so every caller must cope with
    any of them being None.
    """
    if line is None:
        return None
    text = line.strip()
    if not (text.startswith("<") and text.endswith(">")):
        return None

    fields = text[1:-1].split("|")
    if not fields or not fields[0]:
        return None

    # "Hold:0" and "Door:2" carry a sub-state after a colon. We only ever care
    # about the word in front of it.
    report = {
        "state": fields[0].split(":")[0],
        "mpos": None,
        "wpos": None,
        "wco": None,
    }

    for field in fields[1:]:
        name, _, value = field.partition(":")
        key = {"MPos": "mpos", "WPos": "wpos", "WCO": "wco"}.get(name)
        if key is None:
            continue
        numbers = []
        for part in value.split(","):
            try:
                numbers.append(float(part))
            except ValueError:
                numbers = []
                break
        # Reports are three-axis (or more) but this machine is flat; X and Y are
        # the whole story.
        if len(numbers) >= 2:
            report[key] = tuple(numbers[:2])

    return report


# ---------------------------------------------------------------------------
# The abstract base
# ---------------------------------------------------------------------------

class MotionDriver(object):
    """What every motion driver promises to do. No implementation lives here.

    Subclasses must override all six methods. The base raises
    NotImplementedError so a half-finished driver fails at the call, with the
    name of the missing method, instead of silently doing nothing.
    """

    def home(self):
        """Find the limit switches and establish where the carriage is.

        Nothing else in the system may be trusted until this has happened: the
        controller powers up with no idea where it is.
        """
        raise NotImplementedError("home")

    def run_segment(self, segment):
        """Execute one segment dict from motion/planner.py.

        The driver owns the magnet during a segment: a "travel" segment runs
        with it disengaged, a "drag" segment with it engaged. The caller never
        has to remember, and the two can never disagree.
        """
        raise NotImplementedError("run_segment")

    def wait_until_idle(self, timeout_s=None):
        """Block until the carriage has actually stopped moving.

        Raises MotionTimeout rather than waiting forever.
        """
        raise NotImplementedError("wait_until_idle")

    def position(self):
        """Last known carriage position as (x_mm, y_mm) in board coordinates."""
        raise NotImplementedError("position")

    def set_magnet(self, engaged):
        """Turn the carriage magnet on (True) or off (False)."""
        raise NotImplementedError("set_magnet")

    def is_homed(self):
        """True once home() has succeeded and position() means something."""
        raise NotImplementedError("is_homed")


# ---------------------------------------------------------------------------
# The fake
# ---------------------------------------------------------------------------

class FakeMotionDriver(MotionDriver):
    """A gantry made of arithmetic.

    It obeys instantly, it never sleeps, and it writes down everything it was
    asked to do. Because it finishes a segment before run_segment() returns,
    wait_until_idle() is always a no-op -- which is exactly what makes the test
    suite finish in under a second.

    Things worth asserting on after a game:
        .history        every segment run, in order, as given
        .position()     where the carriage ended up
        .magnet_engaged whether the magnet is on right now
        .events         segments AND magnet changes AND homings, interleaved
        .sim_time_s     how long this would have taken on real hardware
        .warnings       plans that did not chain head to tail
    """

    def __init__(self, x_mm=None, y_mm=None):
        # A real machine that was switched off politely is sitting on its park
        # spot, so that is where the fake starts. It is still not homed: the
        # controller has no idea where it is until the switches say so, and
        # code that forgets to home should be caught by is_homed(), not
        # rescued by a lucky starting guess.
        self._x_mm = config.PARK_X_MM if x_mm is None else float(x_mm)
        self._y_mm = config.PARK_Y_MM if y_mm is None else float(y_mm)
        self._homed = False

        self.magnet_engaged = False
        self.history = []
        self.events = []
        self.warnings = []
        self.home_count = 0
        self.distance_mm = 0.0
        self.sim_time_s = 0.0

    # -- the interface ------------------------------------------------------

    def home(self):
        # Never crawl to the limit switches with a piece stuck underneath. Homing
        # is a long slow move across the whole frame, so a magnet left engaged
        # would rake a piece off the board and jam the carriage against the stop.
        self.set_magnet(False)
        self._x_mm = config.HOMING_OFFSET_X_MM
        self._y_mm = config.HOMING_OFFSET_Y_MM
        self._homed = True
        self.home_count += 1
        self.events.append(("home", None))

    def run_segment(self, segment):
        kind, points, speed_mms, why = check_segment(segment)

        # Magnet first, motion second. On the real machine the order is not
        # optional: engaging halfway through a drag leaves the piece behind,
        # and releasing early drops it in a lane between squares.
        self.set_magnet(kind == DRAG)

        start_x, start_y = points[0]
        gap_mm = math.hypot(start_x - self._x_mm, start_y - self._y_mm)
        if gap_mm > POSITION_TOLERANCE_MM:
            # Not fatal here, but it means somebody planned from a stale
            # position. On the real board the carriage would take a diagonal
            # shortcut across the squares to reach the start of this segment,
            # ploughing through whatever is in the way.
            self.warnings.append(
                "segment %r starts %.2f mm from the carriage at (%.2f, %.2f)"
                % (why or kind, gap_mm, self._x_mm, self._y_mm)
            )

        length_mm = segment_length_mm(points)
        self.distance_mm += length_mm
        self.sim_time_s += length_mm / speed_mms

        # A one-point segment commands no motion, so the carriage has not gone
        # anywhere. Moving the recorded position to that point instead would let
        # a degenerate plan teleport our idea of where the gantry is.
        if len(points) > 1:
            self._x_mm, self._y_mm = points[-1]

        self.history.append(dict(segment))
        self.events.append(("segment", dict(segment)))

    def wait_until_idle(self, timeout_s=None):
        # Nothing to wait for: run_segment() already finished the motion. The
        # argument exists only so the fake and the real driver have the same
        # shape.
        return True

    def position(self):
        return (self._x_mm, self._y_mm)

    def set_magnet(self, engaged):
        engaged = bool(engaged)
        if engaged == self.magnet_engaged:
            return
        self.magnet_engaged = engaged
        # Real hardware needs a moment for the piece to settle onto the magnet
        # before it can be dragged. Free here, but it counts against the move
        # budget on the real board, so the simulated clock charges for it.
        self.sim_time_s += (
            config.MAGNET_ENGAGE_SETTLE_S if engaged else config.MAGNET_RELEASE_SETTLE_S
        )
        self.events.append(("magnet", engaged))

    def is_homed(self):
        return self._homed

    # -- extras the fake can afford ----------------------------------------

    def reset_history(self):
        """Forget what has happened so far, keep where the carriage is."""
        self.history = []
        self.events = []
        self.warnings = []
        self.distance_mm = 0.0
        self.sim_time_s = 0.0


# ---------------------------------------------------------------------------
# The real one
# ---------------------------------------------------------------------------

class GrblMotionDriver(MotionDriver):
    """Drives a GRBL-style controller over USB serial.

    Line protocol, 115200 8N1. Every line we send comes back with exactly one
    "ok" or one "error:N". Single-byte realtime commands ("?", "!", "~", 0x18)
    are the exception: they jump the line buffer entirely and are answered out
    of band, so they are written as raw bytes with no newline.

    Coordinates on the wire are board coordinates -- the same millimetres the
    rest of the program uses, origin at the centre of a1. home() sets that up
    with a G92 immediately after the homing cycle finishes.
    """

    def __init__(self, port=None, baud=None, timeout_s=None,
                 serial_factory=None, time_fn=None, sleep_fn=None):
        # LAZY IMPORT. "import x" inside a function runs when the function is
        # called, not when the file is loaded, so motion/driver.py imports fine
        # on a laptop with no pyserial installed -- which is every laptop on
        # this team. Only building a real driver needs the library.
        if serial_factory is None:
            import serial  # noqa: F401  (only reachable on the Pi)
            serial_factory = serial.Serial

        self._port_name = config.SERIAL_PORT if port is None else port
        self._baud = config.SERIAL_BAUD if baud is None else baud
        self._timeout_s = config.SERIAL_TIMEOUT_S if timeout_s is None else timeout_s

        # Injected so tests can run a whole homing cycle in zero real seconds.
        self._now = time.monotonic if time_fn is None else time_fn
        self._sleep = time.sleep if sleep_fn is None else sleep_fn

        self._serial = serial_factory(
            port=self._port_name,
            baudrate=self._baud,
            timeout=self._timeout_s,
        )

        self._x_mm = 0.0
        self._y_mm = 0.0
        self._homed = False
        self.magnet_engaged = False

        # Work coordinate offset from the last status report that carried one, or
        # None when we do not know it yet. GRBL 1.1 sends WCO only every tenth
        # report or so and WPos = MPos - WCO, so we have to remember the last one
        # we were told. Zero is a fair guess before homing -- a controller with no
        # offsets set has WCO 0 -- and position() means nothing before homing
        # anyway, but see home(): after a G92 zero is flatly wrong.
        self._wco = (0.0, 0.0)

        self._drain_banner()

        # NO G-CODE HERE. A controller with homing enabled boots straight into
        # Alarm (grbl main.c) and answers every G-code line with error:9,
        # STATUS_SYSTEM_GC_LOCK (grbl protocol.c), until it has homed. Sending the
        # preamble at connect would make the constructor raise on every real
        # machine. It goes out in home() instead, once the controller is Idle.

    # -- the interface ------------------------------------------------------

    def home(self):
        self._homed = False

        # Never crawl to the limit switches with a piece stuck underneath: homing
        # is a slow move across the whole frame and would rake the piece off the
        # board. Sends nothing at all if the magnet is already off, which is the
        # normal case at boot.
        try:
            self.set_magnet(False)
        except ControllerError:
            # The controller is sitting in Alarm and rejecting G-code (error:9).
            # Its own reset already dropped the coolant output (grbl
            # motion_control.c mc_reset() calls coolant_stop()), so the magnet is
            # off whatever we believed. Homing is how we get OUT of Alarm, so
            # this safety step must not be what blocks it.
            self.magnet_engaged = False

        # "$H" is answered with its "ok" only after the entire homing cycle has
        # finished -- grbl runs it inline in system_execute_line() and reports the
        # status afterwards (grbl protocol.c). Homing crawls the full travel, so
        # this one line needs the homing deadline; the ordinary two-second serial
        # deadline would abort every healthy home.
        self._send_line("$H", timeout_s=HOMING_TIMEOUT_S)
        self.wait_until_idle(timeout_s=HOMING_TIMEOUT_S)

        # Now that the controller is out of Alarm it will accept G-code. Units
        # first: G92's numbers are read in whatever units are current, so setting
        # millimetres after it would silently reinterpret the offset.
        for line in GRBL_PREAMBLE:
            self._send_line(line)

        # The switches are not at the centre of a1. Declare the point we just
        # homed to as its known board coordinate so every later G0/G1 can be
        # written in plain board millimetres.
        self._send_line(
            "G92 X%.3f Y%.3f" % (config.HOMING_OFFSET_X_MM, config.HOMING_OFFSET_Y_MM)
        )
        self._x_mm = config.HOMING_OFFSET_X_MM
        self._y_mm = config.HOMING_OFFSET_Y_MM

        # The work offset is now whatever machine coordinate the switches sit at,
        # which is exactly the number we do not know. GRBL reports WCO as
        # coord_system + coord_offset (grbl report.c) and flags a refresh the
        # moment a G92 lands, so the very next status report will tell us. Until
        # it does, an MPos-only report must not be believed -- treating the offset
        # as zero would put the carriage a whole machine travel away.
        self._wco = None
        self._homed = True

    def run_segment(self, segment):
        kind, points, speed_mms, _why = check_segment(segment)

        # Wait for the PREVIOUS segment to finish before touching the magnet.
        # The magnet is switched by a line in the same queue as the moves, but
        # the settle dwell and our own bookkeeping assume the carriage is
        # standing still, and a magnet that flips mid-drag drops the piece.
        self.wait_until_idle()
        self.set_magnet(kind == DRAG)

        feed_mm_per_min = speed_mms * SECONDS_PER_MINUTE
        for index in range(1, len(points)):
            x_mm, y_mm = points[index]
            if kind == DRAG:
                # G1 is a coordinated feed move at F. Dragging is the only time
                # we care what the speed actually is.
                self._send_line("G1 X%.3f Y%.3f F%.1f" % (x_mm, y_mm, feed_mm_per_min))
            else:
                # G0 is a rapid: the controller uses its own maximum rate, which
                # is what we want for an empty carriage.
                self._send_line("G0 X%.3f Y%.3f" % (x_mm, y_mm))

        # Where the carriage WILL be. The moves are only queued at this point;
        # wait_until_idle() replaces this with a measured position. A one-point
        # segment queued no move at all, so it must not move our idea of where
        # the carriage is either.
        if len(points) > 1:
            self._x_mm, self._y_mm = points[-1]

    def wait_until_idle(self, timeout_s=None):
        """Poll "?" until the controller says Idle. This is the whole point of
        the class -- see the module docstring."""
        limit_s = config.MOTION_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = self._now() + limit_s

        while True:
            self._send_realtime("?")
            report = self._read_status()

            if report is not None:
                state = report["state"]
                if state == "Alarm":
                    raise ControllerError(
                        "controller is in Alarm; it will refuse to move until "
                        "it is reset and re-homed"
                    )
                self._update_position_from(report)
                if state in ("Idle", "Check", "Sleep"):
                    return True

            if self._now() >= deadline:
                raise MotionTimeout(
                    "carriage did not reach Idle within %.1f s -- controller "
                    "wedged, cable unplugged, or a limit switch is stuck" % (limit_s,)
                )
            self._sleep(config.MOTION_POLL_INTERVAL_S)

    def position(self):
        return (self._x_mm, self._y_mm)

    def set_magnet(self, engaged):
        engaged = bool(engaged)
        if engaged == self.magnet_engaged:
            return
        self._send_line(MAGNET_ON_GCODE if engaged else MAGNET_OFF_GCODE)

        # G4 is a dwell: the controller sits still for P seconds. Better than
        # sleeping in Python, because Linux is not a real-time system and may
        # stall this process for milliseconds at a time -- and because the
        # dwell lands in the queue in the right place, after the magnet line
        # and before the next move.
        settle_s = (
            config.MAGNET_ENGAGE_SETTLE_S if engaged else config.MAGNET_RELEASE_SETTLE_S
        )
        self._send_line("G4 P%.3f" % (settle_s,))
        self.magnet_engaged = engaged

    def is_homed(self):
        return self._homed

    # -- extras --------------------------------------------------------------

    def feed_hold(self):
        """Decelerate to a stop but keep position. Resume with cycle_start()."""
        self._send_realtime("!")

    def cycle_start(self):
        """Resume from a feed hold."""
        self._send_realtime("~")

    def soft_reset(self):
        """Ctrl-X. Clears the queue and any alarm; position is then unknown."""
        self._send_realtime("\x18")
        self._homed = False

        # A reset kills the coolant output on its way past -- grbl
        # motion_control.c mc_reset() calls coolant_stop() -- so the magnet is
        # physically OFF now. If we went on believing it was on, the next
        # set_magnet(True) would see no change, send no M8, and the following
        # drag would run with a dead magnet: the piece stays where it was and
        # every square on the board is wrong from then on.
        self.magnet_engaged = False

        # The reset also throws away the G92 offset that home() installed, so the
        # work offset we were remembering no longer describes anything.
        self._wco = None

    def close(self):
        """Drop the magnet before letting go of the port.

        If the program exits mid-drag with the magnet on, the piece stays stuck
        to a dead carriage under the board.
        """
        try:
            self.set_magnet(False)
        except MotionError:
            # Already broken; closing the port is still the right move.
            pass
        self._serial.close()

    # -- serial plumbing -----------------------------------------------------

    def _send_line(self, text, timeout_s=None):
        """Send one G-code line and wait for its "ok".

        timeout_s overrides the usual serial deadline for the rare line whose
        "ok" only comes back after real motion, which today is just "$H".
        """
        self._serial.write((text + "\n").encode("ascii"))
        self._await_ok(text, timeout_s)

    def _send_realtime(self, char):
        """Send a single realtime byte, with NO newline.

        Realtime commands are picked out of the serial interrupt handler and
        acted on immediately; they never enter the line buffer. Appending a
        newline would push a stray empty line into the buffer, which GRBL
        answers with its own "ok" -- and from then on every ok we read belongs
        to the wrong command.
        """
        self._serial.write(char.encode("ascii"))

    def _await_ok(self, sent, timeout_s=None):
        limit_s = self._timeout_s if timeout_s is None else timeout_s
        deadline = self._now() + limit_s
        while True:
            line = self._read_line()
            if line == "ok":
                return
            if line.startswith("error:"):
                raise ControllerError("controller rejected %r with %r" % (sent, line))
            if line.startswith("ALARM"):
                raise ControllerError("controller alarmed on %r: %r" % (sent, line))
            # Anything else is a status report or a banner line that arrived
            # while we were waiting. Ignore it and keep reading.
            if not line:
                # Empty means the read timed out with nothing on the wire. Wait
                # a poll interval instead of spinning: a tight loop on a silent
                # port eats a whole CPU core on the Pi for no reason.
                self._sleep(config.MOTION_POLL_INTERVAL_S)
            if self._now() >= deadline:
                raise MotionTimeout(
                    "no reply to %r within %.1f s -- controller wedged or the "
                    "USB cable came out" % (sent, limit_s)
                )

    def _read_status(self):
        """Read lines until one of them is a status report, or we give up."""
        deadline = self._now() + self._timeout_s
        while True:
            line = self._read_line()
            report = parse_status(line)
            if report is not None:
                return report
            if not line:
                self._sleep(config.MOTION_POLL_INTERVAL_S)
            if self._now() >= deadline:
                return None

    def _read_line(self):
        raw = self._serial.readline()
        if not raw:
            return ""
        return raw.decode("ascii", "replace").strip()

    def _drain_banner(self):
        """Swallow the "Grbl 1.1f ['$' for help]" greeting and any leftovers."""
        for _ in range(MAX_BANNER_LINES):
            if not self._read_line():
                return

    def _update_position_from(self, report):
        if report["wco"] is not None:
            self._wco = report["wco"]
        if report["wpos"] is not None:
            self._x_mm, self._y_mm = report["wpos"]
        elif report["mpos"] is not None:
            # WPos = MPos - WCO. Board coordinates are work coordinates here,
            # so a report that only carries MPos has to be corrected by the
            # last work offset we were told about.
            self._x_mm = report["mpos"][0] - self._wco[0]
            self._y_mm = report["mpos"][1] - self._wco[1]


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def make_motion_driver(**kwargs):
    """Return the driver this machine should be using.

    Read at call time, not at import time, so a test can flip
    config.USE_FAKE_HARDWARE and get the other one. Keyword arguments are
    passed straight through to whichever class gets built.
    """
    if config.USE_FAKE_HARDWARE:
        return FakeMotionDriver(**kwargs)
    return GrblMotionDriver(**kwargs)
