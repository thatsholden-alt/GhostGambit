"""Reading the 64-square occupancy grid, and filtering the noise out of it.

Three separate jobs live here, and keeping them separate is the point:

    occupancy_from_board()   what the board SHOULD look like, from a chess
                             position. Pure arithmetic, no hardware.
    OccupancySensor          what the board actually looks like, from the
                             sensor grid. Real and fake versions.
    Debouncer                deciding when a reading is trustworthy enough to
                             act on.

The occupancy mask is a plain 64-bit int. Bit i is set when square i holds a
piece, where i is python-chess's numbering: a1 == 0, h8 == 63. Everything in
the sensing stack speaks this one format -- sensing/inference.py compares a
sensor reading against occupancy_from_board() of a candidate position, so if
the two ever disagreed about bit order nothing would work at all.

Physical rule 3 lives just outside this file: the sensor grid is IGNORED while
the gantry is moving, because our own carriage magnet passing underneath reads
as a piece. Nothing here can detect that on its own, so the caller (the state
machine) is responsible for only polling once the carriage has parked, and for
calling Debouncer.reset() after motion so readings taken mid-move cannot be
mistaken for a settled board.
"""

import time

import chess

import config


NUM_SQUARES = config.BOARD_FILES * config.BOARD_RANKS

# Every square occupied. Handy for range-checking a mask that came off the wire.
FULL_BOARD_MASK = (1 << NUM_SQUARES) - 1


# --- Wiring constants -------------------------------------------------------
#
# THESE SHOULD MOVE TO config.py once the sensor board is designed. They are
# physical facts about hardware that does not exist yet, so putting provisional
# numbers in config.py would make them look more decided than they are. When the
# PCB is real, move all four of these across and delete them here.

# Four MCP23017 I2C port expanders, 16 inputs each, 4 * 16 = 64 squares. The
# address is strapped with the A0/A1/A2 pins, giving 0x20 through 0x27; we use
# the bottom four.
EXPANDER_ADDRESSES = (0x20, 0x21, 0x22, 0x23)
EXPANDER_SQUARES = 16           # 8 pins on port A + 8 on port B

# Bus 1 is the header I2C on every Pi since the model B rev 2.
I2C_BUS_NUMBER = 1

# Reed switches wired between the input pin and ground, with the expander's
# internal pull-up on. An OPEN switch (no magnet) therefore floats HIGH and a
# CLOSED switch (piece present) reads LOW, so the byte has to be inverted.
# Wiring it this way needs no external resistors.
#
# This flag also decides whether the internal pull-ups get switched on at all --
# see _configure_expanders(). The two decisions are the same decision.
SENSOR_ACTIVE_LOW = True

# MCP23017 register addresses, from the datasheet, with IOCON.BANK left at its
# power-on default of 0.
_REG_IODIRA = 0x00      # 1 = pin is an input
_REG_IODIRB = 0x01
_REG_GPPUA = 0x0C       # 1 = enable the 100k internal pull-up
_REG_GPPUB = 0x0D
_REG_GPIOA = 0x12       # reading this gives the pin states
_REG_GPIOB = 0x13

_ALL_INPUTS = 0xFF
_ALL_PULLUPS_ON = 0xFF
_ALL_PULLUPS_OFF = 0x00

# One MCP23017 port is eight pins wide. A datasheet fact about the chip, not a
# choice about our board, which is why it lives here and not in config.py.
_PORT_BITS = 8
_PORT_MASK = 0xFF


def occupancy_from_board(board):
    """Return the 64-bit occupancy mask for a python-chess Board.

    Bit i is set when square i holds a piece of either colour. This is the most
    reused function in the sensing stack: move inference works by playing a
    candidate move, calling this, and comparing the result with what the sensors
    read.
    """
    mask = 0
    # python-chess already keeps a mask like this in board.occupied, but we
    # build it from piece_at() so that the bit order is defined here, in the
    # open, rather than inherited from a library detail.
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            mask |= 1 << square
    return mask


def mask_from_squares(squares):
    """Return the mask with exactly these square indices set. Mostly for tests."""
    mask = 0
    for square in squares:
        mask |= 1 << square
    return mask


def squares_in_mask(mask):
    """Return a sorted list of the square indices set in a mask."""
    return [square for square in chess.SQUARES if mask & (1 << square)]


def is_occupied(mask, square):
    """True when the mask says this square holds a piece."""
    return bool(mask & (1 << square))


def format_mask(mask):
    """Return the mask as eight lines of ASCII, rank 8 at the top.

    '#' is an occupied square, '.' is empty. This is for the screen and for test
    failure messages -- staring at a 64-bit hex number to find which square is
    wrong is miserable.
    """
    lines = []
    for rank in range(config.BOARD_RANKS - 1, -1, -1):
        row = []
        for file_index in range(config.BOARD_FILES):
            square = chess.square(file_index, rank)
            row.append("#" if is_occupied(mask, square) else ".")
        lines.append(" ".join(row))
    return "\n".join(lines)


def _check_mask(mask):
    """Raise unless mask is a plain 64-bit non-negative int.

    Catching a bad shift here turns a silent wrong-square bug into an immediate,
    obvious failure.
    """
    if not isinstance(mask, int) or isinstance(mask, bool):
        raise TypeError("occupancy mask must be an int, got %r" % (mask,))
    if mask < 0 or mask > FULL_BOARD_MASK:
        raise ValueError(
            "occupancy mask must fit in %d bits, got %r" % (NUM_SQUARES, mask)
        )
    return mask


class OccupancySensor(object):
    """Abstract base for anything that can report which squares are occupied.

    Subclasses implement read_raw(). "Raw" means exactly that: one instantaneous
    look at the grid, with no filtering. Feed it to a Debouncer before believing
    it.
    """

    def read_raw(self):
        raise NotImplementedError("subclasses must implement read_raw()")

    def close(self):
        """Release any hardware. The fake does not need it; the real one does."""
        pass


class FakeOccupancySensor(OccupancySensor):
    """A sensor grid made of a single variable.

    Used by every test and by the simulator: set the mask by hand, or from a
    python-chess Board, and read_raw() hands it straight back. No hardware, no
    timing, no noise unless a test adds some by calling set_mask() repeatedly.
    """

    def __init__(self, mask=0):
        self._mask = _check_mask(mask)
        self.read_count = 0     # lets a test prove polling actually happened

    def set_mask(self, mask):
        self._mask = _check_mask(mask)

    def set_from_board(self, board):
        self._mask = occupancy_from_board(board)

    def read_raw(self):
        self.read_count += 1
        return self._mask


class Mcp23017OccupancySensor(OccupancySensor):
    """The real grid: 64 reed switches on four MCP23017 I2C port expanders.

    Square numbering assumed by the wiring: expander 0 carries squares 0..15
    (a1..h2), and within an expander port A carries the low 8 squares and port B
    the high 8. That is a guess at a PCB nobody has laid out yet; when the board
    is real, either wire it to match or change the bit assembly in read_raw().

    All the I2C traffic goes through _read_register() and _write_register(), so
    the rest of the class is ordinary arithmetic that can be tested by passing a
    stand-in bus object to the constructor.
    """

    def __init__(self, bus=None, bus_number=I2C_BUS_NUMBER,
                 addresses=EXPANDER_ADDRESSES, active_low=SENSOR_ACTIVE_LOW):
        self.addresses = tuple(addresses)
        self.active_low = active_low
        if len(self.addresses) * EXPANDER_SQUARES != NUM_SQUARES:
            raise ValueError(
                "%d expanders * %d pins does not cover %d squares"
                % (len(self.addresses), EXPANDER_SQUARES, NUM_SQUARES)
            )

        if bus is not None:
            # A caller (a test, or a future multi-device bus manager) supplied
            # an already-open bus, so do not open another one -- and remember
            # that it is not ours to close.
            self._bus = bus
            self._owns_bus = False
        else:
            # smbus is imported HERE and not at the top of the file. It only
            # installs on a Linux machine with an I2C bus, and this module has
            # to import on a Windows laptop with nothing plugged in, or no test
            # in the project could run before December.
            try:
                import smbus2 as smbus
            except ImportError:
                import smbus
            self._bus = smbus.SMBus(bus_number)
            self._owns_bus = True

        self._configure_expanders()

    def _read_register(self, address, register):
        """The only place this class reads I2C."""
        return self._bus.read_byte_data(address, register)

    def _write_register(self, address, register, value):
        """The only place this class writes I2C."""
        self._bus.write_byte_data(address, register, value)

    def _configure_expanders(self):
        """Make every pin an input, with the pull-ups on only if we need them.

        The MCP23017's internal resistor pulls UP, toward VDD, and the chip has
        no internal pull-down at all. So the pull-up belongs to the active-low
        wiring (switch between pin and ground) and ONLY to it. Leaving it on for
        active-high wiring would hold every pin high whether its switch was
        closed or not, and the board would read as permanently full -- so an
        active-high PCB has to bring its own external pull-down resistors.
        """
        pullups = _ALL_PULLUPS_ON if self.active_low else _ALL_PULLUPS_OFF
        for address in self.addresses:
            self._write_register(address, _REG_IODIRA, _ALL_INPUTS)
            self._write_register(address, _REG_IODIRB, _ALL_INPUTS)
            self._write_register(address, _REG_GPPUA, pullups)
            self._write_register(address, _REG_GPPUB, pullups)

    def read_raw(self):
        mask = 0
        for expander, address in enumerate(self.addresses):
            # Clamp each port to eight bits before using it. A real SMBus read
            # can only return 0..255, but a stand-in bus or a future backend
            # might not: on the active-low path the inversion below would
            # silently truncate an oversized value, while on the active-high
            # path the surplus bits would slide up into the NEXT expander's
            # squares and report pieces that are not there. Same clamp, both
            # paths, so read_raw() can never hand out a mask wider than 64 bits.
            port_a = self._read_register(address, _REG_GPIOA) & _PORT_MASK
            port_b = self._read_register(address, _REG_GPIOB) & _PORT_MASK
            sixteen = (port_b << _PORT_BITS) | port_a
            if self.active_low:
                # A closed switch pulls its pin to ground, so 0 means "piece".
                sixteen = ~sixteen & 0xFFFF
            mask |= sixteen << (expander * EXPANDER_SQUARES)
        return mask

    def close(self):
        """Close the I2C bus, but only if this object is the one that opened it.

        A caller who handed us a bus is probably still using it for something
        else, and a sensor quietly closing it out from under them would be a
        miserable bug to track down.
        """
        if self._owns_bus:
            self._bus.close()


class Debouncer(object):
    """Holds a sensor reading at arm's length until the whole board sits still.

    Why WHOLE-BOARD and not per-square: a per-square debounce lets each square
    settle on its own schedule, so the mask you finally hand to move inference
    is stitched together from different instants -- square e2 as it was 300 ms
    ago, square e4 as it is now. A half-finished move can look exactly like a
    completed different move that way. Whole-board means one timer for all 64
    bits: any square changing restarts it, so a mask that comes out of here is
    one snapshot of a board that was completely motionless. It also matches how
    a turn actually ends -- the player moves, lifts their hand away, and
    everything stops at once.

    The cost is real and intended: a hand resting over one square forever means
    no report ever comes, and the screen keeps saying "waiting for the board to
    settle". That is the correct behaviour.

    Why time_fn and now: the clock is injected instead of called directly, so a
    test can hand in a function that returns whatever number it likes and step
    it forward by hand. That turns "wait 100 ms and see" into instant
    arithmetic, and the tests never sleep.

    Why time.monotonic and never time.time: a Raspberry Pi has no
    battery-backed real-time clock. Its wall clock is wrong at boot and then
    JUMPS, possibly by years, the moment NTP syncs after the network comes up.
    A debounce built on wall-clock time would fire instantly or never fire at
    all the moment that happens. monotonic only ever counts forward from an
    arbitrary zero, which is all a stopwatch needs.
    """

    def __init__(self, hold_s=None, time_fn=None):
        self.hold_s = config.SENSOR_DEBOUNCE_S if hold_s is None else hold_s
        self.time_fn = time.monotonic if time_fn is None else time_fn
        self._candidate = None          # the mask currently being timed
        self._deadline = None           # the time it becomes believable
        self._last_reported = None      # so a settled board is reported once

    def update(self, raw_mask, now=None):
        """Feed one raw reading in. Return the stable mask, or None.

        A mask is returned exactly once, on the poll where it has held steady
        for hold_s and it differs from whatever was reported last. Every other
        call returns None, so the caller can poll at SENSOR_POLL_HZ forever and
        only hear about real changes.
        """
        if now is None:
            now = self.time_fn()

        if raw_mask != self._candidate:
            self._candidate = raw_mask
            # Store the deadline rather than the start time and subtract later.
            # Both are one line, but "now - started < hold_s" loses the exact
            # boundary to floating-point rounding: a clock that reached exactly
            # started + hold_s can subtract back to a hair under hold_s and the
            # sample gets rejected. Comparing against a deadline computed once
            # has no such gap.
            self._deadline = now + self.hold_s

        # A reading that has held for exactly the hold time has met the
        # requirement, so this is "still early", not "not yet finished".
        if now < self._deadline:
            return None

        if raw_mask == self._last_reported:
            return None

        self._last_reported = raw_mask
        return raw_mask

    def reset(self):
        """Forget everything seen so far.

        Call this after the gantry has moved and parked. Readings taken while
        the carriage was under the board are worthless -- our own magnet reads
        as a piece -- and this makes sure none of them can count toward the next
        settled reading, and that the board gets re-reported even if the mask
        happens to match the last one.
        """
        self._candidate = None
        self._deadline = None
        self._last_reported = None

    def settling(self):
        """True when a reading is being timed but has not been reported yet."""
        return self._candidate is not None and self._candidate != self._last_reported


def make_occupancy_sensor(**kwargs):
    """Return the fake or the real sensor, per config.USE_FAKE_HARDWARE."""
    if config.USE_FAKE_HARDWARE:
        return FakeOccupancySensor(**kwargs)
    return Mcp23017OccupancySensor(**kwargs)
