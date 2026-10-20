"""Every constant that describes the real world, in one place.

The rule: never inline a number that describes physical reality. These values
change after calibration, and hunting for them across ten files is how we lose
an afternoon.

Values fall into three groups, and the difference matters:

    MEASURED      someone put calipers on it. Trustworthy.
    THEORETICAL   calculated from a datasheet. Probably wrong by a few percent.
    PROVISIONAL   a placeholder for an open question. Will change. Do not build
                  anything on it that would be expensive to unbuild.

Anything marked PROVISIONAL is flagged loudly below. If a number you need is
PROVISIONAL and getting it wrong would matter, stop and ask the team.
"""

# ---------------------------------------------------------------------------
# Hardware switch
# ---------------------------------------------------------------------------

# The hardware does not exist until December. Every hardware module has a real
# implementation and a fake one, and this flag picks which. If a change breaks
# play_text.py on a laptop with this set to True, the change is wrong.
USE_FAKE_HARDWARE = True


# ---------------------------------------------------------------------------
# Board geometry  (MEASURED from the design drawings, not from a built board)
# ---------------------------------------------------------------------------

SQUARE_MM = 57.15               # 2.25 in
PLAYING_AREA_MM = 457.2         # 8 squares
BOARD_FILES = 8
BOARD_RANKS = 8

# Design target for the piece base diameter, with a hard ceiling. The ceiling is
# half the square size: any wider and two pieces on adjacent squares touch, so
# the magnet cannot pull one between them.
PIECE_BASE_MAX_MM = 26.0
PIECE_BASE_CEILING_MM = 28.575  # SQUARE_MM / 2


# ---------------------------------------------------------------------------
# Coordinate system  -- defined once here, used everywhere, never redefined
# ---------------------------------------------------------------------------
#
#   Origin (0, 0) = centre of square a1
#   +X = toward the h file   (right, from White's side)
#   +Y = toward rank 8       (away from White)
#   Units = millimetres, float
#
#   Square centre, file f (0=a..7=h), rank r (0=rank1..7=rank8):
#       x = f * SQUARE_MM
#       y = r * SQUARE_MM
#
#   Lane intersections sit on the half-square grid:
#       x = (f + 0.5) * SQUARE_MM
#       y = (r + 0.5) * SQUARE_MM
#
# Pieces route along the lanes, never straight through square centres. That is
# the only reason a knight can leave a crowded back rank without shoving its
# neighbours.


# ---------------------------------------------------------------------------
# Motion  (THEORETICAL -- none of this has been measured)
# ---------------------------------------------------------------------------

# 200 full steps/rev * 16 microsteps / 40 mm of belt per rev = 80.
# THEORETICAL. Calibrate by commanding a known distance and measuring with
# calipers, then replace this number. Open question #6.
STEPS_PER_MM = 80.0

# Two speeds, not one. The empty carriage travelling to the source square has
# nothing attached and can go fast; the loaded drag cannot, because accelerating
# hard while loaded makes the piece slip off the magnet or topple over.
TRAVEL_SPEED_MMS = 300.0        # magnet disengaged, nothing attached
DRAG_SPEED_MMS = 100.0          # magnet engaged, piece attached

# Acceleration matters more than top speed here: the magnetic coupling breaks
# on acceleration, not on velocity. Over a single 57 mm square the carriage
# never reaches top speed anyway. THEORETICAL -- tune on the mock-up.
TRAVEL_ACCEL_MMSS = 1000.0
DRAG_ACCEL_MMSS = 300.0

# Where the carriage sits when it is not doing anything.
#
# It MUST be off the 64 squares. Our own carriage magnet passing under the
# sensor grid reads as a piece, so parking on a square makes that square report
# occupied forever.
#
# Centred on the near edge rather than tucked in a corner, because the empty
# travel to the source square is dead time inside the 4 s move budget, and
# centring cuts the worst case by about 100 mm.
PARK_X_MM = 3.5 * SQUARE_MM     # centred left-to-right, 200.025
PARK_Y_MM = -1.0 * SQUARE_MM    # one square below rank 1, clear of the board

# Homing. The limit switch trigger point is not the centre of a1; the offset
# between them is a physical distance that can only be measured once the frame
# exists. PROVISIONAL -- open question #7.
HOMING_OFFSET_X_MM = 0.0
HOMING_OFFSET_Y_MM = 0.0


# ---------------------------------------------------------------------------
# Capture area  -- PROVISIONAL, open question #5
# ---------------------------------------------------------------------------
#
# !! NOT A DECISION. The capture area has not been designed. These numbers exist
# !! so the simulator can run end to end; they are not a layout anyone agreed to.
# !! Do not cut wood to these numbers.
#
# Shape assumed here: a strip down the +X side of the board, 16 slots, 8 per
# colour. One reference build used a perimeter ring instead and allocated the
# first free slot in raster order, which is a better idea and worth stealing --
# but it is still the team's call.
CAPTURE_AREA_PROVISIONAL = True
CAPTURE_X_MM = PLAYING_AREA_MM + 15.0
CAPTURE_SLOT_PITCH_MM = 30.0
CAPTURE_SLOTS_PER_SIDE = 8


# ---------------------------------------------------------------------------
# Magnet  (PROVISIONAL -- open question #1, the project's biggest unknown)
# ---------------------------------------------------------------------------
#
# Total gap between carriage magnet and piece magnet is board surface + sensor
# layer + bottom pad. Magnetic force falls off roughly with the CUBE of
# distance, so doubling the gap leaves about an eighth of the force.
#
# Nobody has measured the pull needed. These are placeholders for the mock-up.
MAGNET_GAP_MM = 10.0            # 6 mm board + 4 mm sensor layer, before the pad

# Sliding the magnet sideways shears the grip; pulling it straight down fights
# the grip head-on and needs far more force. The carriage is designed for
# lateral retraction for that reason.
MAGNET_ENGAGE_SETTLE_S = 0.3    # let the piece settle before moving
MAGNET_RELEASE_SETTLE_S = 0.3


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------

# A square must hold a new state this long before we believe it. A piece slid
# across the board triggers every square it crosses, and a hovering hand makes
# a square flicker.
SENSOR_DEBOUNCE_S = 0.10
SENSOR_POLL_HZ = 20.0

# The sensor grid is ignored entirely while the gantry is moving, because our
# own magnet reads as a piece. Occupancy is only trusted once the carriage has
# parked.


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Fairy-Stockfish, not stock Stockfish. Measured from the binaries: stock
# Stockfish will not go below 1320 Elo, which is not a beatable beginner level.
# Fairy-Stockfish reaches 500 and packages for arm64 at 604 kB.
# Confirm the installed path with: dpkg -L fairy-stockfish
ENGINE_PATH = "/usr/games/fairy-stockfish"
ENGINE_FALLBACK_PATHS = (
    "/usr/games/stockfish",
    "fairy-stockfish",
    "stockfish",
)

# Half a second. The motion eats the rest of the 4 s move budget, and the engine
# is already deliberately handicapped by the difficulty ladder, so giving it
# more time would not make the low levels better anyway.
ENGINE_TIME_LIMIT_S = 0.5

# 16 MB is the engine default. At a half-second search it cannot fill even that,
# so raising it would cost RAM for nothing. Threads 2 of the Pi 5's 4 cores,
# leaving room for the game loop and the serial I/O.
ENGINE_HASH_MB = 16
ENGINE_THREADS = 2

DEFAULT_LEVEL = 3               # see game/difficulty.py for the ladder


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

# Fischer increment: a base time plus a few seconds added after each move.
# Increment 0 gives plain sudden death; None gives an untimed casual game.
DEFAULT_BASE_TIME_S = 600.0     # 10 minutes each
DEFAULT_INCREMENT_S = 5.0

# Whose clock runs while the gantry is physically dragging a piece: NOBODY'S.
# A capture is two trips and takes 7-10 s. Charging that to a player means they
# can lose on time because the robot is slow, which would be a visible failure
# in front of the review panel. The engine's 0.5 s of thinking IS charged to the
# engine, because that is real thinking.
CHARGE_MOTION_TIME_TO_PLAYER = False


# ---------------------------------------------------------------------------
# Motor controller link
# ---------------------------------------------------------------------------

# USB CDC serial, not I2C. Simpler in Python, debuggable from a plain terminal,
# no address conflicts, no pull-up or clock-stretching questions.
#
# The Pi never generates step pulses. Linux is not real-time and will stall the
# process for milliseconds, which loses steps and loses position. The Pi asks;
# the MCU moves.
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_S = 2.0

# An "ok" from a GRBL-style controller means "parsed and queued", NOT "the
# carriage has arrived". Motion completion is a separate question, asked by
# polling the status report until it reads Idle.
MOTION_POLL_INTERVAL_S = 0.05
MOTION_TIMEOUT_S = 30.0
