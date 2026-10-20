"""Coordinate and kinematics maths for the CoreXY gantry.

Everything in here is a small pure function: same inputs, same outputs, no
state, no hardware, no I/O. That is deliberate. This is the one module the rest
of the motion code leans on, so it has to be the one module nobody is afraid of.

WHAT COREXY IS, IN ONE PARAGRAPH
    Both motors are bolted to the frame and never move. One long belt runs from
    motor A, around the carriage, and back; a second does the same from motor B.
    Turning one motor alone drives the carriage along a 45-degree diagonal;
    turning both together drives it straight along X or Y. So neither motor
    "owns" an axis, and every commanded motion has to be converted from
    Cartesian millimetres into per-motor millimetres before it is sent.

        dA = dx + dy        dB = dx - dy            (position -> motors)
        dx = (dA + dB) / 2  dy = (dA - dB) / 2      (motors -> position)

    The reward for the extra maths is that both motors stay on the frame, so the
    only moving mass is the carriage itself. A light carriage can accelerate
    without the magnet losing its grip on the piece, which is the whole game.

COORDINATE SYSTEM
    Defined in config.py and never redefined here. Origin (0, 0) is the centre
    of a1, +X runs toward the h file, +Y runs toward rank 8, units are mm.
    Square centres sit on multiples of SQUARE_MM; lanes sit on the half-square
    grid between them.
"""

import math

import chess

import config


# The A/B vector is uniformly sqrt(2) times as long as the Cartesian one:
#
#     ||(dx + dy, dx - dy)|| = sqrt(2) * ||(dx, dy)||     for EVERY direction
#
# (Expand it: (dx+dy)^2 + (dx-dy)^2 = 2*dx^2 + 2*dy^2. The cross terms cancel.)
# Unlike most kinematic scale factors this one does not depend on direction, so
# it is a single constant rather than something you have to recompute per move.
AB_LENGTH_RATIO = math.sqrt(2.0)


def to_motor_deltas(dx_mm, dy_mm):
    """Cartesian millimetres -> A/B motor millimetres (inverse kinematics)."""
    return (dx_mm + dy_mm, dx_mm - dy_mm)


def to_cartesian(da_mm, db_mm):
    """A/B motor millimetres -> Cartesian millimetres (forward kinematics)."""
    return ((da_mm + db_mm) / 2.0, (da_mm - db_mm) / 2.0)


def feedrate_for_speed_mms(speed_mms):
    """Feedrate, in mm per SECOND, so the CARRIAGE really moves at speed_mms.

    UNITS FIRST, BECAUSE THIS IS THE EASY WAY TO BE WRONG BY 60x.
    Everything in this project is millimetres per SECOND, and so is the number
    this returns. G-code's F word is millimetres per MINUTE. Whoever writes the
    G-code multiplies by 60:  F = 60 * feedrate_for_speed_mms(v). Send this
    number straight after an F and the machine crawls at a sixtieth of speed,
    which reads on the bench as "the motors are too weak".

    READ THIS BEFORE YOU CALIBRATE THE MACHINE IN JANUARY.

    grbl applies the CoreXY transform inside plan_buffer_line() BEFORE the
    acceleration limiter runs, so block->millimeters ends up being the length of
    the move in A/B space -- which is sqrt(2) times the real distance the
    carriage travels. The feedrate F you send is applied to that longer number.
    A commanded F therefore produces roughly F / sqrt(2) of actual carriage
    speed, in every direction equally.

    Two consequences, and the second one is the trap:
      - Steps-per-mm calibration is NOT affected. Commanding a 100 mm move still
        moves the carriage 100 mm; only the speed it does it at is scaled.
      - So a machine that measures perfectly with calipers can still be running
        ~30% slow, and that looks like a mechanical problem for about a day.

    Verified in grbl's source, grbl/planner.c, the COREXY blocks (the
    delta_mm assignments for A_MOTOR / B_MOTOR, then convert_delta_vector_to_
    unit_vector): https://github.com/gnea/grbl
    """
    return speed_mms * AB_LENGTH_RATIO


def square_to_xy(square):
    """Centre of a python-chess square (0..63, a1 == 0) in board millimetres."""
    # python-chess numbers squares 0..63 with a1 == 0 and h8 == 63. Doing that
    # arithmetic by hand is how people end up with a mirrored board, so ask the
    # library instead.
    if not 0 <= square <= 63:
        raise ValueError("square must be 0..63 (python-chess numbering), got %r" % (square,))
    return file_rank_to_xy(chess.square_file(square), chess.square_rank(square))


def file_rank_to_xy(file_index, rank_index):
    """Centre of the square at file 0..7 (a..h), rank 0..7 (rank1..rank8)."""
    if not 0 <= file_index < config.BOARD_FILES:
        raise ValueError("file_index must be 0..%d, got %r" % (config.BOARD_FILES - 1, file_index))
    if not 0 <= rank_index < config.BOARD_RANKS:
        raise ValueError("rank_index must be 0..%d, got %r" % (config.BOARD_RANKS - 1, rank_index))
    return (file_index * config.SQUARE_MM, rank_index * config.SQUARE_MM)


# The lane grid runs one index wider than the board at each end, and the two
# ends are NOT mirror images of each other. Read the numbers before routing
# anything, because the obvious guess is wrong:
#
#     index -1  ->  -0.5 * SQUARE_MM  =  the outer edge of the a file / rank 1
#     index  7  ->   7.5 * SQUARE_MM  =  the outer edge of the h file / rank 8
#     index  8  ->   8.5 * SQUARE_MM  =  a FULL SQUARE past the board edge
#
# So the lane that hugs the board is -1 at the low end but 7 at the high end.
# A captured piece leaves the board along -1 or 7, and the park position sits
# half a square further out than lane -1. Index 8 is an extra ring beyond the
# playing area, further out than even CAPTURE_X_MM; it is accepted because the
# spec asks for -1..8, but nothing should route through it until someone has
# checked the gantry physically reaches that far.
#
# LANE_MIN_INDEX fixes the outer ring at one lane deep. That is a layout call,
# not maths: if the capture area ever needs the carriage to route two lanes
# outside the board, this SHOULD MOVE TO config.py beside the CAPTURE_* block.
LANE_MIN_INDEX = -1
LANE_MAX_FILE_INDEX = config.BOARD_FILES
LANE_MAX_RANK_INDEX = config.BOARD_RANKS


def lane_xy(file_index, rank_index):
    """Lane intersection on the half-square grid, in board millimetres.

    lane_xy(0, 0) is the crossing just north-east of a1's centre. Indices run
    from -1 to 8 inclusive so the ring of lanes around the outside of the board
    is addressable; see the note above about the two ends not being mirror
    images -- the board-hugging lanes are -1 and 7, not -1 and 8.
    """
    if not LANE_MIN_INDEX <= file_index <= LANE_MAX_FILE_INDEX:
        raise ValueError(
            "lane file_index must be %d..%d, got %r"
            % (LANE_MIN_INDEX, LANE_MAX_FILE_INDEX, file_index)
        )
    if not LANE_MIN_INDEX <= rank_index <= LANE_MAX_RANK_INDEX:
        raise ValueError(
            "lane rank_index must be %d..%d, got %r"
            % (LANE_MIN_INDEX, LANE_MAX_RANK_INDEX, rank_index)
        )
    return ((file_index + 0.5) * config.SQUARE_MM, (rank_index + 0.5) * config.SQUARE_MM)


def mm_to_steps(mm):
    """Millimetres of belt travel -> whole motor steps."""
    steps = mm * config.STEPS_PER_MM
    # We want "round half away from zero", and two tempting ways to get it are
    # both wrong.
    #
    # Python's built-in round() breaks exact .5 ties toward the even number:
    # round(0.5) is 0 but round(1.5) is 2. Fine for statistics, wrong here,
    # because the carriage moves in both directions and an asymmetric rule makes
    # it creep one way over a long game.
    #
    # int(math.floor(x + 0.5)) looks like the fix and is not. Adding 0.5 to a
    # value a hair UNDER 0.5 rounds up to exactly 1.0 inside the float before
    # floor() ever sees it, so 0.49999999999999994 comes out as 1 step. Taking
    # the fractional part adds nothing to anything, so it cannot invent a carry.
    magnitude = abs(steps)
    whole = math.floor(magnitude)
    if magnitude - whole >= 0.5:
        whole += 1
    if steps < 0.0:
        return -int(whole)
    return int(whole)


def steps_to_mm(steps):
    """Whole motor steps -> millimetres. Inverse of mm_to_steps, up to rounding."""
    return steps / config.STEPS_PER_MM


def path_length_mm(points):
    """Total Cartesian length of a polyline [(x, y), ...], in millimetres.

    Zero for an empty path or a single point. Used to estimate how long a
    segment will take, so the state machine knows when to stop waiting.
    """
    total = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        # math.hypot is just sqrt(dx*dx + dy*dy) without the overflow worries.
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def ab_path_length_mm(points):
    """Length of the same polyline measured in A/B space.

    Exactly AB_LENGTH_RATIO times path_length_mm, because the sqrt(2) stretch
    applies to every direction equally, so it survives summing the segments up.
    This is the number grbl thinks it is feeding at -- see feedrate_for_speed_mms.
    """
    return AB_LENGTH_RATIO * path_length_mm(points)
