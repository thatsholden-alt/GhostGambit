"""Turn one legal chess move into an ordered list of carriage segments.

This is the module that knows the SHAPE of a move. It does not talk to any
hardware and it does not know what a step pulse is; it produces a list of
dictionaries and something else drives them.

A segment is exactly:

    {"kind": "travel" or "drag", "points": [(x, y), ...],
     "speed_mms": float, "why": str}

    "travel"  magnet DISENGAGED, carriage empty, config.TRAVEL_SPEED_MMS
    "drag"    magnet ENGAGED, a piece is stuck to the carriage,
              config.DRAG_SPEED_MMS

The first point of a segment is where the carriage already is, so segments
chain head to tail with no teleports. segments_are_chained() below checks that,
and the tests lean on it hard.


THE ONE RULE THAT SHAPES EVERYTHING
-----------------------------------
While a piece is attached it travels along the LANES between squares -- the
half-square grid -- never through the centre of a square that is not its own
start or end. A piece sitting on a lane is 0.5 * SQUARE_MM from the centre of
each of the four squares around it; a piece sitting on a square centre is
1.0 * SQUARE_MM from its neighbours but is standing exactly where a neighbour
might be. Lane routing is the only reason a knight can leave a crowded back
rank without shoving the bishop next to it.

Only the very first and the very last hop of a drag touch a square centre: the
diagonal step from the start centre out to the nearest lane crossing, and the
diagonal cut from the last lane crossing into the destination centre. Those two
hops pass no closer than 0.707 * SQUARE_MM (about 40 mm) to any neighbouring
centre, which is roomier than the lanes themselves.

The knight trick -- step half a square sideways FIRST so you are on a lane,
then run the long leg along that lane, then cut in -- is lifted (as an idea,
not as code) from:
    https://github.com/AndChen153/ChessBoard   chess_move.py, move_steppers_uneven
That build special-cases knights with a half-square nudge. Doing the nudge for
EVERY piece, which is what the exit-corner code below does, makes knights stop
being a special case at all.


THE RISK THIS WHOLE MODULE IS BETTING AGAINST
---------------------------------------------
Lane routing assumes our magnet grabs exactly one piece. One reference build
found out the hard way that theirs did not, and abandoned lane routing:

    "Note that moving pieces between tiles does not work given our hardware
     constraints. The magnet is powerful enough that it will start dragging
     along other adjacent pieces."
    https://github.com/TripleSteak/Automatic-Chessboard  chess_algorithm.c ~line 882

Their fallback was to route through square centres and run a graph search that
physically evacuates any blocker out of the way first. That is a much bigger
program and it makes every move slower, so we are not writing it now.

IF THE DECEMBER MOCK-UP SHOWS THE SAME PROBLEM, THIS FILE IS WHERE THE FALLBACK
GOES. Two smaller retreats are available before the full graph search:
  1. Route drags around the OUTSIDE of the board (the lanes at index -1 and 8
     have pieces on one side only, or none at all) and accept the extra seconds.
  2. Shrink the piece bases. The lane clearance is
     SQUARE_MM/2 - PIECE_BASE_MAX_MM = 28.575 - 26.0 = 2.575 mm per side today.
Measure the pull-off distance on the mock-up before choosing.
"""

import chess

import config


# --- Constants that are NOT yet in config.py --------------------------------
#
# House rule: physical numbers live in config.py. These two describe the
# capture area, whose layout is still an open question (config marks it
# CAPTURE_AREA_PROVISIONAL), so they sit here until the team decides. BOTH
# SHOULD MOVE TO config.py once the capture area is designed.

# Where slot 0 sits. The strip is centred on the board's depth so the longest
# haul to a slot is as short as possible. Derived, not measured.
CAPTURE_SLOT_COUNT = 2 * config.CAPTURE_SLOTS_PER_SIDE
_STRIP_SPAN_MM = (CAPTURE_SLOT_COUNT - 1) * config.CAPTURE_SLOT_PITCH_MM
_BOARD_MID_Y_MM = (config.BOARD_RANKS - 1) * config.SQUARE_MM / 2.0
CAPTURE_SLOT_Y0_MM = _BOARD_MID_Y_MM - _STRIP_SPAN_MM / 2.0

# Which half of the strip a colour uses. White's losses stack nearest White's
# end of the board so a player can see their own graveyard without leaning over.
CAPTURE_FIRST_SLOT = {chess.WHITE: 0, chess.BLACK: config.CAPTURE_SLOTS_PER_SIDE}

# Two points count as the same point if they are closer than this. Not a
# physical distance -- it exists because 3.5 * 57.15 computed two different ways
# does not always land on the same float.
_EPS_MM = 1e-6


PARK_XY = (config.PARK_X_MM, config.PARK_Y_MM)

# Where the carriage ends up after the motor controller has run its limit-switch
# seek. config.HOMING_OFFSET_*_MM is the offset from that trigger point to the
# centre of a1, and both are 0.0 and PROVISIONAL until the frame exists.
HOME_XY = (config.HOMING_OFFSET_X_MM, config.HOMING_OFFSET_Y_MM)

# The lane just off the +X edge of the board. Captured pieces run out to here
# and then travel up or down it, because a rim lane has pieces on one side only.
_RIM_LANE_FILE = config.BOARD_FILES - 1


# A custom exception is just a class that inherits from an existing error type.
# Raising one lets a caller write "except PlannerError" and catch our problems
# without also catching, say, a typo that produced a TypeError.
class PlannerError(RuntimeError):
    """The planner was asked for something it cannot express as segments."""


class CaptureAreaFullError(PlannerError):
    """Every slot in the capture strip already holds a piece."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
#
# motion/corexy.py owns the canonical version of square_xy() and lane_xy().
# It is being written in parallel with this file, so the planner keeps its own
# copy rather than importing a module that may not exist yet. COLLAPSE THESE
# INTO corexy.py ONCE BOTH MODULES ARE IN. The arithmetic is fixed by the
# coordinate system in config.py, so the two copies cannot drift silently --
# if they ever disagree, one of them is simply wrong.


def square_xy(square):
    """Centre of a python-chess square (0..63, a1 == 0) in millimetres."""
    return (chess.square_file(square) * config.SQUARE_MM,
            chess.square_rank(square) * config.SQUARE_MM)


def lane_xy(file_index, rank_index):
    """A lane crossing on the half-square grid. Indices may be -1..BOARD_FILES.

    Index f means "the lane between file f and file f+1", so -1 is the lane
    running along the outside of the a file and 7 is the one along the outside
    of the h file.
    """
    return ((file_index + 0.5) * config.SQUARE_MM,
            (rank_index + 0.5) * config.SQUARE_MM)


def _same_point(a, b):
    return abs(a[0] - b[0]) < _EPS_MM and abs(a[1] - b[1]) < _EPS_MM


def _drop_repeats(points):
    """Remove consecutive duplicates so no segment contains a zero-length hop."""
    kept = []
    for point in points:
        if not kept or not _same_point(kept[-1], point):
            kept.append(point)
    return kept


def _side_toward(index, target_index, count):
    """Which side of a square we step off: +1 = higher indices, -1 = lower.

    When the move has no travel on this axis at all there is no obvious side,
    so we hug the nearer rim of the board. A lane on the rim has pieces on one
    side only, which is half the chances to knock something over.
    """
    if target_index > index:
        return 1
    if target_index < index:
        return -1
    return -1 if index <= (count - 1) / 2.0 else 1


def _corner_lane_index(square_index, side):
    """The lane index on the given side of a square."""
    return square_index if side > 0 else square_index - 1


def drag_path(from_square, to_square):
    """The full point list for dragging a piece from one square to another.

    start centre -> exit corner -> (at most one lane turn) -> entry corner
    -> destination centre.

    The exit corner is the corner of the start square that faces the
    destination; the entry corner is the corner of the destination square that
    faces where we came from. For a knight those two corners differ on exactly
    one axis, so the whole lane run collapses to a single straight leg -- which
    is the eight-line trick from the ChessBoard build, arrived at by accident.
    """
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    exit_file_lane = _corner_lane_index(
        from_file, _side_toward(from_file, to_file, config.BOARD_FILES))
    exit_rank_lane = _corner_lane_index(
        from_rank, _side_toward(from_rank, to_rank, config.BOARD_RANKS))
    entry_file_lane = _corner_lane_index(
        to_file, _side_toward(to_file, from_file, config.BOARD_FILES))
    entry_rank_lane = _corner_lane_index(
        to_rank, _side_toward(to_rank, from_rank, config.BOARD_RANKS))

    # An L, not a diagonal. A diagonal drag between two lane crossings runs at
    # 45 degrees straight across the corner where four squares meet, so it gets
    # close to four pieces at once instead of two. Which leg goes first is
    # arbitrary -- both are the same length and both make one turn -- so we
    # always run the X leg first and keep the planner deterministic.
    points = [square_xy(from_square),
              lane_xy(exit_file_lane, exit_rank_lane),
              lane_xy(entry_file_lane, exit_rank_lane),
              lane_xy(entry_file_lane, entry_rank_lane),
              square_xy(to_square)]
    return _drop_repeats(points)


# ---------------------------------------------------------------------------
# Capture area
# ---------------------------------------------------------------------------


def capture_slot_xy(slot):
    """Centre of a capture slot, slot 0 nearest White's end of the strip."""
    if not 0 <= slot < CAPTURE_SLOT_COUNT:
        raise PlannerError("capture slot %r is outside 0..%d"
                           % (slot, CAPTURE_SLOT_COUNT - 1))
    return (config.CAPTURE_X_MM,
            CAPTURE_SLOT_Y0_MM + slot * config.CAPTURE_SLOT_PITCH_MM)


def next_capture_slot(used_slots, color):
    """First free slot for a captured piece of this colour.

    used_slots is any collection of slot numbers already holding a piece --
    a set, a list, whatever. Scanning for the first number that is NOT in it is
    the whole anti-collision mechanism: a reference build had no allocator at
    all and stacked two captured pieces on the same spot.

    A colour prefers its own half of the strip. It spills into the other half
    when its own half is full, because CAPTURE_SLOTS_PER_SIDE is 8 and one side
    can lose 15 pieces -- see the module report for that open question.
    """
    taken = set(used_slots)
    first = CAPTURE_FIRST_SLOT[color]
    preferred = range(first, first + config.CAPTURE_SLOTS_PER_SIDE)
    for slot in preferred:
        if slot not in taken:
            return slot
    for slot in range(CAPTURE_SLOT_COUNT):
        if slot not in taken:
            return slot
    raise CaptureAreaFullError(
        "all %d capture slots are full; the board needs clearing by hand"
        % CAPTURE_SLOT_COUNT)


def used_capture_slots(board):
    """Which slots are already occupied, worked out from the position.

    plan_move() is only handed a board, so it has to infer the graveyard. A
    side starts with 16 pieces, so 16 minus what is left is what has been
    captured, and replaying the allocator that many times reproduces the slots.
    Promotions do not fool this because it counts pieces, not piece types.

    Two assumptions, both of which a caller can sidestep by tracking the real
    capture history and passing plan_move(used_slots=...) itself:

      * The strip holds 16 and a game can produce 30 captures, so the tray HAS
        to be emptied by hand partway through a long game. The modulo below is
        that assumption written down: once a colour's half has filled, we treat
        it as cleared and start again at its first slot. Without this the
        planner would refuse to plan an endgame at all.
      * Colours are replayed white first, which only changes the answer once a
        colour has spilled out of its own half.
    """
    used = set()
    for color in (chess.WHITE, chess.BLACK):
        captured = 16 - chess.popcount(board.occupied_co[color])
        for _ in range(captured % config.CAPTURE_SLOTS_PER_SIDE):
            used.add(next_capture_slot(used, color))
    return used


def removal_path(victim_square, slot):
    """Point list for hauling a captured piece off the board into a slot.

    Out to the +X rim first, THEN up or down. Getting clear of the 64 squares
    early means the long leg of the haul runs where there is nothing to hit.
    The final hop east into the slot is off the board entirely, so it cannot
    disturb the pieces already parked in the strip.
    """
    slot_x, slot_y = capture_slot_xy(slot)
    victim_file = chess.square_file(victim_square)
    victim_rank = chess.square_rank(victim_square)
    victim_x, victim_y = square_xy(victim_square)

    # Leave the square toward +X, because that is where the strip is.
    exit_file_lane = _corner_lane_index(victim_file, 1)
    if slot_y > victim_y:
        rank_side = 1
    elif slot_y < victim_y:
        rank_side = -1
    else:
        rank_side = _side_toward(victim_rank, victim_rank, config.BOARD_RANKS)
    exit_rank_lane = _corner_lane_index(victim_rank, rank_side)

    rim_x, lane_y = lane_xy(_RIM_LANE_FILE, exit_rank_lane)
    points = [(victim_x, victim_y),
              lane_xy(exit_file_lane, exit_rank_lane),
              (rim_x, lane_y),
              (rim_x, slot_y),
              (slot_x, slot_y)]
    return _drop_repeats(points)


# ---------------------------------------------------------------------------
# Move classification
# ---------------------------------------------------------------------------


def _classify_move(board, move):
    """Same answer game/rules.py's classify_move() gives, worked out locally.

    THIS IS A DELIBERATE DUPLICATE. game/rules.py is being written at the same
    time as this file, so importing it would mean the planner could not be
    tested on its own. DELETE THIS FUNCTION AND IMPORT rules.classify_move()
    ONCE BOTH MODULES EXIST -- the returned dictionary is deliberately the same
    shape, so the swap is a one-line change.
    """
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise PlannerError("no piece on %s to move"
                           % chess.square_name(move.from_square))

    captured_square = None
    rook_from = None
    rook_to = None
    promotion_piece = None

    if move.promotion is not None:
        promotion_piece = chess.piece_symbol(move.promotion)

    if board.is_castling(move):
        kind = "castle"
        rook_from, rook_to, _king_to = _castling_squares(board, move)
    else:
        if board.is_en_passant(move):
            # The captured pawn is NOT on the destination square. It is beside
            # the moving pawn: same file as the destination, same rank as the
            # start. Getting this wrong leaves a dead pawn standing on the board.
            captured_square = chess.square(chess.square_file(move.to_square),
                                           chess.square_rank(move.from_square))
            kind = "en_passant"
        elif board.piece_at(move.to_square) is not None:
            captured_square = move.to_square
            kind = "capture_promotion" if promotion_piece else "capture"
        else:
            kind = "promotion" if promotion_piece else "simple"

    return {"kind": kind,
            "captured_square": captured_square,
            "rook_from": rook_from,
            "rook_to": rook_to,
            "promotion_piece": promotion_piece}


def _castling_squares(board, move):
    """(rook_from, rook_to, king_to) for a castling move.

    Do not trust move.to_square here. Standard chess writes castling as
    e1 -> g1, but python-chess will also accept the "king takes own rook"
    spelling e1 -> h1, and those two mean the same move with different
    destinations. Both agree on which SIDE we are castling, so derive
    everything from the side.
    """
    rank = chess.square_rank(move.from_square)
    kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)

    rook = board.piece_at(move.to_square)
    if rook is not None and rook.piece_type == chess.ROOK and rook.color == board.turn:
        rook_from = move.to_square
    else:
        rook_from = chess.square(config.BOARD_FILES - 1 if kingside else 0, rank)

    if kingside:
        return rook_from, chess.square(5, rank), chess.square(6, rank)
    return rook_from, chess.square(3, rank), chess.square(2, rank)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def _travel(from_xy, to_xy, why):
    """A straight empty run. The magnet is retracted, so nothing can be hit."""
    return {"kind": "travel",
            "points": [tuple(from_xy), tuple(to_xy)],
            "speed_mms": config.TRAVEL_SPEED_MMS,
            "why": why}


def _drag(points, why):
    return {"kind": "drag",
            "points": [tuple(point) for point in points],
            "speed_mms": config.DRAG_SPEED_MMS,
            "why": why}


def _name(piece):
    color = "white" if piece.color == chess.WHITE else "black"
    return "%s %s" % (color, chess.piece_name(piece.piece_type))


def plan_move(board, move, start_xy=None, used_slots=None):
    """Segments that make `move` happen on the physical board.

    board is the position BEFORE the move; do not push it first, because the
    planner needs to see the piece that is about to be captured.

    start_xy says where the carriage is now and defaults to the park position,
    which is where every plan leaves it.
    """
    if board.chess960:
        # Chess960 castling can put the rook and king almost anywhere, and the
        # king sometimes does not move at all. Nothing here is built for it.
        raise PlannerError("chess960 castling is not supported")

    info = _classify_move(board, move)
    mover = board.piece_at(move.from_square)
    here = PARK_XY if start_xy is None else tuple(start_xy)
    segments = []

    # RULE: removal first, always. If the capturing piece went first there
    # would be two pieces on one square and no way to tell them apart.
    if info["captured_square"] is not None:
        victim_square = info["captured_square"]
        victim = board.piece_at(victim_square)
        if victim is None:
            raise PlannerError("nothing to capture on %s"
                               % chess.square_name(victim_square))
        slot = next_capture_slot(
            used_capture_slots(board) if used_slots is None else used_slots,
            victim.color)
        haul = removal_path(victim_square, slot)
        note = "" if info["kind"] != "en_passant" else " (en passant, not on %s)" % chess.square_name(move.to_square)
        segments.append(_travel(
            here, haul[0],
            "go to the captured %s on %s%s"
            % (_name(victim), chess.square_name(victim_square), note)))
        segments.append(_drag(
            haul,
            "haul the captured %s to slot %d" % (_name(victim), slot)))
        here = haul[-1]

    if info["kind"] == "castle":
        # Rook first. Once the rook has landed the king's route is a straight
        # run along the lane outside its own rank, with the rook already parked
        # clear of it. King first leaves the rook needing to get around a king
        # that is standing in the middle of its path.
        rook_from = info["rook_from"]
        rook_to = info["rook_to"]
        _rf, _rt, king_to = _castling_squares(board, move)
        rook = board.piece_at(rook_from)
        rook_leg = drag_path(rook_from, rook_to)
        segments.append(_travel(here, rook_leg[0],
                                "go to the %s on %s" % (_name(rook), chess.square_name(rook_from))))
        segments.append(_drag(rook_leg,
                              "castling, rook first: %s %s -> %s"
                              % (_name(rook), chess.square_name(rook_from),
                                 chess.square_name(rook_to))))
        here = rook_leg[-1]

        king_leg = drag_path(move.from_square, king_to)
        segments.append(_travel(here, king_leg[0],
                                "go to the %s on %s" % (_name(mover), chess.square_name(move.from_square))))
        segments.append(_drag(king_leg,
                              "castling, king second: %s %s -> %s"
                              % (_name(mover), chess.square_name(move.from_square),
                                 chess.square_name(king_to))))
        here = king_leg[-1]
    else:
        main = drag_path(move.from_square, move.to_square)
        segments.append(_travel(here, main[0],
                                "go to the %s on %s"
                                % (_name(mover), chess.square_name(move.from_square))))
        why = "%s %s -> %s" % (_name(mover), chess.square_name(move.from_square),
                               chess.square_name(move.to_square))
        if info["promotion_piece"] is not None:
            # The board cannot manufacture a queen, so the pawn is driven to the
            # last rank as usual and a human swaps it. The UI reads this text.
            why += (" -- PROMOTION: swap the pawn on %s for a %s by hand"
                    % (chess.square_name(move.to_square),
                       chess.piece_name(move.promotion)))
        segments.append(_drag(main, why))
        here = main[-1]

    segments.append(_travel(here, PARK_XY,
                            "park clear of the 64 squares so the sensor grid can be trusted"))
    return segments


def plan_park(start_xy=None):
    """Get the carriage off the squares.

    Rule 3 of the build: the sensor grid cannot be trusted while the carriage
    is under the board, so occupancy is only read with the carriage parked and
    the park position is off the 64 squares.

    With no argument this returns a travel segment that already starts at the
    park position. That is a no-op to execute and still safe, because a travel
    segment is an absolute move with the magnet retracted. Pass
    driver.position() when you know it.
    """
    here = PARK_XY if start_xy is None else tuple(start_xy)
    return [_travel(here, PARK_XY, "park clear of the 64 squares")]


def plan_home(start_xy=None):
    """What the planner contributes to a homing cycle: the retreat afterwards.

    The limit-switch seek itself is not in here. The Pi never generates step
    pulses, so homing is a single command to the motor controller and the
    driver owns it. By the time this plan runs the carriage is sitting on the
    home reference point, which is on top of square a1 and therefore exactly
    where it must not stay.
    """
    here = HOME_XY if start_xy is None else tuple(start_xy)
    return [_travel(here, PARK_XY, "leave the home corner and park off the squares")]


# ---------------------------------------------------------------------------
# Checks a caller can run on a plan
# ---------------------------------------------------------------------------


def segments_are_chained(segments, tol_mm=_EPS_MM):
    """True if every segment starts where the previous one ended.

    A break here means the executor would silently jump the carriage across the
    board, and if the magnet happened to be engaged it would plough through
    whatever was in the way.
    """
    for index in range(1, len(segments)):
        previous_end = segments[index - 1]["points"][-1]
        this_start = segments[index]["points"][0]
        if (abs(previous_end[0] - this_start[0]) > tol_mm
                or abs(previous_end[1] - this_start[1]) > tol_mm):
            return False
    return True


def plan_length_mm(segments):
    """Total distance the carriage covers, travel and drag together."""
    total = 0.0
    for segment in segments:
        points = segment["points"]
        for index in range(1, len(points)):
            dx = points[index][0] - points[index - 1][0]
            dy = points[index][1] - points[index - 1][1]
            total += (dx * dx + dy * dy) ** 0.5
    return total


def estimate_duration_s(segments):
    """Roughly how long a plan takes, for the move-budget check and the UI.

    Ignores acceleration, so it is an optimistic floor rather than a promise --
    the carriage never reaches top speed over a single 57 mm square. Each drag
    also costs the magnet settle times from config, which are real seconds and
    add up on a capture.
    """
    seconds = 0.0
    for segment in segments:
        points = segment["points"]
        length = 0.0
        for index in range(1, len(points)):
            dx = points[index][0] - points[index - 1][0]
            dy = points[index][1] - points[index - 1][1]
            length += (dx * dx + dy * dy) ** 0.5
        seconds += length / segment["speed_mms"]
        if segment["kind"] == "drag":
            seconds += config.MAGNET_ENGAGE_SETTLE_S + config.MAGNET_RELEASE_SETTLE_S
    return seconds
