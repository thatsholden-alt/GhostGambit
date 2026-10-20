"""Tests for the path planner.

Runs on a bare laptop -- no hardware, no engine, no sleeping:
    python -m pytest tests/test_planner.py -q

The interesting tests here are not "does it return three dictionaries". They
are the physical ones: does a dragged piece stay off other pieces' squares,
does the victim leave before the capturer arrives, does the rook move before
the king.
"""

import math

import chess
import pytest

import config
from motion import planner


# How close a dragged piece may come to the centre of a square it is not
# starting on or ending on. Two bases touch at PIECE_BASE_MAX_MM between
# centres, so anything at or above that is a clean pass. Lane routing actually
# gives SQUARE_MM / 2 = 28.575 mm, so there is 2.575 mm of margin.
CLEARANCE_MM = config.PIECE_BASE_MAX_MM


# --- helpers ---------------------------------------------------------------


def distance_point_to_segment(point, start, end):
    """Shortest distance from a point to a straight line segment.

    Projects the point onto the infinite line, then clamps the projection to
    the ends of the segment so a point beside the segment's end measures to
    the end rather than to some spot off in space.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0.0 and dy == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def distance_to_path(point, points):
    return min(distance_point_to_segment(point, points[i - 1], points[i])
               for i in range(1, len(points)))


def same_point(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6


def assert_drag_clears_other_squares(segment):
    """No square centre except the drag's own endpoints comes near the path."""
    points = segment["points"]
    for square in chess.SQUARES:
        centre = planner.square_xy(square)
        if same_point(centre, points[0]) or same_point(centre, points[-1]):
            continue
        distance = distance_to_path(centre, points)
        assert distance >= CLEARANCE_MM, (
            "drag %r passes %.2f mm from the centre of %s"
            % (segment["why"], distance, chess.square_name(square)))


def is_lane_point(point):
    """True if a point sits on the half-square grid."""
    for value in point:
        offset = value / config.SQUARE_MM - 0.5
        if abs(offset - round(offset)) > 1e-9:
            return False
    return True


def positions():
    """A handful of real positions to run the invariants over."""
    boards = [chess.Board()]

    sicilian = chess.Board()
    for san in ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]:
        sicilian.push_san(san)
    boards.append(sicilian)

    # Both sides still able to castle either way, back ranks otherwise empty.
    boards.append(chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"))
    boards.append(chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"))

    # A lone pawn one step from promoting, with a knight it can take instead.
    boards.append(chess.Board("1n6/P6k/8/8/8/8/6K1/8 w - - 0 1"))

    boards.append(en_passant_board())
    return boards


def en_passant_board():
    """White pawn on e5, Black has just played f7-f5, so exf6 e.p. is legal."""
    board = chess.Board()
    for san in ["e4", "a6", "e5", "f5"]:
        board.push_san(san)
    return board


# --- geometry --------------------------------------------------------------


def test_square_centres_match_the_coordinate_system():
    assert planner.square_xy(chess.A1) == (0.0, 0.0)
    assert planner.square_xy(chess.H8) == pytest.approx(
        (7 * config.SQUARE_MM, 7 * config.SQUARE_MM))
    # e4 is file 4, rank 3.
    assert planner.square_xy(chess.E4) == pytest.approx(
        (4 * config.SQUARE_MM, 3 * config.SQUARE_MM))


def test_lane_points_sit_on_the_half_square_grid():
    assert planner.lane_xy(0, 0) == pytest.approx(
        (0.5 * config.SQUARE_MM, 0.5 * config.SQUARE_MM))
    # The rims: one lane outside the a file and one outside the h file.
    assert planner.lane_xy(-1, -1) == pytest.approx(
        (-0.5 * config.SQUARE_MM, -0.5 * config.SQUARE_MM))
    assert planner.lane_xy(7, 7) == pytest.approx(
        (7.5 * config.SQUARE_MM, 7.5 * config.SQUARE_MM))


def test_drag_starts_and_ends_on_square_centres_and_uses_lanes_between():
    points = planner.drag_path(chess.B1, chess.C3)
    assert points[0] == pytest.approx(planner.square_xy(chess.B1))
    assert points[-1] == pytest.approx(planner.square_xy(chess.C3))
    for point in points[1:-1]:
        assert is_lane_point(point), "%r is not on the lane grid" % (point,)


def test_a_pawn_push_leaves_by_a_corner_and_runs_one_lane():
    # e2-e4. The e file is index 4, which is on the h-file side of the middle,
    # so the pawn hugs the lane between e and f: out to the corner of e2/f2/e3/f3,
    # up that lane, then a cut into e4.
    points = planner.drag_path(chess.E2, chess.E4)
    assert points == [pytest.approx(planner.square_xy(chess.E2)),
                      pytest.approx(planner.lane_xy(4, 1)),
                      pytest.approx(planner.lane_xy(4, 2)),
                      pytest.approx(planner.square_xy(chess.E4))]


def test_a_move_between_diagonally_touching_squares_uses_the_shared_corner():
    points = planner.drag_path(chess.C1, chess.D2)
    assert len(points) == 3
    assert points[1] == pytest.approx(planner.lane_xy(2, 0))


def test_a_knight_hop_is_one_straight_lane_leg():
    # This is the whole point of the exit-corner rule: for a 2x1 hop the corner
    # we leave by and the corner we enter by differ on exactly one axis.
    points = planner.drag_path(chess.G1, chess.F3)
    assert len(points) == 4
    assert points[1] == pytest.approx(planner.lane_xy(5, 0))
    assert points[2] == pytest.approx(planner.lane_xy(5, 1))
    assert points[1][0] == pytest.approx(points[2][0])


def test_a_knight_leaving_the_opening_back_rank_clears_its_neighbours():
    board = chess.Board()
    segments = planner.plan_move(board, chess.Move.from_uci("g1f3"))
    drags = [s for s in segments if s["kind"] == "drag"]
    assert len(drags) == 1
    assert_drag_clears_other_squares(drags[0])
    # Spelled out for the squares that actually have pieces on them at the
    # start of a game, because this is the move the whole module exists for.
    for name in ["e1", "f1", "h1", "e2", "f2", "g2", "h2"]:
        square = chess.parse_square(name)
        distance = distance_to_path(planner.square_xy(square), drags[0]["points"])
        assert distance >= CLEARANCE_MM


def test_a_rook_on_the_a_file_hugs_the_outside_lane():
    # No sideways travel, so the tie-break sends it to the nearer rim, where
    # there are pieces on one side only.
    points = planner.drag_path(chess.A1, chess.A8)
    assert points[1][0] == pytest.approx(-0.5 * config.SQUARE_MM)
    assert points[-2][0] == pytest.approx(-0.5 * config.SQUARE_MM)


def test_drags_never_run_diagonally_between_lane_crossings():
    # Every leg between the first and last hop must be axis-aligned.
    points = planner.drag_path(chess.C1, chess.H6)
    for index in range(2, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        assert (abs(previous[0] - current[0]) < 1e-9
                or abs(previous[1] - current[1]) < 1e-9)


# --- classification --------------------------------------------------------


def test_classify_simple_and_capture():
    board = chess.Board()
    info = planner._classify_move(board, chess.Move.from_uci("e2e4"))
    assert info["kind"] == "simple"
    assert info["captured_square"] is None

    board = chess.Board()
    for san in ["e4", "d5"]:
        board.push_san(san)
    info = planner._classify_move(board, chess.Move.from_uci("e4d5"))
    assert info["kind"] == "capture"
    assert info["captured_square"] == chess.D5


def test_classify_en_passant_names_the_pawn_beside_the_mover():
    board = en_passant_board()
    move = board.parse_san("exf6")
    info = planner._classify_move(board, move)
    assert info["kind"] == "en_passant"
    assert move.to_square == chess.F6
    assert info["captured_square"] == chess.F5


def test_classify_castling_and_promotion():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    info = planner._classify_move(board, chess.Move.from_uci("e1g1"))
    assert info["kind"] == "castle"
    assert info["rook_from"] == chess.H1
    assert info["rook_to"] == chess.F1
    assert info["captured_square"] is None

    board = chess.Board("1n6/P6k/8/8/8/8/6K1/8 w - - 0 1")
    info = planner._classify_move(board, chess.Move.from_uci("a7a8q"))
    assert info["kind"] == "promotion"
    assert info["promotion_piece"] == "q"

    info = planner._classify_move(board, chess.Move.from_uci("a7b8q"))
    assert info["kind"] == "capture_promotion"
    assert info["captured_square"] == chess.B8


def test_planning_a_move_with_no_piece_on_it_is_an_error():
    board = chess.Board()
    with pytest.raises(planner.PlannerError):
        planner.plan_move(board, chess.Move.from_uci("e4e5"))


# --- capture slots ---------------------------------------------------------


def test_slots_are_spaced_by_the_configured_pitch():
    first = planner.capture_slot_xy(0)
    second = planner.capture_slot_xy(1)
    assert first[0] == pytest.approx(config.CAPTURE_X_MM)
    assert second[1] - first[1] == pytest.approx(config.CAPTURE_SLOT_PITCH_MM)


def test_the_capture_strip_sits_clear_of_the_playing_area():
    for slot in range(planner.CAPTURE_SLOT_COUNT):
        x, _y = planner.capture_slot_xy(slot)
        assert x > (config.BOARD_FILES - 1) * config.SQUARE_MM + config.SQUARE_MM / 2


def test_asking_for_a_slot_that_does_not_exist_is_an_error():
    with pytest.raises(planner.PlannerError):
        planner.capture_slot_xy(planner.CAPTURE_SLOT_COUNT)
    with pytest.raises(planner.PlannerError):
        planner.capture_slot_xy(-1)


def test_the_allocator_never_hands_out_the_same_slot_twice():
    used = set()
    for _ in range(config.CAPTURE_SLOTS_PER_SIDE):
        slot = planner.next_capture_slot(used, chess.WHITE)
        assert slot not in used
        used.add(slot)
    assert used == set(range(config.CAPTURE_SLOTS_PER_SIDE))


def test_the_two_colours_use_different_halves_of_the_strip():
    white = planner.next_capture_slot(set(), chess.WHITE)
    black = planner.next_capture_slot(set(), chess.BLACK)
    assert white != black
    assert black == config.CAPTURE_SLOTS_PER_SIDE


def test_a_colour_spills_into_the_other_half_rather_than_stacking():
    # A side can lose 15 pieces and only has 8 slots of its own, so spilling
    # has to work. What must never happen is two pieces on one slot.
    used = set()
    for _ in range(planner.CAPTURE_SLOT_COUNT):
        slot = planner.next_capture_slot(used, chess.WHITE)
        assert slot not in used
        used.add(slot)
    assert len(used) == planner.CAPTURE_SLOT_COUNT


def test_a_full_capture_area_raises_rather_than_stacking():
    used = set(range(planner.CAPTURE_SLOT_COUNT))
    with pytest.raises(planner.CaptureAreaFullError):
        planner.next_capture_slot(used, chess.BLACK)


def test_used_slots_are_read_back_from_the_position():
    board = chess.Board()
    assert planner.used_capture_slots(board) == set()

    for san in ["e4", "d5", "exd5", "Qxd5"]:
        board.push_san(san)
    # One black pawn and one white pawn have gone.
    assert planner.used_capture_slots(board) == {0, config.CAPTURE_SLOTS_PER_SIDE}


def test_two_captures_in_a_row_go_to_different_slots():
    board = chess.Board()
    for san in ["e4", "d5", "exd5"]:
        board.push_san(san)
    first_slot_used = planner.used_capture_slots(board)
    assert first_slot_used == {config.CAPTURE_SLOTS_PER_SIDE}

    segments = planner.plan_move(board, board.parse_san("Qxd5"))
    haul = [s for s in segments if s["kind"] == "drag"][0]
    # The white pawn on d5 is the victim now, so it takes White's first slot,
    # not the one the black pawn is already sitting in.
    assert haul["points"][-1] == pytest.approx(planner.capture_slot_xy(0))


def test_the_haul_to_the_capture_area_stays_off_other_squares():
    board = chess.Board()
    for san in ["e4", "d5"]:
        board.push_san(san)
    segments = planner.plan_move(board, board.parse_san("exd5"))
    haul = [s for s in segments if s["kind"] == "drag"][0]
    assert_drag_clears_other_squares(haul)
    # It leaves the board on the +X rim before running north or south.
    rim_x = 7.5 * config.SQUARE_MM
    assert max(point[0] for point in haul["points"][:-1]) >= rim_x - 1e-9


# --- whole plans -----------------------------------------------------------


def test_a_simple_move_is_travel_drag_travel_between_park_and_park():
    board = chess.Board()
    segments = planner.plan_move(board, chess.Move.from_uci("e2e4"))
    assert [s["kind"] for s in segments] == ["travel", "drag", "travel"]
    assert segments[0]["points"][0] == pytest.approx(planner.PARK_XY)
    assert segments[-1]["points"][-1] == pytest.approx(planner.PARK_XY)
    assert segments[1]["points"][0] == pytest.approx(planner.square_xy(chess.E2))
    assert segments[1]["points"][-1] == pytest.approx(planner.square_xy(chess.E4))


def test_speeds_come_from_config_and_differ_by_kind():
    board = chess.Board()
    for segment in planner.plan_move(board, chess.Move.from_uci("e2e4")):
        if segment["kind"] == "travel":
            assert segment["speed_mms"] == config.TRAVEL_SPEED_MMS
        else:
            assert segment["speed_mms"] == config.DRAG_SPEED_MMS


def test_a_plan_can_start_from_where_the_carriage_actually_is():
    board = chess.Board()
    start = (10.0, 20.0)
    segments = planner.plan_move(board, chess.Move.from_uci("e2e4"), start_xy=start)
    assert segments[0]["points"][0] == pytest.approx(start)
    assert planner.segments_are_chained(segments)


def test_a_capture_removes_the_victim_before_the_capturer_moves():
    board = chess.Board()
    for san in ["e4", "d5"]:
        board.push_san(san)
    segments = planner.plan_move(board, board.parse_san("exd5"))
    drags = [s for s in segments if s["kind"] == "drag"]
    assert len(drags) == 2

    removal, capture = drags
    assert removal["points"][0] == pytest.approx(planner.square_xy(chess.D5))
    assert removal["points"][-1][0] == pytest.approx(config.CAPTURE_X_MM)
    assert capture["points"][0] == pytest.approx(planner.square_xy(chess.E4))
    assert capture["points"][-1] == pytest.approx(planner.square_xy(chess.D5))
    # The capturing piece must not set off until the square is clear.
    assert segments.index(removal) < segments.index(capture)


def test_en_passant_drags_the_pawn_that_is_not_on_the_destination():
    board = en_passant_board()
    move = board.parse_san("exf6")
    segments = planner.plan_move(board, move)
    removal, capture = [s for s in segments if s["kind"] == "drag"]

    assert removal["points"][0] == pytest.approx(planner.square_xy(chess.F5))
    assert removal["points"][0] != pytest.approx(planner.square_xy(chess.F6))
    assert capture["points"][0] == pytest.approx(planner.square_xy(chess.E5))
    assert capture["points"][-1] == pytest.approx(planner.square_xy(chess.F6))
    # Belt and braces: after the plan, nothing should still be standing on f5.
    board.push(move)
    assert board.piece_at(chess.F5) is None


@pytest.mark.parametrize("fen,uci,king_from,king_to,rook_from,rook_to", [
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
     "e1g1", chess.E1, chess.G1, chess.H1, chess.F1),
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
     "e1c1", chess.E1, chess.C1, chess.A1, chess.D1),
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
     "e8g8", chess.E8, chess.G8, chess.H8, chess.F8),
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
     "e8c8", chess.E8, chess.C8, chess.A8, chess.D8),
])
def test_castling_moves_the_rook_first(fen, uci, king_from, king_to, rook_from, rook_to):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    segments = planner.plan_move(board, move)
    drags = [s for s in segments if s["kind"] == "drag"]
    assert len(drags) == 2

    rook_leg, king_leg = drags
    assert rook_leg["points"][0] == pytest.approx(planner.square_xy(rook_from))
    assert rook_leg["points"][-1] == pytest.approx(planner.square_xy(rook_to))
    assert king_leg["points"][0] == pytest.approx(planner.square_xy(king_from))
    assert king_leg["points"][-1] == pytest.approx(planner.square_xy(king_to))
    assert planner.segments_are_chained(segments)


def test_the_castling_king_runs_outside_its_own_rank():
    # With the rook already parked on f1, the king's route is a straight run
    # along the lane below rank 1 and then a cut into g1.
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    segments = planner.plan_move(board, chess.Move.from_uci("e1g1"))
    king_leg = [s for s in segments if s["kind"] == "drag"][1]
    for point in king_leg["points"][1:-1]:
        assert point[1] == pytest.approx(-0.5 * config.SQUARE_MM)


def test_castling_paths_clear_every_other_square():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    for uci in ["e1g1", "e1c1"]:
        for segment in planner.plan_move(board, chess.Move.from_uci(uci)):
            if segment["kind"] == "drag":
                assert_drag_clears_other_squares(segment)


def test_promotion_asks_a_human_to_swap_the_piece():
    board = chess.Board("1n6/P6k/8/8/8/8/6K1/8 w - - 0 1")
    segments = planner.plan_move(board, chess.Move.from_uci("a7a8q"))
    drags = [s for s in segments if s["kind"] == "drag"]
    assert len(drags) == 1
    assert "PROMOTION" in drags[0]["why"]
    assert "queen" in drags[0]["why"]
    assert drags[0]["points"][-1] == pytest.approx(planner.square_xy(chess.A8))


def test_a_capture_promotion_still_removes_the_victim_first():
    board = chess.Board("1n6/P6k/8/8/8/8/6K1/8 w - - 0 1")
    segments = planner.plan_move(board, chess.Move.from_uci("a7b8q"))
    drags = [s for s in segments if s["kind"] == "drag"]
    assert len(drags) == 2
    assert drags[0]["points"][0] == pytest.approx(planner.square_xy(chess.B8))
    assert drags[0]["points"][-1][0] == pytest.approx(config.CAPTURE_X_MM)
    assert "PROMOTION" in drags[1]["why"]


def test_planning_does_not_touch_the_board():
    board = en_passant_board()
    before = board.fen()
    planner.plan_move(board, board.parse_san("exf6"))
    assert board.fen() == before


# --- invariants over every legal move in several positions -----------------


def test_every_legal_move_produces_a_well_formed_chained_plan():
    for board in positions():
        for move in board.legal_moves:
            segments = planner.plan_move(board, move)
            label = "%s in %s" % (move.uci(), board.fen())

            assert segments, label
            assert planner.segments_are_chained(segments), label
            assert segments[0]["points"][0] == pytest.approx(planner.PARK_XY), label
            assert segments[-1]["kind"] == "travel", label
            assert segments[-1]["points"][-1] == pytest.approx(planner.PARK_XY), label

            for segment in segments:
                assert set(segment.keys()) == {"kind", "points", "speed_mms", "why"}
                assert segment["kind"] in ("travel", "drag")
                assert len(segment["points"]) >= 2
                assert isinstance(segment["why"], str) and segment["why"]
                for point in segment["points"]:
                    assert isinstance(point, tuple) and len(point) == 2


def test_every_legal_drag_stays_off_other_squares():
    for board in positions():
        for move in board.legal_moves:
            for segment in planner.plan_move(board, move):
                if segment["kind"] == "drag":
                    assert_drag_clears_other_squares(segment)


def test_every_legal_drag_touches_square_centres_only_at_its_ends():
    for board in positions():
        for move in board.legal_moves:
            for segment in planner.plan_move(board, move):
                if segment["kind"] != "drag":
                    continue
                for point in segment["points"][1:-1]:
                    # Intermediate points are on lanes, except once the haul to
                    # the capture strip has left the board on the +X side --
                    # out there the slot pitch has nothing to do with squares.
                    off_the_board = point[0] >= (config.BOARD_FILES - 0.5) * config.SQUARE_MM
                    assert is_lane_point(point) or off_the_board


def test_the_carriage_never_parks_on_a_square():
    # Rule 3: our own magnet reads as a piece, so the park position has to be
    # off the 64 squares or one square reports occupied forever.
    for square in chess.SQUARES:
        centre = planner.square_xy(square)
        assert math.hypot(centre[0] - planner.PARK_XY[0],
                          centre[1] - planner.PARK_XY[1]) > config.SQUARE_MM / 2


# --- park, home and the plan-level helpers ---------------------------------


def test_park_ends_at_the_park_position():
    segments = planner.plan_park((100.0, 100.0))
    assert len(segments) == 1
    assert segments[0]["kind"] == "travel"
    assert segments[0]["points"][0] == pytest.approx((100.0, 100.0))
    assert segments[0]["points"][-1] == pytest.approx(planner.PARK_XY)


def test_park_with_no_argument_is_a_harmless_no_op():
    segments = planner.plan_park()
    assert segments[0]["points"][0] == pytest.approx(planner.PARK_XY)
    assert segments[0]["points"][-1] == pytest.approx(planner.PARK_XY)


def test_home_leaves_the_home_corner_and_parks():
    segments = planner.plan_home()
    assert len(segments) == 1
    assert segments[0]["kind"] == "travel"
    assert segments[0]["points"][0] == pytest.approx(planner.HOME_XY)
    assert segments[0]["points"][-1] == pytest.approx(planner.PARK_XY)


def test_a_broken_chain_is_detected():
    good = planner.plan_move(chess.Board(), chess.Move.from_uci("e2e4"))
    assert planner.segments_are_chained(good)

    broken = planner.plan_move(chess.Board(), chess.Move.from_uci("e2e4"))
    broken[1]["points"][0] = (0.0, 0.0)
    assert not planner.segments_are_chained(broken)


def test_a_capture_costs_more_time_than_a_quiet_move():
    board = chess.Board()
    for san in ["e4", "d5"]:
        board.push_san(san)
    quiet = planner.estimate_duration_s(planner.plan_move(board, board.parse_san("Nf3")))
    capture = planner.estimate_duration_s(planner.plan_move(board, board.parse_san("exd5")))
    assert capture > quiet > 0.0


def test_plan_length_adds_up_the_legs():
    segments = planner.plan_park((planner.PARK_XY[0] + 100.0, planner.PARK_XY[1]))
    assert planner.plan_length_mm(segments) == pytest.approx(100.0)
