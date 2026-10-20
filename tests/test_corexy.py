"""Tests for the CoreXY coordinate and kinematics maths.

Pure maths, so these run anywhere: no hardware, no engine, no sleeping.
    python -m pytest tests/test_corexy.py -q

The numbers in here are written out longhand on purpose. If someone changes
SQUARE_MM these tests should fail loudly rather than quietly agreeing with
whatever the new value is.
"""

import math

import chess
import pytest

import config
from motion import corexy


# pytest.approx compares floats with a tolerance instead of demanding they match
# bit for bit, which they never do after a couple of multiplications. abs=1e-9 mm
# is a millionth of a millimetre -- far tighter than the machine could ever be.
TOL = 1e-9


# ---------------------------------------------------------------------------
# Square centres
# ---------------------------------------------------------------------------

def test_a1_is_the_origin():
    assert corexy.square_to_xy(chess.A1) == pytest.approx((0.0, 0.0), abs=TOL)


def test_h8_is_the_far_corner():
    # 7 squares of 57.15 mm in each direction.
    assert corexy.square_to_xy(chess.H8) == pytest.approx((400.05, 400.05), abs=TOL)


def test_the_other_two_corners():
    assert corexy.square_to_xy(chess.H1) == pytest.approx((400.05, 0.0), abs=TOL)
    assert corexy.square_to_xy(chess.A8) == pytest.approx((0.0, 400.05), abs=TOL)


def test_e4_is_four_files_right_and_three_ranks_up():
    assert corexy.square_to_xy(chess.E4) == pytest.approx((228.6, 171.45), abs=TOL)


def test_named_squares_off_the_diagonal():
    # Hand-computed from the square names, not from file/rank arithmetic, so a
    # transposed board cannot agree with the test by using the same wrong maths.
    # b7 is 1 file right and 6 ranks up; g2 is 6 files right and 1 rank up.
    assert corexy.square_to_xy(chess.parse_square("b7")) == pytest.approx(
        (57.15, 342.9), abs=TOL
    )
    assert corexy.square_to_xy(chess.parse_square("g2")) == pytest.approx(
        (342.9, 57.15), abs=TOL
    )


def test_every_square_matches_the_file_rank_formula():
    # The failure this guards against is a mirrored or transposed board, which
    # looks fine on the corners of a symmetric layout and is wrong everywhere else.
    for square in chess.SQUARES:
        x, y = corexy.square_to_xy(square)
        assert x == pytest.approx(chess.square_file(square) * config.SQUARE_MM, abs=TOL)
        assert y == pytest.approx(chess.square_rank(square) * config.SQUARE_MM, abs=TOL)


def test_adjacent_squares_are_one_square_apart():
    x_a1, y_a1 = corexy.square_to_xy(chess.A1)
    x_b1, y_b1 = corexy.square_to_xy(chess.B1)
    x_a2, y_a2 = corexy.square_to_xy(chess.A2)
    assert x_b1 - x_a1 == pytest.approx(config.SQUARE_MM, abs=TOL)
    assert y_b1 == pytest.approx(y_a1, abs=TOL)
    assert y_a2 - y_a1 == pytest.approx(config.SQUARE_MM, abs=TOL)
    assert x_a2 == pytest.approx(x_a1, abs=TOL)


def test_square_index_outside_the_board_is_rejected():
    with pytest.raises(ValueError):
        corexy.square_to_xy(-1)
    with pytest.raises(ValueError):
        corexy.square_to_xy(64)


def test_file_rank_to_xy_rejects_off_board_indices():
    with pytest.raises(ValueError):
        corexy.file_rank_to_xy(8, 0)
    with pytest.raises(ValueError):
        corexy.file_rank_to_xy(0, -1)


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------

def test_lane_north_east_of_a1():
    assert corexy.lane_xy(0, 0) == pytest.approx((28.575, 28.575), abs=TOL)


def test_lane_sits_halfway_between_two_square_centres():
    for file_index in range(config.BOARD_FILES - 1):
        left, _ = corexy.file_rank_to_xy(file_index, 0)
        right, _ = corexy.file_rank_to_xy(file_index + 1, 0)
        lane_x, _ = corexy.lane_xy(file_index, 0)
        assert lane_x == pytest.approx((left + right) / 2.0, abs=TOL)


def test_every_lane_clears_every_square_centre_by_half_a_square_on_both_axes():
    # Rule 1 of the build: pieces route between squares, never through centres.
    #
    # The clearance has to be measured per AXIS, not as a straight-line distance.
    # A piece sitting on a centre is a disc, and what stops the dragged piece
    # from clipping it is how far the lane runs to the side of it in x and in y
    # separately. A straight-line check is far too generous: a lane offset of a
    # quarter square instead of a half would still be 0.35 squares away as the
    # crow flies and would pass, while physically scraping every neighbour.
    centres = [corexy.square_to_xy(square) for square in chess.SQUARES]
    half = config.SQUARE_MM / 2.0
    for file_index in range(-1, config.BOARD_FILES + 1):
        for rank_index in range(-1, config.BOARD_RANKS + 1):
            x, y = corexy.lane_xy(file_index, rank_index)
            for cx, cy in centres:
                assert abs(x - cx) >= half - TOL
                assert abs(y - cy) >= half - TOL


def test_a_lane_is_exactly_half_a_square_up_and_right_of_its_square():
    # Pins the half-square offset itself. The clearance test above says "at
    # least"; this one says "exactly", so a lane grid shifted by a third of a
    # square cannot sneak through both.
    for file_index in range(config.BOARD_FILES):
        for rank_index in range(config.BOARD_RANKS):
            cx, cy = corexy.file_rank_to_xy(file_index, rank_index)
            lx, ly = corexy.lane_xy(file_index, rank_index)
            assert lx - cx == pytest.approx(config.SQUARE_MM / 2.0, abs=TOL)
            assert ly - cy == pytest.approx(config.SQUARE_MM / 2.0, abs=TOL)


def test_a_square_centre_is_the_middle_of_its_four_surrounding_lanes():
    # Independent cross-check between the two coordinate functions: whatever the
    # square size is, a centre must sit dead in the middle of the four lane
    # crossings that box it in. Catches a half-square offset in either function.
    for square in chess.SQUARES:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        corners = [
            corexy.lane_xy(file_index - 1, rank_index - 1),
            corexy.lane_xy(file_index, rank_index - 1),
            corexy.lane_xy(file_index - 1, rank_index),
            corexy.lane_xy(file_index, rank_index),
        ]
        mid_x = sum(point[0] for point in corners) / 4.0
        mid_y = sum(point[1] for point in corners) / 4.0
        assert (mid_x, mid_y) == pytest.approx(corexy.square_to_xy(square), abs=TOL)


def test_the_two_ends_of_the_lane_ring_are_not_mirror_images():
    # The board's own edges are at -0.5 and +7.5 squares, so the lane that hugs
    # the board is index -1 at the low end but index 7 at the high end. Index 8
    # is a whole square further out, past the playing area and past CAPTURE_X_MM.
    # Anyone who assumes the ring is symmetric will route a piece off the frame,
    # so the asymmetry is pinned here rather than left as a comment.
    assert corexy.lane_xy(-1, -1) == pytest.approx(
        (-config.SQUARE_MM / 2.0, -config.SQUARE_MM / 2.0), abs=TOL
    )
    assert corexy.lane_xy(7, 7) == pytest.approx((428.625, 428.625), abs=TOL)
    east_edge_of_h_file = corexy.square_to_xy(chess.H1)[0] + config.SQUARE_MM / 2.0
    assert corexy.lane_xy(7, 0)[0] == pytest.approx(east_edge_of_h_file, abs=TOL)
    assert corexy.lane_xy(8, 0)[0] == pytest.approx(
        east_edge_of_h_file + config.SQUARE_MM, abs=TOL
    )
    assert corexy.lane_xy(8, 0)[0] > config.PLAYING_AREA_MM
    assert corexy.lane_xy(8, 0)[0] > config.CAPTURE_X_MM


def test_outer_ring_of_lanes_exists():
    # Rank -1 is the lane below rank 1. That is how a captured piece leaves the
    # board on its way to the capture area, so it has to be addressable.
    assert corexy.lane_xy(-1, -1) == pytest.approx((-28.575, -28.575), abs=TOL)
    assert corexy.lane_xy(8, 8) == pytest.approx((485.775, 485.775), abs=TOL)
    assert corexy.lane_xy(3, -1) == pytest.approx((200.025, -28.575), abs=TOL)
    assert corexy.lane_xy(-1, 4) == pytest.approx((-28.575, 257.175), abs=TOL)


def test_all_lane_indices_from_minus_one_to_eight_are_accepted():
    for file_index in range(-1, 9):
        for rank_index in range(-1, 9):
            x, y = corexy.lane_xy(file_index, rank_index)
            assert isinstance(x, float)
            assert isinstance(y, float)


def test_lane_indices_beyond_the_outer_ring_are_rejected():
    with pytest.raises(ValueError):
        corexy.lane_xy(-2, 0)
    with pytest.raises(ValueError):
        corexy.lane_xy(9, 0)
    with pytest.raises(ValueError):
        corexy.lane_xy(0, -2)
    with pytest.raises(ValueError):
        corexy.lane_xy(0, 9)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def test_pure_x_move_turns_both_motors_the_same_way():
    assert corexy.to_motor_deltas(1.0, 0.0) == pytest.approx((1.0, 1.0), abs=TOL)


def test_pure_y_move_turns_the_motors_opposite_ways():
    assert corexy.to_motor_deltas(0.0, 1.0) == pytest.approx((1.0, -1.0), abs=TOL)


def test_one_motor_alone_gives_a_forty_five_degree_diagonal():
    # Motor A alone: dA = 10, dB = 0 -> equal parts X and Y.
    dx, dy = corexy.to_cartesian(10.0, 0.0)
    assert dx == pytest.approx(5.0, abs=TOL)
    assert dy == pytest.approx(5.0, abs=TOL)


def test_a_worked_example():
    assert corexy.to_motor_deltas(10.0, 4.0) == pytest.approx((14.0, 6.0), abs=TOL)
    assert corexy.to_cartesian(14.0, 6.0) == pytest.approx((10.0, 4.0), abs=TOL)


def test_standing_still_moves_no_motors():
    assert corexy.to_motor_deltas(0.0, 0.0) == pytest.approx((0.0, 0.0), abs=TOL)
    assert corexy.to_cartesian(0.0, 0.0) == pytest.approx((0.0, 0.0), abs=TOL)


SPREAD = [
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (3.0, 4.0),
    (-3.0, 4.0),
    (3.0, -4.0),
    (-3.0, -4.0),
    (config.SQUARE_MM, config.SQUARE_MM),
    (config.SQUARE_MM / 2.0, -config.SQUARE_MM),
    (400.05, 400.05),
    (-457.2, 0.0),
    (0.001, -0.002),
    (123.456, -78.9),
]


def test_forward_then_inverse_round_trips():
    for dx, dy in SPREAD:
        da, db = corexy.to_motor_deltas(dx, dy)
        assert corexy.to_cartesian(da, db) == pytest.approx((dx, dy), abs=TOL)


def test_inverse_then_forward_round_trips():
    # Same check starting from motor space, because a sign error in only one of
    # the two functions would still round-trip in the other direction.
    for da, db in SPREAD:
        dx, dy = corexy.to_cartesian(da, db)
        assert corexy.to_motor_deltas(dx, dy) == pytest.approx((da, db), abs=TOL)


def test_full_board_diagonal_round_trips_through_motor_space():
    x0, y0 = corexy.square_to_xy(chess.A1)
    x1, y1 = corexy.square_to_xy(chess.H8)
    da, db = corexy.to_motor_deltas(x1 - x0, y1 - y0)
    assert (da, db) == pytest.approx((800.1, 0.0), abs=TOL)
    assert corexy.to_cartesian(da, db) == pytest.approx((400.05, 400.05), abs=TOL)


# ---------------------------------------------------------------------------
# The sqrt(2) property -- the thing that eats a day of calibration if missed
# ---------------------------------------------------------------------------

def test_ab_vector_is_always_sqrt_two_times_the_cartesian_one():
    # Written as a test and not only as a comment, because a comment cannot fail.
    # The point is "for EVERY direction", so walk the whole circle.
    for degrees in range(0, 360, 5):
        radians = math.radians(degrees)
        dx = 100.0 * math.cos(radians)
        dy = 100.0 * math.sin(radians)
        da, db = corexy.to_motor_deltas(dx, dy)
        cartesian_length = math.hypot(dx, dy)
        ab_length = math.hypot(da, db)
        assert ab_length == pytest.approx(math.sqrt(2.0) * cartesian_length, abs=1e-9)


def test_sqrt_two_property_on_the_awkward_directions():
    for dx, dy in SPREAD:
        da, db = corexy.to_motor_deltas(dx, dy)
        assert math.hypot(da, db) == pytest.approx(
            corexy.AB_LENGTH_RATIO * math.hypot(dx, dy), abs=1e-9
        )


def test_ab_length_ratio_constant_really_is_sqrt_two():
    assert corexy.AB_LENGTH_RATIO == pytest.approx(1.4142135623730951, abs=1e-15)


def test_feedrate_is_scaled_up_so_the_carriage_moves_at_the_speed_we_asked_for():
    # grbl feeds along the A/B-space length, so a commanded F yields F/sqrt(2) of
    # real carriage speed. Ask for sqrt(2) times as much and it comes out right.
    for speed in (config.TRAVEL_SPEED_MMS, config.DRAG_SPEED_MMS, 1.0, 0.0):
        commanded = corexy.feedrate_for_speed_mms(speed)
        assert commanded / math.sqrt(2.0) == pytest.approx(speed, abs=TOL)


def test_feedrate_comes_back_in_mm_per_second_not_mm_per_minute():
    # Tripwire for a 60x mistake at integration. G-code's F word is mm per
    # MINUTE; every speed in this project is mm per SECOND, and this function
    # stays in mm per second. The G-code writer multiplies by 60. Pinned to a
    # literal so a future edit that quietly folds the 60 in here fails loudly.
    assert corexy.feedrate_for_speed_mms(100.0) == pytest.approx(141.4213562373095, abs=1e-9)
    assert corexy.feedrate_for_speed_mms(config.TRAVEL_SPEED_MMS) < 1000.0


def test_ab_path_length_is_sqrt_two_times_the_cartesian_path_length():
    points = [(0.0, 0.0), (57.15, 0.0), (57.15, 114.3), (-28.575, 114.3)]
    assert corexy.ab_path_length_mm(points) == pytest.approx(
        math.sqrt(2.0) * corexy.path_length_mm(points), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def test_one_square_of_travel_in_steps():
    # 57.15 mm * 80 steps/mm.
    assert corexy.mm_to_steps(config.SQUARE_MM) == 4572


def test_zero_mm_is_zero_steps():
    assert corexy.mm_to_steps(0.0) == 0


def test_steps_are_whole_numbers():
    for mm in (0.0, 1.0, 12.3456, -7.77, config.PLAYING_AREA_MM):
        steps = corexy.mm_to_steps(mm)
        assert isinstance(steps, int)
        assert not isinstance(steps, bool)


def test_rounding_is_symmetric_about_zero():
    # Rounding +0.5 up and -0.5 to zero would make the carriage creep in one
    # direction over a long game. Whatever we do to +x we must do to -x.
    half_step_mm = 0.5 / config.STEPS_PER_MM
    for mm in (0.001, 0.01, half_step_mm, 1.5 * half_step_mm, 3.3, 57.15, 123.456):
        assert corexy.mm_to_steps(-mm) == -corexy.mm_to_steps(mm)


def test_a_half_step_rounds_away_from_zero():
    half_step_mm = 0.5 / config.STEPS_PER_MM
    # Worth checking that this really is an exact tie in float, or the test is
    # only pretending to exercise the tie case.
    assert half_step_mm * config.STEPS_PER_MM == 0.5
    assert corexy.mm_to_steps(half_step_mm) == 1
    assert corexy.mm_to_steps(-half_step_mm) == -1


def test_a_hair_under_half_a_step_rounds_to_zero():
    # 0.49999999999999994 is the largest float below 0.5, and adding 0.5 to it
    # gives exactly 1.0 -- floats have no room to hold the difference. So the
    # usual int(floor(x + 0.5)) trick reports a whole step for a move that is
    # less than half a step, in both directions. Guard against that coming back.
    just_under_mm = 0.49999999999999994 / config.STEPS_PER_MM
    assert just_under_mm * config.STEPS_PER_MM < 0.5
    assert corexy.mm_to_steps(just_under_mm) == 0
    assert corexy.mm_to_steps(-just_under_mm) == 0


def test_steps_never_overshoot_by_more_than_half_a_step():
    # The real contract of a rounding function, checked over a spread wide
    # enough to catch a rule that is right near zero and wrong further out.
    for mm in (0.0, 0.003, 0.00624, 0.00626, 1.0, 12.3456, 57.15, 228.6, 400.05):
        for signed_mm in (mm, -mm):
            error_steps = abs(corexy.mm_to_steps(signed_mm) - signed_mm * config.STEPS_PER_MM)
            assert error_steps <= 0.5 + TOL


def test_steps_round_trip_back_to_millimetres():
    for mm in (0.0, 1.0, 57.15, 400.05, -228.6):
        steps = corexy.mm_to_steps(mm)
        # One step is 1/80 mm, so the round trip can be off by half a step.
        assert corexy.steps_to_mm(steps) == pytest.approx(mm, abs=0.5 / config.STEPS_PER_MM)


# ---------------------------------------------------------------------------
# Path length
# ---------------------------------------------------------------------------

def test_empty_and_single_point_paths_have_no_length():
    assert corexy.path_length_mm([]) == 0.0
    assert corexy.path_length_mm([(12.0, 34.0)]) == 0.0


def test_straight_line_length():
    assert corexy.path_length_mm([(0.0, 0.0), (30.0, 40.0)]) == pytest.approx(50.0, abs=TOL)


def test_length_adds_up_over_several_segments():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert corexy.path_length_mm(points) == pytest.approx(30.0, abs=TOL)


def test_repeated_point_adds_nothing():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
    assert corexy.path_length_mm(points) == pytest.approx(15.0, abs=TOL)


def test_a_closed_loop_has_length_but_no_displacement():
    points = [(0.0, 0.0), (57.15, 0.0), (57.15, 57.15), (0.0, 57.15), (0.0, 0.0)]
    assert corexy.path_length_mm(points) == pytest.approx(4 * config.SQUARE_MM, abs=TOL)


def test_lane_dogleg_is_longer_than_the_straight_line():
    # Routing along lanes buys clearance and costs distance. Worth knowing the
    # cost is real when someone is fitting a move inside the 4 s budget.
    start = corexy.square_to_xy(chess.B1)
    end = corexy.square_to_xy(chess.C3)
    via = corexy.lane_xy(1, 0)
    direct = corexy.path_length_mm([start, end])
    dogleg = corexy.path_length_mm([start, via, end])
    assert dogleg > direct


def test_park_position_is_off_the_board():
    # Rule 3: the carriage must not park on any of the 64 squares, or that square
    # reports occupied forever. Guarded here because this module is where the
    # square coordinates come from.
    #
    # A square is a SQUARE, so "outside it" is a per-axis test, not a
    # straight-line one. Measuring the distance to the centre would happily
    # approve a park spot sitting on the lane between d4 and e4 -- 34 mm from
    # both centres, comfortably past half a square as the crow flies, and
    # directly over the sensor grid in the middle of the board.
    half = config.SQUARE_MM / 2.0
    for square in chess.SQUARES:
        cx, cy = corexy.square_to_xy(square)
        outside_in_x = abs(config.PARK_X_MM - cx) > half
        outside_in_y = abs(config.PARK_Y_MM - cy) > half
        assert outside_in_x or outside_in_y, "park sits on %s" % chess.square_name(square)


def test_the_park_test_would_catch_a_park_spot_in_the_middle_of_the_board():
    # The check above is only worth having if it can fail, so prove it can. This
    # point is on the d/e lane at rank 4 -- off every centre, but squarely on the
    # board. The old straight-line version of the test passed it.
    bad_park = (3.5 * config.SQUARE_MM, 3.0 * config.SQUARE_MM)
    half = config.SQUARE_MM / 2.0
    on_the_board = False
    for square in chess.SQUARES:
        cx, cy = corexy.square_to_xy(square)
        if abs(bad_park[0] - cx) <= half + TOL and abs(bad_park[1] - cy) <= half + TOL:
            on_the_board = True
    assert on_the_board
