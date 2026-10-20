"""Tests for working out the human's move from the occupancy grid.

No hardware, no engine, no sleeping. Every position here is either a real
opening sequence or a FEN written out square by square, because the whole point
of the module is that it must survive real positions and not just tidy ones.

    python -m pytest tests/test_inference.py -q
"""

import chess
import pytest

from sensing import inference


# --- helpers -----------------------------------------------------------------

def board_after(moves_san):
    """A board after a list of moves in algebraic notation, e.g. ["e4", "d5"]."""
    board = chess.Board()
    for san in moves_san:
        board.push_san(san)
    return board


def sensors_would_read(board, uci):
    """The mask the grid would report if `uci` were played on `board`.

    This is the fake sensor for these tests: it works out the truth from the
    rules, exactly the way the real grid would report it once the move is made
    on the wooden board.
    """
    board.push(chess.Move.from_uci(uci))
    mask = inference._occupancy_from_board(board)
    board.pop()
    return mask


def bits_changed(mask_a, mask_b):
    return bin(mask_a ^ mask_b).count("1")


# --- the mask itself ----------------------------------------------------------

def test_bit_zero_is_a1_and_bit_63_is_h8():
    mask = inference._occupancy_from_board(chess.Board())
    assert mask & (1 << chess.A1)
    assert mask & (1 << chess.H8)
    assert not mask & (1 << chess.E4)


def test_starting_position_has_thirty_two_pieces():
    mask = inference._occupancy_from_board(chess.Board())
    assert bin(mask).count("1") == 32
    assert mask <= inference.FULL_BOARD_MASK


def test_mask_is_sixty_four_squares_wide():
    assert inference.SQUARE_COUNT == 64
    assert inference.FULL_BOARD_MASK == (1 << 64) - 1


# --- the easy cases -----------------------------------------------------------

def test_quiet_move_is_identified():
    board = chess.Board()
    move, reason = inference.infer_move(board, sensors_would_read(board, "e2e4"))
    assert move == chess.Move.from_uci("e2e4")
    assert reason


def test_knight_move_out_of_a_full_back_rank_is_identified():
    board = chess.Board()
    move, reason = inference.infer_move(board, sensors_would_read(board, "g1f3"))
    assert move == chess.Move.from_uci("g1f3")


def test_a_capture_changes_only_one_square_and_is_still_identified():
    # THE case that breaks the naive bitmap diff. After 1.e4 d5, exd5 empties
    # e4 but d5 was already occupied and stays occupied, so the sensor grid sees
    # exactly ONE square change. A decoder that reads "one square changed" as
    # "a piece was lifted" has no idea a capture happened.
    board = board_after(["e4", "d5"])
    before = inference._occupancy_from_board(board)
    observed = sensors_would_read(board, "e4d5")
    assert bits_changed(before, observed) == 1

    move, reason = inference.infer_move(board, observed)
    assert move == chess.Move.from_uci("e4d5")
    assert board.is_capture(move)


def test_en_passant_is_identified():
    board = board_after(["e4", "a6", "e5", "d5"])
    observed = sensors_would_read(board, "e5d6")
    move, reason = inference.infer_move(board, observed)
    assert move == chess.Move.from_uci("e5d6")
    assert board.is_en_passant(move)
    # The pawn that physically has to be removed is on d5, NOT on the
    # destination square d6. Three squares change, and nothing in this module
    # had to be told that.
    assert not observed & (1 << chess.D5)
    assert observed & (1 << chess.D6)
    assert bits_changed(inference._occupancy_from_board(board), observed) == 3


def test_black_en_passant_is_identified_too():
    # The mirror image of the test above, and worth its own test because en
    # passant is the one rule where the removed pawn sits on the OTHER side of
    # the destination depending on colour: white removes from below, black from
    # above. Legal-move matching never has to know that, which is the point.
    board = board_after(["a3", "d5", "a4", "d4", "e4"])
    assert board.turn == chess.BLACK
    observed = sensors_would_read(board, "d4e3")
    move, reason = inference.infer_move(board, observed)
    assert move == chess.Move.from_uci("d4e3")
    assert board.is_en_passant(move)
    # The white pawn that must be physically removed is on e4, one rank ABOVE
    # the destination e3.
    assert not observed & (1 << chess.E4)
    assert observed & (1 << chess.E3)


# --- castling, all four, including the one the reference build cannot see ------

CASTLING_FEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"


def test_all_four_castlings_are_identified():
    for fen, uci in [(CASTLING_FEN, "e1g1"),
                     (CASTLING_FEN, "e1c1"),
                     (CASTLING_FEN.replace(" w ", " b "), "e8g8"),
                     (CASTLING_FEN.replace(" w ", " b "), "e8c8")]:
        board = chess.Board(fen)
        move, reason = inference.infer_move(board, sensors_would_read(board, uci))
        assert move == chess.Move.from_uci(uci), reason
        assert board.is_castling(move)


def test_black_queenside_castling_is_identified():
    # Called out on its own because this is the exact move that the ut-ras
    # chessbot build can never recognise: its castling patterns are written
    # descending and its changed-square list is always ascending. Legal-move
    # matching has no patterns to get backwards.
    board = chess.Board(CASTLING_FEN.replace(" w ", " b "))
    observed = sensors_would_read(board, "e8c8")
    move, reason = inference.infer_move(board, observed)
    assert move == chess.Move.from_uci("e8c8")
    assert bits_changed(inference._occupancy_from_board(board), observed) == 4


# --- promotion ----------------------------------------------------------------

PROMOTION_FEN = "8/4P3/8/8/8/8/8/K6k w - - 0 1"


def test_promotion_is_identified_and_the_queen_is_assumed_out_loud():
    board = chess.Board(PROMOTION_FEN)
    move, reason = inference.infer_move(board, sensors_would_read(board, "e7e8q"))
    assert move == chess.Move.from_uci("e7e8q")
    assert move.promotion == chess.QUEEN
    # The message must admit it assumed, so the UI can ask the player.
    assert "assum" in reason.lower()


def test_promotion_piece_cannot_be_seen_and_strict_mode_refuses_to_guess():
    board = chess.Board(PROMOTION_FEN)
    observed = sensors_would_read(board, "e7e8q")
    # A queen and a knight on e8 light up the identical sensor.
    assert observed == sensors_would_read(board, "e7e8n")

    move, reason = inference.infer_move(board, observed, promotion_default=None)
    assert move is None
    assert reason.startswith("ambiguous")


def test_the_assumed_promotion_piece_is_named_honestly():
    # Regression. The reason string used to say "assuming a queen" no matter
    # what the caller asked for, so a caller that set the default to a rook got
    # a rook move described as a queen. The screen is the only place a wrong
    # promotion can be caught, so the screen must not lie.
    board = chess.Board(PROMOTION_FEN)
    observed = sensors_would_read(board, "e7e8q")

    move, reason = inference.infer_move(board, observed,
                                        promotion_default=chess.ROOK)
    assert move == chess.Move.from_uci("e7e8r")
    assert "rook" in reason
    assert "queen" not in reason

    move, reason = inference.infer_move(board, observed)
    assert move == chess.Move.from_uci("e7e8q")
    assert "queen" in reason


def test_capture_promotion_is_identified():
    board = chess.Board("5r2/4P3/8/8/8/8/8/K6k w - - 0 1")
    move, reason = inference.infer_move(board, sensors_would_read(board, "e7f8q"))
    assert move == chess.Move.from_uci("e7f8q")
    assert board.is_capture(move)


# --- ambiguity, constructed rather than assumed -------------------------------

AMBIGUOUS_FEN = "k7/8/2p1p3/8/3N4/8/8/7K w - - 0 1"


def test_the_ambiguous_position_really_is_ambiguous():
    # Knight on d4, black pawns on c6 and e6. Nxc6 and Nxe6 both empty d4 and
    # both leave the target occupied, so the two are the same 64 bits. Prove it
    # here rather than trusting the claim in the next test.
    board = chess.Board(AMBIGUOUS_FEN)
    assert sensors_would_read(board, "d4c6") == sensors_would_read(board, "d4e6")


def test_two_captures_by_the_same_piece_are_reported_as_ambiguous():
    board = chess.Board(AMBIGUOUS_FEN)
    move, reason = inference.infer_move(board, sensors_would_read(board, "d4c6"))
    assert move is None, "guessing here would desynchronise the physical board"
    assert reason.startswith("ambiguous")
    assert "Nxc6" in reason and "Nxe6" in reason


def test_an_unambiguous_move_in_the_same_position_still_works():
    board = chess.Board(AMBIGUOUS_FEN)
    move, reason = inference.infer_move(board, sensors_would_read(board, "d4f5"))
    assert move == chess.Move.from_uci("d4f5")


# --- nothing matches ----------------------------------------------------------

def test_a_piece_knocked_onto_an_empty_square_names_both_squares():
    board = chess.Board()
    observed = inference._occupancy_from_board(board)
    observed &= ~(1 << chess.B1)        # somebody's sleeve caught the knight
    observed |= 1 << chess.D4           # and it ended up in the middle

    move, reason = inference.infer_move(board, observed)
    assert move is None
    assert "b1 should have a piece" in reason
    assert "d4 should be empty" in reason


def test_a_piece_that_simply_vanished_is_reported():
    board = chess.Board()
    observed = inference._occupancy_from_board(board) & ~(1 << chess.A2)
    move, reason = inference.infer_move(board, observed)
    assert move is None
    assert "a2 should have a piece" in reason


def test_board_unchanged_says_so_instead_of_listing_nothing():
    board = chess.Board()
    move, reason = inference.infer_move(board, inference._occupancy_from_board(board))
    assert move is None
    assert "unchanged" in reason


def test_game_over_position_does_not_crash():
    board = board_after(["f3", "e5", "g4", "Qh4#"])
    assert board.is_game_over()
    move, reason = inference.infer_move(board, inference._occupancy_from_board(board))
    assert move is None
    assert reason


# --- describe_mismatch --------------------------------------------------------

def test_describe_mismatch_hints_the_closest_move_when_one_explains_most_of_it():
    # The player made a real move AND knocked a pawn off the board.
    board = chess.Board()
    observed = sensors_would_read(board, "e2e4") & ~(1 << chess.D7)
    text = inference.describe_mismatch(board, observed)
    assert "e4" in text
    assert "d7 should have a piece" in text
    # Only the leftover problem is listed, not e2 and e4 as well.
    assert "e2 should have a piece" not in text


def test_describe_mismatch_lists_plain_differences_when_no_move_is_close():
    board = chess.Board()
    observed = inference._occupancy_from_board(board)
    observed &= ~(1 << chess.B1)
    observed |= 1 << chess.D4
    text = inference.describe_mismatch(board, observed)
    assert text == "d4 should be empty; b1 should have a piece"


def test_describe_mismatch_stays_short_enough_for_a_five_inch_screen():
    board = chess.Board()
    text = inference.describe_mismatch(board, 0)     # every sensor reads empty
    assert "+28 more" in text
    assert len(text) < 80


def test_describe_mismatch_on_an_untouched_board():
    board = chess.Board()
    text = inference.describe_mismatch(board, inference._occupancy_from_board(board))
    assert "unchanged" in text


def test_describe_mismatch_on_a_reading_that_is_perfectly_fine():
    # Regression. describe_mismatch is public, so state_machine can call it on
    # any reading -- including one a legal move explains exactly. It used to
    # answer "closest is e4, but no difference", which is gibberish on a 5 inch
    # screen and reads like an error when nothing is wrong.
    board = chess.Board()
    text = inference.describe_mismatch(board, sensors_would_read(board, "e2e4"))
    assert "no difference" not in text
    assert "e4" in text
    assert len(text) < 80


# --- promises the rest of the system relies on --------------------------------

def test_infer_move_never_modifies_the_board():
    board = board_after(["e4", "d5"])
    fen_before = board.fen()
    stack_before = len(board.move_stack)

    inference.infer_move(board, sensors_would_read(board, "e4d5"))
    inference.infer_move(board, 0)
    inference.describe_mismatch(board, 0)

    assert board.fen() == fen_before
    assert len(board.move_stack) == stack_before


def test_every_opening_move_is_identified_exactly():
    # All twenty first moves, each one asserted unconditionally. Written this
    # way on purpose: an earlier version skipped the comparison whenever the
    # module returned None, so a module that never identified anything would
    # still have passed.
    board = chess.Board()
    for move in list(board.legal_moves):
        observed = sensors_would_read(board, move.uci())
        found, reason = inference.infer_move(board, observed)
        assert found == move, reason
        assert board.is_legal(found)
        assert reason


def test_no_legal_move_can_ever_reproduce_the_current_reading():
    # infer_move short-circuits an unchanged mask to "no move seen yet" WITHOUT
    # checking the legal moves, which is only safe if no legal move can leave
    # the occupied squares exactly as they are. Every move empties its origin
    # and nothing refills it, so the claim should hold -- checked here over a
    # real game rather than left as an argument in a comment.
    board = board_after(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                         "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O"])
    while board.move_stack:
        here = inference._occupancy_from_board(board)
        for move in list(board.legal_moves):
            board.push(move)
            after = inference._occupancy_from_board(board)
            board.pop()
            assert after != here, "%s left the sensors unchanged" % move.uci()
        board.pop()


def test_a_hand_still_holding_the_piece_reads_as_a_finished_capture():
    # Not a bug in this module, a fact about occupancy sensing that the state
    # machine has to defend against: while White holds the e4 pawn on its way
    # to d5, the grid reads e4 empty and d5 occupied -- bit for bit the same as
    # the completed capture. Pinned here so nobody "fixes" it in inference.py;
    # the fix belongs upstream, in requiring a settled reading plus a confirm.
    board = board_after(["e4", "d5"])
    lifted = inference._occupancy_from_board(board) & ~(1 << chess.E4)
    completed = sensors_would_read(board, "e4d5")
    assert lifted == completed


def test_a_mask_that_could_not_come_from_64_squares_is_rejected():
    board = chess.Board()
    with pytest.raises(ValueError):
        inference.infer_move(board, 1 << 64)
    with pytest.raises(ValueError):
        inference.infer_move(board, -1)


def test_a_mask_that_is_not_a_whole_number_is_rejected():
    # Regression. A float mask used to be silently truncated with int(). A
    # float cannot hold 64 bits exactly, so the truncation produced a wholly
    # different board that the module then described with total confidence.
    board = chess.Board()
    real = inference._occupancy_from_board(board)
    with pytest.raises(ValueError):
        inference.infer_move(board, float(real))
    with pytest.raises(ValueError):
        inference.describe_mismatch(board, 12.5)
    with pytest.raises(ValueError):
        inference.infer_move(board, "0xffff")
