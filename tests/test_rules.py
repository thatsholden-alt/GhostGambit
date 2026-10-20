"""Tests for game/rules.py -- move classification into physical actions.

No hardware, no engine, no sleeping. Positions are built from FEN strings so
that every case is exactly the one being tested and nothing else.
    python -m pytest tests/test_rules.py -q
"""

import chess
import pytest

from game import rules


# --- Positions used by more than one test -----------------------------------

# Kings and rooks on their home squares, everything else swept away, so all four
# castlings are legal in one position.
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

# White has just met ...f7-f5 with a pawn on e5, so e5xf6 is en passant.
WHITE_EP_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"

# Mirror image: White has just played d2-d4 past a black pawn on e4.
BLACK_EP_FEN = "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3"

CONTRACT_KEYS = (
    "kind",
    "captured_square",
    "rook_from",
    "rook_to",
    "promotion_piece",
)


def sq(name):
    """a1 -> 0, h8 -> 63. Never do this arithmetic by hand."""
    return chess.parse_square(name)


# --- The ordinary case -------------------------------------------------------


def test_quiet_move_is_simple_and_asks_for_no_extra_work():
    board = chess.Board()
    info = rules.classify_move(board, chess.Move.from_uci("e2e4"))
    assert info["kind"] == "simple"
    assert info["captured_square"] is None
    assert info["rook_from"] is None
    assert info["rook_to"] is None
    assert info["promotion_piece"] is None


def test_every_classification_carries_the_whole_contract():
    # Other modules index this dict without checking. A missing key would show
    # up as a KeyError in the middle of a move, with a piece halfway across the
    # board, which is the worst possible time to find out.
    cases = [
        (chess.Board(), "e2e4"),
        (chess.Board(CASTLING_FEN), "e1g1"),
        (chess.Board(WHITE_EP_FEN), "e5f6"),
        (chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1"), "a7a8q"),
        (chess.Board("1n5k/P7/8/8/8/8/7K/8 w - - 0 1"), "a7b8q"),
        (chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
         "e4d5"),
    ]
    for board, uci in cases:
        info = rules.classify_move(board, chess.Move.from_uci(uci))
        for key in CONTRACT_KEYS:
            assert key in info, "%s is missing %r" % (uci, key)
        assert info["kind"] in rules.KINDS


# --- Captures ----------------------------------------------------------------


def test_ordinary_capture_removes_the_piece_on_the_destination():
    board = chess.Board(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    )
    move = chess.Move.from_uci("e4d5")
    info = rules.classify_move(board, move)
    assert info["kind"] == "capture"
    assert info["captured_square"] == move.to_square
    assert info["captured_square"] == sq("d5")
    assert board.piece_at(info["captured_square"]).piece_type == chess.PAWN


# --- En passant: the assertion this whole file exists for --------------------


def test_white_en_passant_victim_is_not_the_destination_square():
    board = chess.Board(WHITE_EP_FEN)
    move = chess.Move.from_uci("e5f6")
    assert board.is_en_passant(move)

    info = rules.classify_move(board, move)

    assert info["kind"] == "en_passant"
    # THE test. Drag the piece on f6 and you drag empty air; the pawn that has
    # to come off the board is standing on f5.
    assert info["captured_square"] != move.to_square
    assert info["captured_square"] == sq("f5")
    assert board.piece_at(move.to_square) is None
    victim = board.piece_at(info["captured_square"])
    assert victim.piece_type == chess.PAWN
    assert victim.color == chess.BLACK


def test_black_en_passant_victim_is_not_the_destination_square():
    board = chess.Board(BLACK_EP_FEN)
    move = chess.Move.from_uci("e4d3")
    assert board.is_en_passant(move)

    info = rules.classify_move(board, move)

    assert info["kind"] == "en_passant"
    assert info["captured_square"] != move.to_square
    assert info["captured_square"] == sq("d4")
    assert board.piece_at(move.to_square) is None
    victim = board.piece_at(info["captured_square"])
    assert victim.piece_type == chess.PAWN
    assert victim.color == chess.WHITE


def test_en_passant_victim_shares_the_destination_file_and_the_origin_rank():
    # Same rule stated the other way round, so a change that swaps file and rank
    # cannot pass by accident.
    for fen, uci in ((WHITE_EP_FEN, "e5f6"), (BLACK_EP_FEN, "e4d3")):
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        victim = rules.classify_move(board, move)["captured_square"]
        assert chess.square_file(victim) == chess.square_file(move.to_square)
        assert chess.square_rank(victim) == chess.square_rank(move.from_square)


def test_en_passant_is_never_reported_as_a_plain_capture():
    # python-chess says is_capture() is true for en passant. A classifier that
    # checks captures first would call this "capture" and remove the wrong pawn.
    board = chess.Board(WHITE_EP_FEN)
    assert board.is_capture(chess.Move.from_uci("e5f6"))
    assert rules.classify_move(board, chess.Move.from_uci("e5f6"))["kind"] == "en_passant"


# --- Castling: all four --------------------------------------------------


def test_white_kingside_castling_rook_goes_h1_to_f1():
    board = chess.Board(CASTLING_FEN)
    info = rules.classify_move(board, chess.Move.from_uci("e1g1"))
    assert info["kind"] == "castle"
    assert info["rook_from"] == sq("h1")
    assert info["rook_to"] == sq("f1")
    assert info["captured_square"] is None


def test_white_queenside_castling_rook_goes_a1_to_d1():
    board = chess.Board(CASTLING_FEN)
    info = rules.classify_move(board, chess.Move.from_uci("e1c1"))
    assert info["kind"] == "castle"
    assert info["rook_from"] == sq("a1")
    assert info["rook_to"] == sq("d1")


def test_black_kingside_castling_rook_goes_h8_to_f8():
    board = chess.Board(CASTLING_FEN.replace(" w ", " b "))
    info = rules.classify_move(board, chess.Move.from_uci("e8g8"))
    assert info["kind"] == "castle"
    assert info["rook_from"] == sq("h8")
    assert info["rook_to"] == sq("f8")


def test_black_queenside_castling_rook_goes_a8_to_d8():
    board = chess.Board(CASTLING_FEN.replace(" w ", " b "))
    info = rules.classify_move(board, chess.Move.from_uci("e8c8"))
    assert info["kind"] == "castle"
    assert info["rook_from"] == sq("a8")
    assert info["rook_to"] == sq("d8")


def test_castling_rook_always_ends_between_king_and_corner():
    # The rook lands on the square the king passed over, for all four cases.
    for fen, uci in (
        (CASTLING_FEN, "e1g1"),
        (CASTLING_FEN, "e1c1"),
        (CASTLING_FEN.replace(" w ", " b "), "e8g8"),
        (CASTLING_FEN.replace(" w ", " b "), "e8c8"),
    ):
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        info = rules.classify_move(board, move)
        king_from = chess.square_file(move.from_square)
        king_to = chess.square_file(info["king_to"])
        rook_to = chess.square_file(info["rook_to"])
        assert min(king_from, king_to) < rook_to < max(king_from, king_to)
        assert chess.square_rank(info["rook_to"]) == chess.square_rank(move.from_square)
        assert board.piece_at(info["rook_from"]).piece_type == chess.ROOK


def test_castling_spelled_as_king_takes_own_rook_gives_the_same_answer():
    # python-chess accepts e1h1 as "castle kingside" as well as e1g1, and
    # move.to_square is then the ROOK's square, not the king's. Anything that
    # trusts to_square blindly puts the king on h1.
    board = chess.Board(CASTLING_FEN)
    hop = rules.classify_move(board, chess.Move.from_uci("e1g1"))
    takes_rook = rules.classify_move(board, chess.Move.from_uci("e1h1"))
    for key in ("kind", "rook_from", "rook_to", "king_to"):
        assert hop[key] == takes_rook[key]
    assert takes_rook["king_to"] == sq("g1")


# --- Promotion ---------------------------------------------------------------


def test_quiet_promotion_reports_the_new_piece_and_no_capture():
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    info = rules.classify_move(board, chess.Move.from_uci("a7a8q"))
    assert info["kind"] == "promotion"
    assert info["promotion_piece"] == "q"
    assert info["captured_square"] is None


def test_underpromotion_reports_the_piece_actually_asked_for():
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    for uci, letter in (("a7a8n", "n"), ("a7a8b", "b"), ("a7a8r", "r")):
        info = rules.classify_move(board, chess.Move.from_uci(uci))
        assert info["kind"] == "promotion"
        assert info["promotion_piece"] == letter


def test_capture_promotion_is_its_own_kind_and_still_names_the_victim():
    board = chess.Board("1n5k/P7/8/8/8/8/7K/8 w - - 0 1")
    move = chess.Move.from_uci("a7b8q")
    info = rules.classify_move(board, move)
    assert info["kind"] == "capture_promotion"
    assert info["promotion_piece"] == "q"
    assert info["captured_square"] == move.to_square == sq("b8")


def test_black_capture_promotion():
    board = chess.Board("7k/8/8/8/8/8/1p5K/N7 b - - 0 1")
    info = rules.classify_move(board, chess.Move.from_uci("b2a1q"))
    assert info["kind"] == "capture_promotion"
    assert info["captured_square"] == sq("a1")
    assert info["promotion_piece"] == "q"


def test_promotion_piece_is_a_single_lowercase_letter():
    # The contract says "q", not "Q" and not chess.QUEEN.
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    letter = rules.classify_move(board, chess.Move.from_uci("a7a8q"))["promotion_piece"]
    assert isinstance(letter, str)
    assert len(letter) == 1
    assert letter.islower()


# --- Bad input ---------------------------------------------------------------


def test_an_illegal_move_raises_instead_of_being_planned():
    board = chess.Board()
    with pytest.raises(ValueError):
        rules.classify_move(board, chess.Move.from_uci("e2e5"))


def test_a_move_by_the_wrong_side_raises():
    board = chess.Board()
    with pytest.raises(ValueError):
        rules.classify_move(board, chess.Move.from_uci("e7e5"))


def test_none_and_the_null_move_raise():
    board = chess.Board()
    with pytest.raises(ValueError):
        rules.classify_move(board, None)
    with pytest.raises(ValueError):
        rules.classify_move(board, chess.Move.null())


# --- The sweep: does the classification actually reproduce the position? -----


def apply_classification(board, move, info):
    """Replay the classification as if the gantry had carried it out.

    Returns a plain {square: piece} map. This is the honest test of the module:
    if following its instructions does not land the pieces where python-chess
    says they belong, the module is wrong, whatever the labels say.
    """
    pieces = dict(board.piece_map())

    # Removal first, exactly as the real board does it -- the capture area trip
    # happens before the capturing piece moves.
    if info["captured_square"] is not None:
        assert info["captured_square"] in pieces, "told to remove an empty square"
        del pieces[info["captured_square"]]

    mover = pieces.pop(move.from_square)
    if info["promotion_piece"] is not None:
        mover = chess.Piece(
            chess.PIECE_SYMBOLS.index(info["promotion_piece"]), mover.color
        )

    if info["kind"] == "castle":
        rook = pieces.pop(info["rook_from"])
        pieces[info["rook_to"]] = rook
        pieces[info["king_to"]] = mover
    else:
        pieces[move.to_square] = mover

    return pieces


SWEEP_FENS = (
    chess.STARTING_FEN,
    CASTLING_FEN,
    CASTLING_FEN.replace(" w ", " b "),
    WHITE_EP_FEN,
    BLACK_EP_FEN,
    "1n5k/P7/8/8/8/8/7K/8 w - - 0 1",
    # A crowded middlegame with captures, checks and castling all available.
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    # Both sides have promotions and capture-promotions on the board.
    "1r5k/P1P4p/8/8/8/8/1p1p3K/1R6 w - - 0 1",
)


def test_following_the_classification_reproduces_the_real_position():
    for fen in SWEEP_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            info = rules.classify_move(board, move)
            replayed = apply_classification(board, move, info)

            board.push(move)
            expected = board.piece_map()
            board.pop()

            assert replayed == expected, "%s in %s" % (move.uci(), fen)


def test_the_sweep_positions_really_do_contain_every_special_case():
    # A sweep that never sees an en passant proves nothing about en passant.
    seen = set()
    for fen in SWEEP_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            seen.add(rules.classify_move(board, move)["kind"])
    assert seen == set(rules.KINDS)


def test_only_captures_and_en_passant_name_a_square_to_clear():
    for fen in SWEEP_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            info = rules.classify_move(board, move)
            if info["kind"] in ("simple", "castle", "promotion"):
                assert info["captured_square"] is None
            else:
                assert info["captured_square"] is not None
                assert board.piece_at(info["captured_square"]) is not None
                # Never drag one of our own pieces to the capture area.
                assert board.piece_at(info["captured_square"]).color != board.turn


def test_rook_squares_are_set_for_castling_and_nothing_else():
    for fen in SWEEP_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            info = rules.classify_move(board, move)
            if info["kind"] == "castle":
                assert info["rook_from"] is not None
                assert info["rook_to"] is not None
            else:
                assert info["rook_from"] is None
                assert info["rook_to"] is None


# --- Game over ---------------------------------------------------------------


def test_a_game_in_progress_is_not_over():
    over, reason = rules.is_game_over(chess.Board())
    assert over is False
    assert reason == ""


def test_checkmate_names_the_winner():
    # Fool's mate: 1. f3 e5 2. g4 Qh4#.
    board = chess.Board(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )
    assert board.is_checkmate()
    over, reason = rules.is_game_over(board)
    assert over is True
    assert "Checkmate" in reason
    assert "Black" in reason


def test_checkmate_the_other_way_round():
    # Back-rank mate, Black to move and mated.
    board = chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
    assert board.is_checkmate()
    over, reason = rules.is_game_over(board)
    assert over is True
    assert "White" in reason


def test_stalemate_is_a_draw_and_not_a_win():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    over, reason = rules.is_game_over(board)
    assert over is True
    assert "Stalemate" in reason
    assert "wins" not in reason


def test_bare_kings_are_insufficient_material():
    over, reason = rules.is_game_over(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1"))
    assert over is True
    assert "material" in reason


def test_seventy_five_move_rule_ends_the_game_without_a_claim():
    board = chess.Board("8/8/8/4k3/8/8/4K3/R7 w - - 150 200")
    assert board.is_seventyfive_moves()
    over, reason = rules.is_game_over(board)
    assert over is True
    assert "75" in reason


def test_fivefold_repetition_ends_the_game_without_a_claim():
    board = chess.Board()
    # Knights out and back, four times over: the starting position occurs five
    # times in total.
    for _ in range(4):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push(chess.Move.from_uci(uci))
    assert board.is_fivefold_repetition()
    over, reason = rules.is_game_over(board)
    assert over is True
    assert "repetition" in reason


def test_threefold_repetition_is_only_a_claim_not_an_ending():
    board = chess.Board()
    for _ in range(2):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push(chess.Move.from_uci(uci))
    assert board.can_claim_threefold_repetition()
    # The game continues until somebody asks for the draw.
    assert rules.is_game_over(board) == (False, "")
    claimable, reason = rules.draw_claim_available(board)
    assert claimable is True
    assert "threefold" in reason


def test_fifty_move_rule_is_only_a_claim():
    board = chess.Board("8/8/8/4k3/8/8/4K3/R7 w - - 100 120")
    assert rules.is_game_over(board) == (False, "")
    claimable, reason = rules.draw_claim_available(board)
    assert claimable is True
    assert "fifty" in reason


def test_reasons_fit_a_small_screen():
    boards = (
        chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
        chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1"),
        chess.Board("8/8/8/4k3/8/8/4K3/R7 w - - 150 200"),
    )
    for board in boards:
        over, reason = rules.is_game_over(board)
        assert over is True
        assert 0 < len(reason) <= 30, reason


# --- The human-readable line -------------------------------------------------


def test_describe_spells_out_the_en_passant_victim():
    board = chess.Board(WHITE_EP_FEN)
    text = rules.describe(board, chess.Move.from_uci("e5f6"))
    assert "f5" in text
    assert "en passant" in text


def test_describe_mentions_both_pieces_that_move_when_castling():
    board = chess.Board(CASTLING_FEN)
    text = rules.describe(board, chess.Move.from_uci("e1c1"))
    assert "king" in text
    assert "rook" in text
    assert "a1" in text and "d1" in text


def test_describe_never_blows_up_on_a_legal_move():
    for fen in SWEEP_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            assert rules.describe(board, move)
