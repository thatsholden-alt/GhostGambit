"""Turn a legal chess move into the list of PHYSICAL actions the gantry owes it.

Everything else in the project asks this module one question: given this move,
which pieces have to physically move, and from where to where? The chess engine
and python-chess both think a move is one thing. The board disagrees -- a
capture is two drags, a castle is two drags, and en passant removes a piece from
a square nobody named in the move.

This module makes no motion decisions. It does not know where the capture area
is, how fast to drag, or which lane to take; motion/planner.py owns all of that.
This file only reports what happened, in board coordinates.

Nothing here is physical, so nothing here imports config.py. If you find
yourself wanting a millimetre in this file, you are writing planner code in the
wrong module.
"""

import chess


# --- The four cases that are not "pick it up and put it down" ----------------
#
# 1. CAPTURE          the victim is standing on move.to_square. Drag it away
#                     FIRST, then make the move, or the two pieces collide.
# 2. EN PASSANT       the victim is NOT on move.to_square. It is on the square
#                     with the destination's FILE and the origin's RANK. Every
#                     build that assumes to_square drags the wrong piece, or
#                     nothing at all, and leaves a ghost pawn on the board.
# 3. CASTLING         two of our own pieces move. python-chess only tells us
#                     about the king; the rook squares are ours to work out.
# 4. PROMOTION        the board cannot manufacture a queen out of nothing. We
#                     only report that it happened. Whoever is driving decides
#                     whether to fetch a queen from the capture area or ask the
#                     human to swap the piece by hand.


# Files on the back rank, as python-chess file indices (0 = a .. 7 = h). These
# are rules of chess, not measurements of our board, so they belong here rather
# than in config.py -- no calibration will ever move the a-file.
_FILE_A = 0
_FILE_C = 2
_FILE_D = 3
_FILE_F = 5
_FILE_G = 6
_FILE_H = 7

KINDS = (
    "simple",
    "capture",
    "en_passant",
    "castle",
    "promotion",
    "capture_promotion",
)


def classify_move(board, move):
    """Describe a legal move as the physical work it implies.

    Returns a dict:
        "kind"              one of KINDS
        "captured_square"   square index of the piece that must be PHYSICALLY
                            REMOVED before the move runs, or None
        "rook_from"         square index, castling only, else None
        "rook_to"           square index, castling only, else None
        "promotion_piece"   lowercase letter such as "q", or None

    Two extra keys ride along for convenience; the five above are the contract
    and are always present:
        "king_from" / "king_to"   castling only, else None. See the note in
                                  _castling_squares for why the king's landing
                                  square is worth stating separately.

    Raises ValueError if the move is not legal in this position. That is
    deliberate: an illegal move arriving here means something upstream is
    broken, and quietly planning motion for it would drag a real piece to a real
    wrong place.
    """
    if move is None or move == chess.Move.null():
        raise ValueError("classify_move needs a real move, got %r" % (move,))
    if not board.is_legal(move):
        raise ValueError("%s is not legal in %s" % (move.uci(), board.fen()))

    result = {
        "kind": "simple",
        "captured_square": None,
        "rook_from": None,
        "rook_to": None,
        "promotion_piece": None,
        "king_from": None,
        "king_to": None,
    }

    if board.is_castling(move):
        king_from, king_to, rook_from, rook_to = _castling_squares(board, move)
        result["kind"] = "castle"
        result["king_from"] = king_from
        result["king_to"] = king_to
        result["rook_from"] = rook_from
        result["rook_to"] = rook_to
        return result

    if board.is_en_passant(move):
        result["kind"] = "en_passant"
        result["captured_square"] = en_passant_victim_square(move)
        return result

    # board.is_capture() is true for en passant as well, which is exactly why
    # the en passant case is checked first and returns above.
    captured = board.is_capture(move)
    promoted = move.promotion is not None

    if captured:
        result["captured_square"] = move.to_square

    if promoted:
        # piece_symbol turns the python-chess piece-type integer (chess.QUEEN is
        # just the number 5) into the single lowercase letter the rest of the
        # project passes around.
        result["promotion_piece"] = chess.piece_symbol(move.promotion)
        result["kind"] = "capture_promotion" if captured else "promotion"
    elif captured:
        result["kind"] = "capture"

    return result


def en_passant_victim_square(move):
    """Square of the pawn removed by an en passant capture.

    The capturing pawn lands on an EMPTY square, one rank beyond the pawn it
    takes, so the victim sits at the destination's FILE and the origin's RANK.
    White plays e5xf6 and the black pawn standing on f5 is the one that comes
    off the board. Never assume the victim is on move.to_square.

    Only meaningful when board.is_en_passant(move) is true; classify_move checks
    that before calling this.
    """
    return chess.square(
        chess.square_file(move.to_square),
        chess.square_rank(move.from_square),
    )


def _castling_squares(board, move):
    """Return (king_from, king_to, rook_from, rook_to) for a castling move.

    Two things make this trickier than it looks.

    First, python-chess sometimes spells castling as KING TAKES OWN ROOK (e1h1)
    rather than as the king's two-square hop (e1g1). board.push() accepts both,
    and both turn up in code that has been anywhere near the Chess960 helpers,
    so move.to_square is NOT reliably where the king lands. We ask which side we
    are castling to and then state all four squares ourselves, which comes out
    right for either spelling.

    Second, the rook is not always on the a or h file in Chess960. We do not
    play Chess960, but reading the rook's square off the board's castling rights
    costs nothing and gives the same answer for standard chess.
    """
    rank = chess.square_rank(move.from_square)
    king_from = move.from_square

    if board.is_kingside_castling(move):
        return (
            king_from,
            chess.square(_FILE_G, rank),
            _rook_square_for_castling(board, king_from, True),
            chess.square(_FILE_F, rank),
        )

    return (
        king_from,
        chess.square(_FILE_C, rank),
        _rook_square_for_castling(board, king_from, False),
        chess.square(_FILE_D, rank),
    )


def _rook_square_for_castling(board, king_square, kingside):
    """Find the rook that is castling, on the castling king's own back rank."""
    # board.castling_rights is a BITBOARD of the rook squares that may still
    # castle: a 64-bit integer with one bit per square, bit 0 = a1. The & keeps
    # only the bits on the rank we care about, and scan_forward walks the bits
    # that are left, handing back square numbers.
    rank = chess.square_rank(king_square)
    candidates = board.castling_rights & chess.BB_RANKS[rank]

    # "Kingside" means the rook is on the far side of the KING, not the far side
    # of the e-file. Those are the same thing in standard chess but not in
    # Chess960, where a king starting on g1 has both of its rooks past the
    # e-file -- comparing against e would then find no rook at all and fall
    # through to the guess below, handing the planner an empty square to drag.
    king_file = chess.square_file(king_square)
    for square in chess.scan_forward(candidates):
        beyond_king = chess.square_file(square) > king_file
        if beyond_king == kingside:
            return square

    # Standard-chess fallback, reached only if the castling-rights bitboard has
    # been cleared out from under us. That would be a python-chess surprise
    # rather than a normal game state, but returning the ordinary rook square
    # beats returning None into the motion planner.
    return chess.square(_FILE_H if kingside else _FILE_A, rank)


def is_game_over(board):
    """Return (over, reason). reason is short enough for a small screen.

    Only AUTOMATIC endings count here -- the ones where the game is over whether
    or not anybody says so. Threefold repetition and the fifty-move rule are
    CLAIMS: play continues until a player asks for the draw. Those live in
    draw_claim_available() so the UI can offer a button instead of ending the
    game behind the players' backs.
    """
    if board.is_checkmate():
        # The side to move is the side that got mated, so the winner is whoever
        # is not on the move.
        winner = "Black" if board.turn == chess.WHITE else "White"
        return True, "Checkmate - %s wins" % winner

    if board.is_stalemate():
        return True, "Stalemate - draw"

    if board.is_insufficient_material():
        return True, "Draw - not enough material"

    # 150 half-moves with no capture and no pawn move. Unlike the fifty-move
    # rule, this one needs no claim from anybody.
    if board.is_seventyfive_moves():
        return True, "Draw - 75-move rule"

    if board.is_fivefold_repetition():
        return True, "Draw - fivefold repetition"

    return False, ""


def draw_claim_available(board):
    """Return (claimable, reason) for the two draws a player has to ask for.

    Kept out of is_game_over so that a repetition in a friendly game does not
    stop the board mid-game without anyone agreeing to it.
    """
    if board.can_claim_fifty_moves():
        return True, "Draw claim - fifty-move rule"
    if board.can_claim_threefold_repetition():
        return True, "Draw claim - threefold repetition"
    return False, ""


def describe(board, move):
    """One short line about the physical work, for the screen and for logs.

    Call this BEFORE the move is pushed: it reads the pieces that are still
    standing on the board.
    """
    info = classify_move(board, move)
    mover = board.piece_at(move.from_square)
    mover_name = chess.piece_name(mover.piece_type) if mover else "piece"
    origin = chess.square_name(move.from_square)

    if info["kind"] == "castle":
        side = "kingside"
        if chess.square_file(info["king_to"]) == _FILE_C:
            side = "queenside"
        return "castle %s: king %s-%s, rook %s-%s" % (
            side,
            origin,
            chess.square_name(info["king_to"]),
            chess.square_name(info["rook_from"]),
            chess.square_name(info["rook_to"]),
        )

    text = "%s %s-%s" % (mover_name, origin, chess.square_name(move.to_square))

    if info["captured_square"] is not None:
        victim = board.piece_at(info["captured_square"])
        victim_name = chess.piece_name(victim.piece_type) if victim else "piece"
        text += ", remove %s on %s" % (
            victim_name,
            chess.square_name(info["captured_square"]),
        )
        if info["kind"] == "en_passant":
            text += " (en passant)"

    if info["promotion_piece"] is not None:
        text += ", promotes to %s" % info["promotion_piece"]

    return text
