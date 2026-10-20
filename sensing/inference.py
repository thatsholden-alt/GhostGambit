"""Work out WHICH chess move the human just made, from a 64-bit occupancy mask.

The sensor grid under the board reports exactly one bit per square: something
magnetic is sitting there, or it is not. It never tells us WHAT is sitting
there. From those 64 bits alone this module has to name a chess move.


HOW WE DO IT, AND WHY IT BEATS THE OBVIOUS WAY
----------------------------------------------
We ask the RULES, not the bitmap. python-chess can list every legal move in the
position. For each one we play it on the board, work out what the sensor grid
WOULD read afterwards, take it back, and compare with what the grid actually
read.

    exactly one match  ->  that is the move
    no match           ->  no legal move explains the board; say what is wrong
    several matches    ->  occupancy genuinely cannot tell them apart; say so

The obvious way -- the way the reference builds do it -- is to XOR the old and
new bitmaps and guess the move from the set of changed squares. It looks
cheaper and it is a bug farm:

  * A CAPTURE changes exactly ONE square. The destination was occupied before
    and is still occupied after, so the diff only shows the origin going empty.
    Every diff-based decoder needs a special case for that.
  * Castling changes four squares at once and en passant changes three, and
    each one needs its own hand-written pattern.
  * https://github.com/ut-ras/chessbot (src/python/realboard.py, function
    move_compare_update) hardcodes castling as a pattern match against the
    exact list of changed square indices. Their decomposer always hands the
    indices back in ascending order, but the black queenside pattern is written
    descending -- so in that build BLACK QUEENSIDE CASTLING CAN NEVER BE
    RECOGNISED. A whole legal move is invisible to their board, and the bug is
    invisible in the code too, because the pattern "looks right".

Matching against the legal move list gets captures, castling, en passant and
promotion for free, with zero special cases, because python-chess already knows
the rules. The cost is a few hundred board pushes per move -- microseconds.


THE ONE THING OCCUPANCY REALLY CANNOT DO
-----------------------------------------
Two different legal moves can leave the identical set of squares occupied. The
common shape is two captures by the same piece: a knight on d4 taking on c6 and
the same knight taking on e6 both leave d4 empty and both targets occupied, so
the two are indistinguishable. When that happens we return None and name the
candidates. We never pick one. A wrong guess desynchronises the wooden board
from the software model, and every move after it is wrong too -- the gantry
would start dragging the wrong pieces and nothing would ever recover.

Promotion is a softer version of the same problem: a queen and a knight on e8
light up the same sensor. See infer_move()'s promotion_default argument.
"""

import chess

import config


# One bit per square, and python-chess numbers squares 0..63 with a1 == 0, so
# bit i of a mask means "square i has something on it".
SQUARE_COUNT = config.BOARD_FILES * config.BOARD_RANKS
FULL_BOARD_MASK = (1 << SQUARE_COUNT) - 1

# Occupancy physically cannot see which piece a pawn turned into, so we have to
# assume one. Queen, because that is what over 95% of promotions are and
# because it is the piece the player will have reached for. The reason string
# always says we assumed, so the UI can ask.
DEFAULT_PROMOTION = chess.QUEEN

# How many square names describe_mismatch() lists before it stops and says
# "+N more". This is a screen-width limit for the 5 inch display, not a chess
# fact; it should move to config.py once ui/ settles on a line length.
MAX_SQUARES_LISTED = 4


def _occupancy_from_board(board):
    """The mask the sensor grid WOULD report for this position.

    This is a private duplicate of sensing/occupancy.py's occupancy_from_board().
    The two were written at the same time by different people; collapse this
    copy and import that one once both modules exist.
    """
    mask = 0
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            mask |= 1 << square
    return mask


def infer_move(board, observed_mask, promotion_default=DEFAULT_PROMOTION):
    """Name the move that turns `board` into `observed_mask`.

    Returns (move, reason) on success and (None, reason) otherwise. The reason
    is always a short human-readable string meant for the screen; it is never
    empty, and on failure it explains what is wrong rather than just saying no.

    Pass promotion_default=None to refuse to assume a promotion piece; the call
    then reports promotions as ambiguous instead of assuming a queen.

    `board` is not modified -- every trial move is taken back.
    """
    observed_mask = _checked_mask(observed_mask)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, "no legal moves here - the game is already over"

    # No legal move can leave the position looking untouched: a quiet move sets
    # a square that was empty, a capture clears the origin, castling moves two
    # pieces. So an unchanged mask means nobody has moved yet, and saying that
    # plainly is friendlier than listing zero differences.
    if observed_mask == _occupancy_from_board(board):
        return None, "board unchanged - no move seen yet"

    candidates = []
    for move in legal_moves:
        board.push(move)
        would_be = _occupancy_from_board(board)
        board.pop()
        if would_be == observed_mask:
            candidates.append(move)

    if not candidates:
        return None, "no legal move fits: " + describe_mismatch(board, observed_mask)

    if len(candidates) == 1:
        move = candidates[0]
        return move, "matched " + board.san(move)

    assumed = _collapse_promotion_choices(candidates, promotion_default)
    if assumed is not None:
        # Name the piece we actually assumed. Saying "queen" while handing back
        # a knight would put a lie on the confirmation screen, and the screen is
        # the only chance anyone gets to catch a wrong promotion.
        return assumed, ("matched " + board.san(assumed)
                         + " - the sensors cannot see which piece was placed,"
                         + " assuming a " + chess.piece_name(promotion_default))

    names = [board.san(move) for move in candidates]
    if len(names) == 2:
        return None, ("ambiguous: " + names[0] + " or " + names[1]
                      + " - both leave the same squares occupied")
    return None, ("ambiguous: " + ", ".join(names)
                  + " - all leave the same squares occupied")


def describe_mismatch(board, observed_mask):
    """A short screen message explaining why the sensors make no sense.

    Names the squares that are unexpectedly empty and unexpectedly occupied,
    e.g. "e4 should be empty; d5 should have a piece". If one legal move
    accounts for more of the reading than doing nothing does, it says which one
    first, and then lists only what is still wrong after that move -- that is
    the common case of a real move plus a knocked-over piece.
    """
    observed_mask = _checked_mask(observed_mask)
    here = _occupancy_from_board(board)
    if observed_mask == here:
        return "board unchanged - no move seen yet"

    move, after_mask, distance = _closest_legal_move(board, observed_mask)
    # This is public, so a caller can hand us a reading that a legal move fits
    # perfectly. Saying "closest is e4, but no difference" would be nonsense on
    # the screen; say plainly that the reading makes sense.
    if move is not None and distance == 0:
        return board.san(move) + " fits this reading exactly - nothing is wrong"
    if move is not None and distance < _bit_count(here ^ observed_mask):
        return ("closest is " + board.san(move) + ", but "
                + _difference_text(after_mask, observed_mask))
    return _difference_text(here, observed_mask)


def _collapse_promotion_choices(candidates, promotion_default):
    """Return the assumed move if the candidates differ only in promotion piece.

    e8=Q, e8=R, e8=B and e8=N light up exactly the same square, so all four
    always match together. That is not a real ambiguity about what the player
    DID with their hand, only about which piece they picked up, so we assume
    and flag it. Returns None if the candidates differ in any other way, which
    is a real ambiguity and must not be assumed away.
    """
    if promotion_default is None:
        return None
    first = candidates[0]
    for move in candidates:
        if (move.promotion is None
                or move.from_square != first.from_square
                or move.to_square != first.to_square):
            return None
    for move in candidates:
        if move.promotion == promotion_default:
            return move
    return None


def _closest_legal_move(board, observed_mask):
    """The one legal move whose result is nearest to what we observed.

    Returns (move, resulting_mask, wrong_bits), or (None, None, None) if there
    are no legal moves or if two moves tie for nearest. A tie is thrown away on
    purpose: this feeds a hint on the screen, and a hint that picks one of two
    equally good explanations at random teaches the player the wrong thing.
    """
    best_move = None
    best_mask = None
    best_distance = None
    tied = False
    # Take the whole list up front. board.legal_moves hands moves out one at a
    # time as it walks the CURRENT position, and the loop below changes that
    # position with push/pop on every pass. Snapshotting first means we are
    # never asking the generator to survive being moved out from under it.
    for move in list(board.legal_moves):
        board.push(move)
        would_be = _occupancy_from_board(board)
        board.pop()
        distance = _bit_count(would_be ^ observed_mask)
        if best_distance is None or distance < best_distance:
            best_move = move
            best_mask = would_be
            best_distance = distance
            tied = False
        elif distance == best_distance:
            tied = True
    if best_move is None or tied:
        return None, None, None
    return best_move, best_mask, best_distance


def _difference_text(expected_mask, observed_mask):
    """Human-readable diff of two masks, from the sensors' point of view."""
    # ~x on a Python int is NOT "flip 64 bits" -- Python integers are unbounded,
    # so ~x is the negative number -(x+1), with an endless run of 1 bits above
    # the top. ANDing it with a mask we already know is 64 bits wide chops it
    # back to the 64 we care about, which is why the order here matters.
    should_be_empty = observed_mask & ~expected_mask
    should_have_piece = expected_mask & ~observed_mask

    parts = []
    if should_be_empty:
        parts.append(_square_list_text(should_be_empty) + " should be empty")
    if should_have_piece:
        parts.append(_square_list_text(should_have_piece) + " should have a piece")
    if not parts:
        return "no difference"
    return "; ".join(parts)


def _square_list_text(mask):
    """"e4, d5" -- capped, because the screen is 5 inches wide."""
    names = [chess.square_name(square) for square in chess.SQUARES
             if mask & (1 << square)]
    extra = len(names) - MAX_SQUARES_LISTED
    if extra > 0:
        names = names[:MAX_SQUARES_LISTED]
        return ", ".join(names) + " +%d more" % extra
    return ", ".join(names)


def _bit_count(mask):
    """How many bits are set. int.bit_count() is Python 3.10+, this is not."""
    return bin(mask).count("1")


def _checked_mask(mask):
    """Reject a mask that could not have come from a 64 square sensor grid.

    A too-wide, negative or non-whole-number mask is a programming mistake
    somewhere upstream, not a sensor reading, and quietly repairing it would
    hide the bug until it showed up as the gantry dragging the wrong piece.

    Floats are rejected rather than truncated. A float cannot even hold 64 bits
    exactly, so int(some_float_mask) comes back as a DIFFERENT board -- close
    enough to look plausible on screen and be believed.
    """
    if not isinstance(mask, int):
        raise ValueError("occupancy mask must be a whole number, got %r" % (mask,))
    if mask < 0 or mask > FULL_BOARD_MASK:
        raise ValueError("occupancy mask must be %d bits, got %r"
                         % (SQUARE_COUNT, mask))
    return mask
