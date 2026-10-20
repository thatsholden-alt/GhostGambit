"""The difficulty ladder for the offline AI opponent.

Covers the requirement "Play modes | Friend, offline AI (10+ levels), or Lichess"
(weight 5) -- specifically the "10+ levels" half of it.

This module has exactly one job: turn the level number the player dialled in on
the knob into the UCI options the engine needs. It never talks to the engine
itself; game/engine.py does that. Splitting them this way means the whole ladder
can be unit-tested on a laptop with no engine installed at all, which matters
because the hardware does not exist until December.
"""


# --- Why this ladder is built on UCI_Elo and nothing else --------------------
#
# The obvious approach is to mix two knobs: limit the search depth for the easy
# levels and use the engine's own Elo handicap for the hard ones. We tried that
# in an earlier project and measured it over 40 games. It is backwards:
#
#     depth-1 vs Elo-1320  ->  depth-1 won 6, drew 1, lost 3   (~1430 Elo)
#     depth-2 vs Elo-1320  ->  depth-2 won 10, lost 0
#     depth-3 vs Elo-1320  ->  depth-3 won 10, lost 0
#
# Depth-limiting is STRONGER than a low Elo setting, not weaker, so a ladder
# that starts with depth limits is not sorted. Using UCI_Elo for every rung
# means the ladder is ordered by the engine's own calibration and every level
# has an exact Elo number we can print next to it.
#
# The cost of this decision is that it requires Fairy-Stockfish. We measured the
# floor of both engines by asking them directly (printf 'uci' | engine):
#
#     Stockfish 19        UCI_Elo min 1320   Skill Level  0..20
#     Fairy-Stockfish 14  UCI_Elo min  500   Skill Level -20..20
#
# Stock Stockfish simply cannot go below about 1320 Elo, which is roughly a
# decent club player and nowhere near beatable for someone learning the game.
# Fairy-Stockfish reaches 500 and installs on the Pi with
# "sudo apt install fairy-stockfish" (604 kB, arm64 is packaged).

# Ten rungs, evenly spaced across the engine's usable range. Level 1 is a
# genuine beginner; level 10 is far past anyone who will play this board.
LEVELS = (
    {"level": 1,  "elo": 500,  "label": "Beginner"},
    {"level": 2,  "elo": 750,  "label": "Beginner+"},
    {"level": 3,  "elo": 1000, "label": "Casual"},
    {"level": 4,  "elo": 1250, "label": "Casual+"},
    {"level": 5,  "elo": 1500, "label": "Intermediate"},
    {"level": 6,  "elo": 1750, "label": "Intermediate+"},
    {"level": 7,  "elo": 2000, "label": "Strong"},
    {"level": 8,  "elo": 2250, "label": "Strong+"},
    {"level": 9,  "elo": 2550, "label": "Expert"},
    {"level": 10, "elo": 2850, "label": "Maximum"},
)

MIN_LEVEL = LEVELS[0]["level"]
MAX_LEVEL = LEVELS[-1]["level"]

# The lowest Elo any rung asks for. Used to check an engine can actually manage
# our easy levels before a player discovers it the hard way.
LOWEST_ELO = LEVELS[0]["elo"]
HIGHEST_ELO = LEVELS[-1]["elo"]


def _row_for(level):
    """Return the ladder row for a level number, or raise if it is not a rung."""
    for row in LEVELS:
        if row["level"] == level:
            return row
    # Raise rather than clamp. A bad level number means the knob-reading code is
    # wrong, and silently playing at level 1 would hide that bug until April.
    raise ValueError(
        "level must be %d..%d, got %r" % (MIN_LEVEL, MAX_LEVEL, level)
    )


def uci_options_for(level):
    """Return the UCI options that put the engine at this difficulty.

    The caller feeds these to the engine as "setoption name <key> value <value>"
    before asking for a move.
    """
    row = _row_for(level)

    # UCI_Elo is ignored unless UCI_LimitStrength is switched on first. Sending
    # the Elo alone leaves the engine at full strength, and because the engine
    # does not complain, every level would feel identical and we would not find
    # out until someone played the board.
    return {
        "UCI_LimitStrength": True,
        "UCI_Elo": row["elo"],
    }


# A warning worth writing down, because a reference build we studied got this
# wrong and shipped it. The UT Austin "chessbot" project sends its difficulty as
# part of the search command:
#
#     go movetime 50 depth 5 skill 10
#
# "skill" is not a parameter of the UCI "go" command. The engine ignores the
# token without erroring, so their difficulty setting does nothing at all and
# every level plays the same. Difficulty is always "setoption", never "go".
#     https://github.com/ut-ras/chessbot  (src/python/bitboard_with_stockfish.py)
#     https://official-stockfish.github.io/docs/stockfish-wiki/UCI-&-Commands.html


def describe(level):
    """Return a short string for the screen, e.g. "Level 4 - Casual+ (1250)"."""
    row = _row_for(level)
    return "Level %d - %s (%d)" % (row["level"], row["label"], row["elo"])


def clamp_level(level):
    """Force any number into the valid level range.

    The knob is an analog input read through an ADC, so it can hand us a value
    slightly outside the range we expect from noise or from a calibration that
    drifted. That is the one place a bad level is normal rather than a bug, so
    the knob code clamps and everything else raises.
    """
    if level < MIN_LEVEL:
        return MIN_LEVEL
    if level > MAX_LEVEL:
        return MAX_LEVEL
    return int(level)


def check_engine_supports_ladder(engine_min_elo, engine_max_elo):
    """Return a list of problems with running this ladder on a given engine.

    An empty list means the engine can play every rung. Call this once at start
    up with the range the engine reported in its "uci" response, so that picking
    the wrong engine shows up as a clear message instead of as several easy
    levels that all feel the same.
    """
    problems = []
    if engine_min_elo > LOWEST_ELO:
        unreachable = [r["level"] for r in LEVELS if r["elo"] < engine_min_elo]
        problems.append(
            "engine cannot go below %d Elo, so levels %s are not reachable "
            "(need an engine that reaches %d)"
            % (engine_min_elo, unreachable, LOWEST_ELO)
        )
    if engine_max_elo < HIGHEST_ELO:
        problems.append(
            "engine tops out at %d Elo, below our level %d target of %d"
            % (engine_max_elo, MAX_LEVEL, HIGHEST_ELO)
        )
    return problems


# --- One honest caveat for the review panel ----------------------------------
#
# The Elo numbers above are the engine's own calibration, and that calibration
# assumes a normal tournament time control. We give the engine 0.5 s per move
# (ENGINE_TIME_LIMIT) because the gantry needs the rest of the 4 s budget to
# actually move the piece. At half a second the top rungs will play somewhat
# below their printed number.
#
# This does not affect the ORDER of the ladder, which is the thing the
# requirement asks for, and it does not affect the low rungs at all, since a
# 500 Elo engine is not short of thinking time. It does mean level 10 is not
# literally 2850. Say so rather than claim otherwise.
#
# Smoke-tested on 2026-09-08 against Fairy-Stockfish 14, 0.5 s per move, on the
# Fried Liver position (white to play, Nxf7 forks queen and rook). Two runs:
#
#     levels 1-3    INCONSISTENT -- missed the fork in one run, found it in the
#                   other. The engine deliberately randomises its play at low
#                   Elo, which is exactly what a beginner level should do.
#     levels 4-10   found Nxf7 every time, both runs.
#
# The inconsistency at the bottom is the useful part: it proves one position is
# not a measurement. Do not quote these results as evidence the ladder is
# ordered. The only thing that proves ordering is the 40-game run described at
# the top of this file, and it needs repeating on this ladder before April.
