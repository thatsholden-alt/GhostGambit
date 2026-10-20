"""The game clock: 10 minutes each plus 5 seconds per move (Fischer increment).

Covers the spec's timed-game requirement. Running out of time loses the game.

How it is used by the state machine:

    clock = ChessClock()
    clock.start(chess.WHITE)        # first move of the game
    ...
    clock.end_turn(chess.WHITE)     # white finished a move: +5 s, black's clock starts
    clock.pause()                   # the gantry is dragging a piece: nobody's clock runs
    clock.resume()                  # the gantry parked: the side to move is charged again
    if clock.flag_fallen() is not None:
        ...                         # that side lost on time

Colours are the booleans python-chess uses (chess.WHITE is True, chess.BLACK is
False), so this file does not need to import python-chess at all.

There is no thread and no timer in here. A clock that "ticks" in the background
would need threading, which nobody on the team has used yet and which makes
tests slow and flaky. Instead the clock remembers WHEN the running side started
and works out the time left whenever somebody asks. The state machine asks
every step, which is far more often than a human can see.
"""

import time

import config


class ChessClock(object):

    def __init__(self, base_s=config.DEFAULT_BASE_TIME_S,
                 increment_s=config.DEFAULT_INCREMENT_S, time_fn=None):
        # base_s of None means an untimed casual game: nobody can lose on time.
        self.untimed = base_s is None
        self.increment_s = increment_s
        start_s = 0.0 if self.untimed else float(base_s)
        self._left = {True: start_s, False: start_s}   # seconds left, per colour

        # Injected so tests can move time forward instantly. time.monotonic is
        # used for real play because the wall clock can jump (daylight saving,
        # the Pi syncing its time over Wi-Fi) and would steal or add minutes.
        self._now = time.monotonic if time_fn is None else time_fn

        self.to_move = True        # whose turn it is; white starts
        self._running = False      # is the to_move side being charged right now?
        self._since = 0.0          # when the current charging stretch began
        self._flagged = None       # the colour that ran out of time, once it happens

    # -- reading the clock ---------------------------------------------------

    def remaining(self, colour):
        """Seconds left for one side, never below zero. None in an untimed game."""
        if self.untimed:
            return None
        left = self._left[colour]
        if self._running and colour == self.to_move:
            left -= self._now() - self._since
        return max(0.0, left)

    def flag_fallen(self):
        """The colour that has run out of time (and so lost), or None.

        Only the side whose clock is running can run out, so this checks just
        that one. Once a flag has fallen it stays fallen: the game is over even
        if somebody presses the button afterwards.
        """
        if self.untimed:
            return None
        if self._flagged is None and self._running and self.remaining(self.to_move) <= 0.0:
            self._flagged = self.to_move
            self._stop_charging()
        return self._flagged

    # -- driving the clock ---------------------------------------------------

    def start(self, colour=True):
        """Start the game with this side to move (normally white)."""
        self.to_move = colour
        self._start_charging()

    def end_turn(self, colour):
        """This side finished its move: add the increment and hand over the turn.

        Raises ValueError if it is not that side's turn, because that can only be
        a bug in the state machine and we would rather crash in testing than
        quietly give a player the wrong clock.
        """
        if colour != self.to_move:
            raise ValueError("it is not %s's turn" % ("white" if colour else "black"))
        if self.flag_fallen() is not None:
            # Too late: the time ran out before the move was finished. No increment.
            return
        was_running = self._running
        self._stop_charging()
        if not self.untimed:
            self._left[colour] += self.increment_s
        self.to_move = not colour
        # If the clock was paused (the robot was still moving), the next side
        # starts when resume() is called, not now.
        if was_running:
            self._start_charging()

    def pause(self):
        """Stop charging anyone, e.g. while the gantry drags a piece.

        config.CHARGE_MOTION_TIME_TO_PLAYER is False: a capture takes the robot
        7-10 s, and a player must never lose on time because the machine is slow.
        The state machine decides when to call this; the clock just obeys.
        """
        self._stop_charging()

    def resume(self):
        """Start charging the side to move again (after the gantry has parked)."""
        if self._flagged is None:
            self._start_charging()

    # -- internals -------------------------------------------------------------

    def _start_charging(self):
        if not self._running:
            self._running = True
            self._since = self._now()

    def _stop_charging(self):
        # Bank the time used so far, so pausing and resuming never loses or
        # gains a fraction of a second.
        if self._running:
            self._left[self.to_move] -= self._now() - self._since
            self._running = False


def format_time(seconds):
    """Text for the display: "9:58", or "0:07.4" with tenths under 10 seconds.

    Tenths only near the end because that is when a player is watching every
    blink, and showing them all game long just makes the screen flicker.
    """
    if seconds is None:
        return "--:--"
    if seconds < 10.0:
        # Round DOWN so the display never shows time the player does not have.
        tenths = int(seconds * 10)
        return "0:%02d.%d" % (tenths // 10, tenths % 10)
    whole = int(seconds)
    return "%d:%02d" % (whole // 60, whole % 60)
