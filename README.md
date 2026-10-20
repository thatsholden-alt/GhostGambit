# GhostGambit
Automatic Chess Board, for High School Project FHS

A wooden chess board that moves its own pieces. An electromagnet on a CoreXY
gantry under the board slides under a piece and drags it along the lines
between squares. A Raspberry Pi runs the chess rules (python-chess) and the
computer opponent (Fairy-Stockfish), and an Arduino running GRBL drives the
motors.

## Status

The board's Python software runs on a laptop with fake hardware, so it can be
built and tested before the gantry exists. The screen (`ui/`) and the simulator
(`sim/`) are not written yet.

## Run the checks

```
pip install python-chess pytest
python -m pytest tests
```

## What's where

| Path | Job |
|---|---|
| `state_machine.py` | Runs a whole game, one step at a time |
| `config.py` | Every physical number in one place |
| `game/` | Chess rules for the physical board, difficulty levels, game clock |
| `motion/` | CoreXY math, routes between squares, the GRBL motor driver |
| `sensing/` | Reading the square sensors and working out the human's move |
| `tests/` | Automated checks |

## How this code was made

The first code commit is a starting point drafted with help from Claude, an AI
assistant, in September and October 2026. Every later commit shows the team's
own changes.
