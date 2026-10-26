"""Round state machine: OPEN -> COLLECTING -> AGGREGATING -> DONE | FAILED."""

from __future__ import annotations

from enum import StrEnum


class RoundState(StrEnum):
    OPEN = "OPEN"
    COLLECTING = "COLLECTING"
    AGGREGATING = "AGGREGATING"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS: dict[RoundState, frozenset[RoundState]] = {
    RoundState.OPEN: frozenset({RoundState.COLLECTING, RoundState.FAILED}),
    RoundState.COLLECTING: frozenset({RoundState.AGGREGATING, RoundState.FAILED}),
    RoundState.AGGREGATING: frozenset({RoundState.DONE, RoundState.FAILED}),
    RoundState.DONE: frozenset(),
    RoundState.FAILED: frozenset(),
}


class InvalidTransition(Exception):
    pass


def transition(current: RoundState, target: RoundState) -> RoundState:
    if target not in TRANSITIONS[current]:
        raise InvalidTransition(f"{current} -> {target} is not allowed")
    return target


def is_terminal(state: RoundState) -> bool:
    return not TRANSITIONS[state]
