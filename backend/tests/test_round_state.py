import pytest

from cfl.coordinator.state import InvalidTransition, RoundState, is_terminal, transition


def test_happy_path():
    s = RoundState.OPEN
    for nxt in (RoundState.COLLECTING, RoundState.AGGREGATING, RoundState.DONE):
        s = transition(s, nxt)
    assert is_terminal(s)


def test_every_non_terminal_state_can_fail():
    for s in (RoundState.OPEN, RoundState.COLLECTING, RoundState.AGGREGATING):
        assert transition(s, RoundState.FAILED) is RoundState.FAILED


def test_invalid_transitions():
    with pytest.raises(InvalidTransition):
        transition(RoundState.OPEN, RoundState.AGGREGATING)
    with pytest.raises(InvalidTransition):
        transition(RoundState.DONE, RoundState.OPEN)
