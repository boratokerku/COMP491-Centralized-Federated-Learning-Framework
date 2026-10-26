"""Coordinator: drives the round state machine. PostgreSQL is the source of truth, so a
restarted coordinator resumes by reading the current run/round state."""
