"""Cancellation primitives shared by the agent, providers, and tools."""

from contextvars import ContextVar
from threading import Event


class ToolInterrupted(RuntimeError):
    """Raised when the user asks the active agent turn to stop."""


class CancellationToken:
    """Thread-safe, cooperative cancellation shared by one agent turn."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ToolInterrupted("interrupted by user")


cancellation_token = ContextVar("orbit_cancellation_token", default=None)


def check_cancellation() -> None:
    token = cancellation_token.get()
    if token is not None:
        token.raise_if_cancelled()
