"""Invocation-local file policy and cooperative cancellation for tools."""

import time
from contextvars import ContextVar
from threading import RLock

from ..cancellation import check_cancellation

path_policy = ContextVar("orbit_path_policy", default=None)
tool_deadline = ContextVar("orbit_tool_deadline", default=None)
file_mutation_lock = RLock()


def check_deadline() -> None:
    check_cancellation()
    deadline = tool_deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("tool execution deadline exceeded")


def can_read_path(path) -> bool:
    check_deadline()
    policy = path_policy.get()
    return policy is None or policy(path)
