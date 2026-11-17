# config.py

"""Global configuration for linox warnings and debug behavior.

Usage:
- Toggle debug prints (e.g., densification warnings):
    from linox.config import set_debug
    set_debug(True)

- Or via environment variable:
    export LINOX_DEBUG=1
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linox._types import AnyType, CallableType

_DEBUG: bool = os.getenv("LINOX_DEBUG", "0") not in {"0", "false", "False", ""}
_DEBUG_HOOK: CallableType[[DebugEvent], None] | None = None


@dataclass(frozen=True)
class DebugEvent:
    """A single debug or profiling event emitted by the library."""

    kind: str  # e.g. "densify", "solve_fallback", "eigh_dense"
    msg: str
    op_type: str | None = None
    op_id: int | None = None
    shape: AnyType = None
    dtype: AnyType = None
    meta: dict[str, AnyType] | None = None
    t: float = 0.0
    duration: float | None = None
    phase: str | None = None  # "start" or "end"


@contextlib.contextmanager
def profile(kind: str, msg: str, **kwargs) -> None:
    """Context manager to profile an operation time."""
    t0 = time.time()
    # emit start event
    emit(DebugEvent(kind=kind, msg=msg, phase="start", t=t0, **kwargs))
    try:
        yield
    finally:
        t1 = time.time()
        # emit end event with duration
        emit(DebugEvent(kind=kind, msg=msg, phase="end", t=t1, duration=t1 - t0, **kwargs))


def set_debug(value: bool) -> None:
    """Enable or disable debug mode (controls warning prints)."""
    global _DEBUG
    _DEBUG = bool(value)


def is_debug() -> bool:
    """Return whether debug mode is enabled."""
    return _DEBUG


def set_debug_hook(hook: CallableType[[DebugEvent], None] | None) -> None:
    """Register/unregister a debug hook that receives DebugEvent objects."""
    global _DEBUG_HOOK
    _DEBUG_HOOK = hook


def emit(event: DebugEvent) -> None:
    """Emit a structured debug event to the hook (if any)."""
    if _DEBUG_HOOK is not None:
        # set timestamp lazily
        if event.t == 0.0:
            object.__setattr__(event, "t", time.time())  # dataclass frozen workaround
        _DEBUG_HOOK(event)


def warn(msg: str, *, prefix: str = "Warning") -> None:
    """Conditionally print a warning message if debug is enabled.

    Args:
        msg: Message to print.
        prefix: Optional prefix for the message, defaults to 'Warning'.
    """
    text = f"{prefix}: {msg}"
    emit(DebugEvent(kind="warn", msg=text))
    if _DEBUG:
        print(text, file=sys.stderr)


_MAX_DENSE_N: int = 2000


def set_max_dense_n(n: int) -> None:
    """Set the maximum size for automatic densification."""
    global _MAX_DENSE_N
    _MAX_DENSE_N = int(n)


def get_max_dense_n() -> int:
    """Get the maximum size for automatic densification."""
    return _MAX_DENSE_N


# --------------------------------------------------------------------------- #
# Method Selection Configuration
# --------------------------------------------------------------------------- #

_WARN_ON_DENSIFY: bool = False
_DEFAULT_METHODS: dict[str, str] = {}


def set_warn_on_densify(value: bool) -> None:
    """Enable or disable warnings when operations trigger densification."""
    global _WARN_ON_DENSIFY
    _WARN_ON_DENSIFY = bool(value)


def get_warn_on_densify() -> bool:
    """Return whether densification warnings are enabled."""
    return _WARN_ON_DENSIFY


def set_default_method(operation: str, method: str) -> None:
    """Set the default method for a specific operation (e.g. 'eigh', 'solve')."""
    _DEFAULT_METHODS[operation] = method


#: Method names each operation accepts. Anything outside these sets is a typo
#: on the caller's part, and silently falling back to the default hides real
#: bugs (e.g. ``sqrt(a, method="lanczos")`` quietly returning the exact result).
VALID_METHODS: dict[str, frozenset[str]] = {
    "trace": frozenset({"auto", "exact", "hutchinson"}),
    "slogdet": frozenset({"auto", "exact", "slq"}),
    "solve": frozenset({"auto", "exact", "lsmr", "cg", "conjugate_gradient"}),
    "inverse": frozenset({"auto", "exact", "approx", "lsmr", "cg"}),
    "sqrt": frozenset({"auto", "exact", "approx", "lanczos"}),
    "eigh": frozenset({"auto", "exact", "approx", "lanczos"}),
    "pinverse": frozenset({"auto", "exact"}),
}


def validate_method(operation: str, requested_method: str) -> str:
    """Check ``requested_method`` against the methods ``operation`` supports.

    Raises
    ------
    ValueError
        If the method is not one this operation understands.
    """
    valid = VALID_METHODS.get(operation)
    if valid is not None and requested_method not in valid:
        msg = f"Unknown method {requested_method!r} for operation {operation!r}. Valid methods are: {', '.join(sorted(valid))}."
        raise ValueError(msg)
    return requested_method


def resolve_method(operation: str, op: AnyType, requested_method: str) -> str:
    """Resolve the execution method based on request, config, and operator properties.

    Priority:
    1. Explicitly requested method (if not 'auto')
    2. Configured default for this operation
    3. 'auto' heuristics (based on size, structure, etc.)

    Args:
        operation: Name of the operation ('solve', 'eigh', 'sqrt', etc.)
        op: The linear operator involved
        requested_method: The method argument provided by the user

    Returns
    -------
        The resolved method name (e.g. 'exact', 'lanczos', 'cg').

    Raises
    ------
    ValueError
        If ``requested_method`` is not valid for ``operation``.
    """
    validate_method(operation, requested_method)

    if requested_method != "auto":
        return requested_method

    # Check config defaults
    if operation in _DEFAULT_METHODS:
        return _DEFAULT_METHODS[operation]

    # 'auto' heuristics
    # Basic logic: use exact if small enough, otherwise approx
    n = op.shape[-1]
    if n <= _MAX_DENSE_N:
        return "exact"

    # Default approx fallbacks for large operators
    if operation == "trace":
        # For large operators, default to Hutchinson
        return "hutchinson"
    if operation == "slogdet":
        # For large operators, default to SLQ (if implemented) or fallback
        # Currently we might not have SLQ hooked up everywhere, so be careful.
        # But 'slq' is the intended approx backend.
        return "slq"
    if operation == "inverse":
        return "lsmr"  # Approx inverse for large scale
    if operation == "solve":
        return "lsmr"
    if operation == "sqrt":
        return "lanczos"
    if operation == "eigh":
        return "lanczos"

    # Fallback to exact (which might fail or be slow if dense)
    return "exact"
