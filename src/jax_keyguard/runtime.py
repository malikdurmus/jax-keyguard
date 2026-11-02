"""Runtime detection of JAX PRNG key reuse, for typed and legacy raw keys."""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys
import warnings

import jax
import jax.random
import numpy as np

# Functions whose first parameter is a key but which do not consume it.
_NON_CONSUMING = frozenset(
    {"fold_in", "key", "PRNGKey", "key_data", "wrap_key_data", "key_impl", "clone"}
)


class KeyReuseError(RuntimeError):
    """Raised when a PRNG key that was already consumed is consumed again."""


def _consuming_functions() -> list[str]:
    names = []
    for name in dir(jax.random):
        if name.startswith("_") or name in _NON_CONSUMING:
            continue
        obj = getattr(jax.random, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        try:
            params = list(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            continue
        if params and params[0] == "key":
            names.append(name)
    return names


try:
    from jax._src.core import trace_state_clean as _not_tracing
except ImportError:  # private API; fall back to checking for tracers only
    warnings.warn(
        "jax_keyguard can't tell when JAX is tracing in this JAX version, so a "
        "concrete key closed over by a jitted function may be reported as reused "
        "when the function is retraced.",
        RuntimeWarning,
        stacklevel=2,
    )

    def _not_tracing():
        return True


def _fingerprint(key):
    """Hashable fingerprint of a concrete key's value, or None if not checkable.

    Raw keys are compared as the typed keys they stand for, so key(0),
    PRNGKey(0) and key_data(key(0)) all count as the same key.
    """
    # While tracing (jit, grad, scan, ...) the call runs once per trace, not once
    # per execution, so even a concrete key closed over by the function is skipped.
    if isinstance(key, jax.core.Tracer) or not _not_tracing():
        return None
    if not (isinstance(key, jax.Array) and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)):
        try:
            key = jax.random.wrap_key_data(key)
        except Exception:
            return None
    data = jax.random.key_data(key)
    if isinstance(data, jax.core.Tracer):
        return None
    data = np.asarray(data)
    return (str(jax.random.key_impl(key)), data.shape, data.tobytes())


def _wrap(name, original, consumed):
    @functools.wraps(original)
    def wrapper(key, *args, **kwargs):
        if not _originals:  # checking is off, e.g. a name captured before disable()
            return original(key, *args, **kwargs)
        fp = _fingerprint(key)
        if fp is not None and fp in consumed:
            first_fn, first_loc = consumed[fp]
            raise KeyReuseError(
                f"jax.random.{name} received a PRNG key that was already consumed "
                f"by jax.random.{first_fn} at {first_loc}. Split the key and use "
                f"each subkey only once."
            )
        out = original(key, *args, **kwargs)
        if fp is not None:
            caller = sys._getframe(1)
            consumed[fp] = (name, f"{caller.f_code.co_filename}:{caller.f_lineno}")
        return out

    return wrapper


_originals: dict = {}
_consumed: dict = {}


def enable() -> None:
    """Start checking: consuming an already consumed key raises KeyReuseError."""
    if _originals:
        return
    for name in _consuming_functions():
        _originals[name] = getattr(jax.random, name)
        setattr(jax.random, name, _wrap(name, _originals[name], _consumed))


def reset() -> None:
    """Forget consumed keys without turning checking off."""
    _consumed.clear()


def disable() -> None:
    """Stop checking, restore jax.random and forget consumed keys."""
    for name, fn in _originals.items():
        setattr(jax.random, name, fn)
    _originals.clear()
    _consumed.clear()


@contextlib.contextmanager
def key_reuse_guard():
    """Enable checking for the duration of a ``with`` block."""
    was_enabled = bool(_originals)
    enable()
    try:
        yield
    finally:
        if not was_enabled:
            disable()
