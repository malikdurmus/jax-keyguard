"""Runtime detection of JAX PRNG key reuse, for typed and legacy raw keys."""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys

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


def _fingerprint(key):
    """Hashable fingerprint of a concrete key's value, or None if not checkable."""
    if isinstance(key, jax.core.Tracer):
        return None
    if isinstance(key, jax.Array) and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        data = np.asarray(jax.random.key_data(key))
        impl = str(jax.random.key_impl(key))
        return ("typed", impl, key.shape, data.tobytes())
    try:
        data = np.asarray(key)
    except Exception:
        return None
    return ("raw", data.dtype.str, data.shape, data.tobytes())


def _wrap(name, original, consumed):
    @functools.wraps(original)
    def wrapper(key, *args, **kwargs):
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
