"""Detect JAX PRNG key reuse, statically from source or at runtime."""

__all__ = ["KeyReuseError", "check_source", "disable", "enable", "key_reuse_guard"]
__version__ = "0.2.0"


def __getattr__(name):
    # Imported lazily so the static checker (CLI, flake8) does not import jax.
    if name == "check_source":
        from jax_keyguard.static import check_source

        return check_source
    if name in __all__:
        from jax_keyguard import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module 'jax_keyguard' has no attribute {name!r}")
