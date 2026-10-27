# jax-keyguard

Detect JAX PRNG key reuse at runtime, for both typed keys (`jax.random.key`)
and legacy raw `uint32` keys (`jax.random.PRNGKey`).

```python
import jax
from jax_keyguard import key_reuse_guard, KeyReuseError

key = jax.random.PRNGKey(0)
with key_reuse_guard():
    x = jax.random.normal(key, (3,))
    y = jax.random.uniform(key, (3,))  # raises KeyReuseError
```

```
KeyReuseError: jax.random.uniform received a PRNG key that was already consumed
by jax.random.normal at /path/to/script.py:6. Split the key and use each subkey only once.
```

Install: `pip install jax-keyguard` (Python >= 3.10).

## Why

JAX has a built-in checker, `jax.config.update("jax_debug_key_reuse", True)`,
but it only tracks typed keys. Raw keys from `jax.random.PRNGKey`, which much
existing code still uses, are silently ignored (verified on JAX 0.12.0.1).
This package covers both.

If your code uses typed keys, or you need checking inside `jit`, prefer the
built-in flag. Use this package for raw keys in eager code.

## Behavior

- Inside `key_reuse_guard()`, every public `jax.random` function whose first
  parameter is `key` (samplers such as `normal`, `uniform`, `bernoulli`, ...,
  and `split`) consumes that key. Passing an already consumed key raises
  `KeyReuseError` (a `RuntimeError`), naming the function and source location
  of the first use.
- Not consuming: `fold_in` (so `fold_in(base, step)` in a loop is fine, as
  with the built-in checker), `key`, `PRNGKey`, `key_data`, `wrap_key_data`,
  `key_impl`, `clone`.
- Detection is by value: a key is identified by its underlying data (plus the
  PRNG implementation for typed keys). Two keys created from the same seed
  count as the same key, which is usually a bug anyway.
- Consumed state belongs to a single guard and is discarded on exit.

## Limitations

- Calls made while tracing (`jit`, `vmap`, `grad`, ...) are skipped silently,
  because the key is an abstract tracer with no value. Use the built-in flag
  with typed keys for traced code.
- The guard works by temporarily replacing attributes on the `jax.random`
  module. Names bound before entering the guard, e.g. via
  `from jax.random import normal`, are not intercepted.
- Patching module attributes is process-global and not thread-safe.

## License

Apache-2.0
