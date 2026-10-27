# jax-keyguard

Find JAX PRNG key reuse — the bug you get when you forget to split a key.

```python
import jax

def train(key, steps):
    for _ in range(steps):
        noise = jax.random.normal(key, (3,))  # same noise every step
```

```
$ jax-keyguard src/
src/train.py:5:35: KG001 PRNG key 'key' reused: already consumed by jax.random.normal in a previous iteration; split it first
```

Install: `pip install jax-keyguard` (Python >= 3.10).

It works in two ways. Use the static checker everywhere and the runtime check
in tests for what static analysis can't see.

## Static checker

Reads source files without running them, so it also covers jitted functions.

- CLI: `jax-keyguard PATH...` (exit status 1 if anything is found).
- flake8: installed automatically as code `KG001` (`flake8 --select KG`).
- Python: `jax_keyguard.check_source(source)` returns `(line, col, message)` tuples.
- Silence a line with `# noqa` (or `# noqa: KG001` under flake8).

Within each function it follows variables that hold keys: values returned by
`jax.random.key`/`PRNGKey`/`split`/`fold_in`, and parameters the function
passes to `jax.random`. A key is consumed when passed to a `jax.random`
function (samplers and `split`; `fold_in` and `key_data` don't consume). It
reports a key consumed twice, including across loop iterations, in
comprehensions, and after either branch of an `if`.

Not seen: keys stored in attributes or containers (`self.key`, `keys[0]`),
keys reused through helper functions (passing a key to your own function does
not count as consuming it, since helpers often `fold_in` it), and closures.

## Runtime check

Raises `KeyReuseError` when a key value is consumed twice, for both typed keys
(`jax.random.key`) and raw keys (`jax.random.PRNGKey`). JAX's built-in
`jax_debug_key_reuse` flag only checks typed keys.

- pytest: `pytest --keyguard` checks the whole test session.
- Anywhere: `jax_keyguard.enable()` / `jax_keyguard.disable()`, or
  `with jax_keyguard.key_reuse_guard(): ...` for a block.

Keys are identified by value, so two keys from the same seed count as the
same key. Calls inside `jit`/`vmap`/`grad` are skipped (use the built-in flag
with typed keys there). It works by patching `jax.random` functions, so names
imported with `from jax.random import normal` before enabling are not
checked, and it is not thread-safe.

## License

Apache-2.0
