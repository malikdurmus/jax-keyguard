# jax-keyguard

Catches PRNG key reuse in JAX code.

In JAX, a key passed to a random function gives the same numbers every time.
If you forget to split it before using it again, your "random" values repeat
and nothing complains. This package finds that mistake, either by reading your
code or while it runs.

```
pip install jax-keyguard
```

Requires Python 3.10 or newer and JAX 0.4.18 or newer.

## Why not JAX's own check?

JAX 0.4.26 added a `jax_debug_key_reuse` option, but it only checks keys made
with `jax.random.key`. Keys made with `jax.random.PRNGKey`, which most existing
code and tutorials use, are not checked, even in the latest JAX. Use this
package if:

- your code uses `jax.random.PRNGKey` keys, on any JAX version,
- you are on a JAX version older than 0.4.26, which has no check at all, or
- you want to find reuse in CI without running the code. The static checker
  only reads your source, so it works with any key type and any JAX version.

If your code only uses `jax.random.key` keys on a recent JAX, turn on
`jax_debug_key_reuse` for runtime checking. It also works inside `jit`, which
the runtime check here does not. The static checker is still useful in CI.

## Checking your code

Say `src/model.py` contains this:

```python
import jax


def init(key):
    w = jax.random.normal(key, (3, 3))
    b = jax.random.normal(key, (3,))
    return w, b


@jax.jit
def dropout(key, x):
    keys = jax.random.split(key)
    mask = jax.random.bernoulli(key, 0.9, x.shape)
    return x * mask
```

Run the checker on a file or a directory:

```
$ jax-keyguard src/
src/model.py:6:27: KG001 PRNG key 'key' reused: already consumed by jax.random.normal on line 5; split it first
src/model.py:13:33: KG001 PRNG key 'key' reused: already consumed by jax.random.split on line 12; split it first
```

The exit status is 1 when something is found and 2 on an error, such as a file
that can't be read or parsed (the other files are still checked), so it can
run in CI as is. The code is never executed, which means jitted functions are
checked too.

The fix is to split first and use each subkey once:

```python
def init(key):
    w_key, b_key = jax.random.split(key)
    w = jax.random.normal(w_key, (3, 3))
    b = jax.random.normal(b_key, (3,))
    return w, b
```

### With flake8

Installing the package registers it with flake8 under the code `KG001`:

```
$ flake8 --select KG src/
```

### Ignoring a line

Add `# noqa` to the line (or `# noqa: KG001` when using flake8).

### From Python

```python
from jax_keyguard import check_source

for line, col, message in check_source(open("src/model.py").read()):
    print(line, message)
```

### What it catches

Inside each function it keeps track of variables holding keys: anything
returned by `jax.random.key`, `PRNGKey`, `split`, `fold_in` or `clone`, loop
variables over those, plain copies like `k2 = key`, and parameters the function
passes to `jax.random`. A key counts as used when it goes into a `jax.random`
sampler or `split`. These are reported:

```python
x = jax.random.normal(key)
y = jax.random.uniform(key)          # same key twice

k1, k2 = jax.random.split(key)
z = jax.random.normal(key)           # key used after splitting it

for _ in range(steps):
    noise = jax.random.normal(key)   # same key on every iteration

samples = [jax.random.normal(key) for _ in range(3)]

keys = jax.random.split(key, 3)
for k in keys:
    a = jax.random.normal(k)
for k in keys:
    b = jax.random.uniform(k)        # the same three keys again
```

These are fine:

```python
for step in range(steps):
    step_key = jax.random.fold_in(key, step)   # fold_in doesn't use up the key

if training:
    x = jax.random.normal(key)
else:
    x = jax.random.uniform(key)                # one use per branch

samples = [jax.random.normal(k) for k in jax.random.split(key, 3)]
```

### What it misses

- Keys kept in attributes or containers, like `self.key` or `keys[0]`.
- Keys returned by your own functions, like `key = make_key(seed)`.
- Reuse through your own functions. Calling `helper(key)` twice is not
  reported, because helpers often call `fold_in` on the key, which is fine.
- Keys a nested function or lambda uses from the enclosing one.
- Keys used indirectly: `functools.partial(jax.random.normal, key)`,
  `jax.vmap(jax.random.normal)(keys)`, or passed along as `*args`.
- Different keys that happen to have the same value, like
  `jax.random.fold_in(key, 0)` inside a loop.

The runtime check below covers most of these for code that actually runs.

It also doesn't know which way an `if` goes or how many times a loop runs, so
it reports a reuse that could happen on any path. A key used in a loop that
only ever runs once, or in an `except` block that can't be reached, still
counts. Add `# noqa` in those cases.

## Checking at runtime

This raises `KeyReuseError` as soon as a key is used a second time. It works
with both kinds of keys, `jax.random.key` and the older `jax.random.PRNGKey`,
on any supported JAX version.

### In tests

Each test starts with a clean slate, so tests that all begin with
`PRNGKey(0)` don't interfere with each other.

```
$ pytest --keyguard
...
E   jax_keyguard.runtime.KeyReuseError: jax.random.uniform received a PRNG key that was already consumed by jax.random.normal at /path/to/tests/test_model.py:6. Split the key and use each subkey only once.
```

### In a script

```python
import jax_keyguard

jax_keyguard.enable()
```

Turn it off again with `jax_keyguard.disable()`, or call
`jax_keyguard.reset()` to forget the keys used so far. To check only part of
the code, use a `with` block:

```python
with jax_keyguard.key_reuse_guard():
    run_experiment()
```

### Things to know

- Keys are compared by value, so two keys made from the same seed count as
  one key. That includes `key(0)` and `PRNGKey(0)`, a key and its
  `key_data`, and a key and its `jax.random.clone`. The static checker treats
  a clone as a new key.
- Nothing is checked while JAX traces a function for `jit`, `vmap`, `grad`,
  `scan` and so on, because that code runs once per trace rather than once per
  call. Use JAX's `jax_debug_key_reuse` with `jax.random.key` keys there, or
  rely on the static checker.
- It slows down random calls. Each checked call copies the key back from the
  device, which also stops JAX from dispatching work ahead asynchronously. A
  loop of `split` and `normal` on CPU ran about twice as slowly with checking
  on. It is meant for tests and debugging, not for full training runs.
- It remembers every key used since checking was turned on (or since the last
  `reset()`), so memory grows by a little with each random call. That is fine
  in tests, where `--keyguard` resets before each test. In a long script, call
  `jax_keyguard.reset()` now and then, for example once per epoch, keeping in
  mind that reuse across a reset is not caught.
- It works by replacing the functions on the `jax.random` module. A name
  imported earlier with `from jax.random import normal` is not checked.
- Reuse inside a `jax.pure_callback` shows up as an XLA error that wraps the
  `KeyReuseError`.
- It is not thread-safe.

## License

Apache-2.0
