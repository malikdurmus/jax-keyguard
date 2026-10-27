import textwrap

import pytest

from jax_keyguard.static import check_source, main


def lines(src):
    return [line for line, _, _ in check_source(textwrap.dedent(src))]


def test_reuse_reported():
    src = """
    import jax
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3,))
    y = jax.random.uniform(key, (3,))
    """
    [(line, _, msg)] = check_source(textwrap.dedent(src))
    assert line == 5
    assert "'key' reused: already consumed by jax.random.normal on line 4" in msg


def test_split_pattern_ok():
    assert lines("""
    import jax
    key = jax.random.key(0)
    key, sub = jax.random.split(key)
    x = jax.random.normal(sub)
    key, sub = jax.random.split(key)
    y = jax.random.normal(sub)
    """) == []


def test_use_after_split_reported():
    assert lines("""
    from jax import random
    key = random.key(0)
    k1, k2 = random.split(key)
    x = random.normal(key)
    """) == [5]


def test_loop_without_split_reported():
    assert lines("""
    import jax.random as jr
    def f(key):
        for _ in range(3):
            x = jr.normal(key)
    """) == [5]


def test_loop_with_split_ok():
    assert lines("""
    import jax
    def f(key):
        for _ in range(3):
            key, sub = jax.random.split(key)
            x = jax.random.normal(sub)
    """) == []


def test_comprehension_reported():
    assert lines("""
    from jax.random import normal
    def f(key):
        return [normal(key) for _ in range(3)]
    """) == [4]


def test_jitted_function_reported():
    assert lines("""
    import jax
    @jax.jit
    def f(key):
        return jax.random.normal(key) + jax.random.normal(key)
    """) == [5]


def test_only_jax_random_calls_consume():
    # Helpers may legitimately fold_in the key, so other calls don't consume it.
    assert lines("""
    import jax
    def f(model, key):
        model.init(key)
        model.apply(key)
        return jax.random.normal(key)
    """) == []


def test_comprehension_variable_shadows():
    assert lines("""
    import jax
    def init(rng, fns):
        rngs = jax.random.split(rng, len(fns))
        return [fn(jax.random.normal(rng)) for fn, rng in zip(fns, rngs)]
    """) == []


def test_fold_in_and_inspection_ok():
    assert lines("""
    import jax
    def f(key):
        for step in range(3):
            x = jax.random.normal(jax.random.fold_in(key, step))
        print(key)
        jax.debug.print("{}", key)
        return jax.random.key_data(key)
    """) == []


def test_branches():
    # One use per path is fine; a use after either branch consumed it is not.
    assert lines("""
    import jax
    def f(key, flag):
        if flag:
            x = jax.random.normal(key)
        else:
            x = jax.random.uniform(key)
        y = jax.random.normal(key) if flag else jax.random.uniform(key)
    """) == [8, 8]


def test_early_return_ok():
    assert lines("""
    import jax
    def f(key, flag):
        if flag:
            return jax.random.normal(key)
        return jax.random.uniform(key)
    """) == []


def test_non_key_names_ignored():
    # 'key' is never passed to jax.random here, so it is a dict key, not a PRNG key.
    assert lines("""
    import jax
    def get(cache, key):
        cache.get(key)
        cache.pop(key)
    """) == []


def test_files_without_jax_ignored():
    assert lines("""
    def f(key):
        normal(key)
        normal(key)
    """) == []


def test_cli(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("import jax\nk = jax.random.key(0)\njax.random.normal(k)\njax.random.normal(k)\n")
    ok = tmp_path / "ok.py"
    ok.write_text("import jax\nk = jax.random.key(0)\njax.random.normal(k)\njax.random.normal(k)  # noqa\n")
    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{bad}:4:19: KG001" in out and "ok.py" not in out
    assert main([str(ok)]) == 0


@pytest.mark.parametrize("module", ["jax_keyguard", "jax_keyguard.static"])
def test_static_does_not_import_jax(module):
    import subprocess
    import sys

    code = f"import {module}, sys; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_messages_distinguish_loops():
    src = """
    import jax
    def f(key):
        return jax.random.normal(key) + jax.random.normal(key)
    def g(key):
        for _ in range(2):
            jax.random.normal(key)
    """
    msgs = [msg for _, _, msg in check_source(textwrap.dedent(src))]
    assert "on line 4" in msgs[0] and "in a previous iteration" in msgs[-1]
