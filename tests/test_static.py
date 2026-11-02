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


# Regression cases: (body of `def f(key, xs, flag):`, lines reported). `r` is jax.random.
CASES = {
    "comprehension over split": ("return [r.normal(k) for k in r.split(key, 3)]", []),
    "nested comprehension over split": (
        "return [[r.normal(k2) for k2 in r.split(k1, 2)] for k1 in r.split(key, 3)]",
        [],
    ),
    "annotated assignment": ("k: jax.Array = r.key(0)\nr.normal(k)\nr.normal(k)", [5]),
    "break then reuse": ("for x in xs:\n    r.normal(key)\n    break\nr.normal(key)", [6]),
    "continue reuse": ("for x in xs:\n    r.normal(key)\n    continue", [4]),
    "raise into handler": (
        "try:\n    r.normal(key)\n    raise ValueError\nexcept ValueError:\n    r.normal(key)",
        [7],
    ),
    "return then finally": ("try:\n    return r.normal(key)\nfinally:\n    r.normal(key)", [6]),
    "with as rebinding": ("r.normal(key)\nwith make() as key:\n    r.normal(key)", []),
    "walrus rebinding": ("r.normal(key)\nif (key := r.key(1)) is not None:\n    r.normal(key)", []),
    "walrus reuse": ("if (k := r.key(1)) is not None:\n    r.normal(k)\n    r.normal(k)", [5]),
    "match capture": ("r.normal(key)\nmatch xs:\n    case key:\n        r.normal(key)", []),
    "while condition": ("while r.uniform(key) > 0.5:\n    pass", [3]),
    "loop over key array": ("keys = r.split(key, 3)\nfor k in keys:\n    r.normal(k)\n    r.normal(k)", [6]),
    "loop over key array ok": ("keys = r.split(key, 3)\nfor k in keys:\n    r.normal(k)", []),
    "enumerate split": ("for i, k in enumerate(r.split(key, 3)):\n    r.normal(k)\n    r.normal(k)", [5]),
    "zip split ok": ("for x, k in zip(xs, r.split(key, 3)):\n    r.normal(k)", []),
    "key_dtype is not a key": ("impl = 'threefry2x32'\nr.key_dtype(impl)\nr.key_dtype(impl)", []),
    "short circuit": ("x = flag and r.normal(key)\nr.uniform(key)", [4]),
    "or reuse": ("x = r.normal(key) or r.normal(key)", [3]),
    "loop var reused after loop": ("for k in r.split(key, 3):\n    r.normal(k)\nr.normal(k)", [5]),
    "reuse via the break path": (
        "for x in xs:\n    r.normal(key)\n    if flag:\n        break\n    key = r.key(x)\n"
        "r.normal(key)",
        [8],
    ),
    # Second round of QA.
    "key array iterated twice": (
        "keys = r.split(key, 3)\nfor k in keys:\n    r.normal(k)\nfor k in keys:\n    r.uniform(k)",
        [7],
    ),
    "key array in repeated comprehension": ("keys = r.split(key, 3)\nfor _ in xs:\n    [r.bits(q) for q in keys]", [5]),
    "key array printed twice ok": (
        "keys = r.split(key, 3)\nfor k in keys:\n    r.normal(k)\nfor k in keys:\n    print(k)",
        [],
    ),
    "exception mid try body": (
        "try:\n    r.normal(key)\n    risky()\n    key = r.key(1)\nexcept ValueError:\n    r.normal(key)", [8]),
    "finally sees entry state": (
        "r.normal(key)\ntry:\n    risky()\n    key = r.key(1)\nfinally:\n    r.split(key)",
        [8],
    ),
    "continue through finally": ("for x in xs:\n    try:\n        continue\n    finally:\n        r.normal(key)", [7]),
    "break through finally": (
        "for x in xs:\n    try:\n        break\n    finally:\n        r.normal(key)\nr.normal(key)", [8]),
    "reuse on third iteration": (
        "for i in xs:\n    if i > 0:\n        r.normal(k)\n    if i == 0:\n        k = r.key(1)", [5]),
    "with suppress then reuse": (
        "with suppress(ValueError):\n    r.normal(key)\n    raise ValueError\nr.normal(key)",
        [6],
    ),
    "with return then dead code ok": (
        "with ctx():\n    if flag:\n        return r.normal(key)\nreturn r.uniform(key)",
        [],
    ),
    "match guard consumes": (
        "match xs:\n    case 1 if r.uniform(key) > 2:\n        pass\n    case _:\n        r.normal(key)", [7]),
    "match wildcard rebinds ok": ("r.normal(key)\nmatch xs:\n    case _:\n        key = r.key(1)\nr.normal(key)", []),
    "while true rebinds ok": ("r.normal(key)\nwhile True:\n    key = r.key(1)\n    break\nr.normal(key)", []),
    "alias": ("k2 = key\nr.normal(key)\nr.normal(k2)", [5]),
    "chained assignment": ("a = b = r.key(0)\nr.normal(a)\nr.normal(b)", [5]),
    "tuple with producer": ("k, n = r.key(0), 3\nr.normal(k)\nr.normal(k)", [5]),
    "tuple swap ok": ("a, b = r.split(key)\na, b = b, a\nr.normal(a)\nr.normal(b)", []),
    "tuple swap then reuse": ("a, b = r.split(key)\na, b = b, a\nr.normal(a)\nr.normal(a)", [6]),
    "tuple rotate ok": ("a, b, c = r.split(key, 3)\na, b, c = b, c, a\nr.normal(a)\nr.normal(b)\nr.normal(c)", []),
    "conditional producer": ("k = r.key(0) if flag else r.PRNGKey(0)\nr.normal(k)\nr.normal(k)", [5]),
    "lambda body": ("g = lambda k: r.normal(k) + r.normal(k)", [3]),
    "default argument": ("def g(x=r.normal(key)):\n    pass\nr.normal(key)", [5]),
    "subscript target": ("xs[r.randint(key, (), 0, 3)] = r.normal(key)", [3]),
    "walrus in comprehension": ("[(k := r.key(i)) for i in xs]\nr.normal(k)\nr.normal(k)", [5]),
}


@pytest.mark.parametrize("name", CASES)
def test_regressions(name):
    body, expected = CASES[name]
    src = "import jax, jax.random as r\ndef f(key, xs, flag):\n" + textwrap.indent(body, "    ") + "\n"
    assert [line for line, _, _ in check_source(src)] == expected


def test_deep_nesting_is_fast():
    import time

    depth = 20
    src = "import jax\ndef f(key):\n"
    for i in range(depth):
        src += "    " * (i + 1) + f"for x{i} in range(2):\n"
    src += "    " * (depth + 1) + "jax.random.normal(key)\n"
    start = time.perf_counter()
    assert len(check_source(src)) == 1
    assert time.perf_counter() - start < 2


def test_columns_count_characters():
    src = "import jax\nk = jax.random.key(0)\njax.random.normal(k)\nñññ = jax.random.normal(k)\n"
    [(_, col, _)] = check_source(src)
    assert col == 24


def test_cli_errors_dont_stop_the_run(tmp_path, capsys):
    (tmp_path / "a_broken.py").write_text("def f(:\n")
    (tmp_path / "b_latin1.py").write_bytes(b"# -*- coding: latin-1 -*-\nimport jax\nk = jax.random.key(0)\n# \xe9\n"
                                           b"jax.random.normal(k)\njax.random.normal(k)\n")
    (tmp_path / "c_bom.py").write_bytes(b"\xef\xbb\xbfimport jax\nk = jax.random.key(0)\njax.random.normal(k)\n"
                                        b"jax.random.normal(k)\n")
    assert main([str(tmp_path), str(tmp_path / "missing.py")]) == 2
    out, err = capsys.readouterr()
    assert "a_broken.py: error:" in err and "missing.py: error:" in err
    assert "b_latin1.py:6:19: KG001" in out and "c_bom.py:4:19: KG001" in out


def test_cli_deduplicates_paths(tmp_path, capsys):
    f = tmp_path / "a.py"
    f.write_text("import jax\nk = jax.random.key(0)\njax.random.normal(k)\njax.random.normal(k)\n")
    assert main([str(f), str(tmp_path), str(f)]) == 1
    assert capsys.readouterr().out.count("KG001") == 1


@pytest.mark.parametrize("suffix, suppressed", [
    ("  # noqa", True),
    ("  # NOQA", True),
    ("  # noqa: KG001", True),
    ("  # noqa:KG", True),
    ("  # noqa: E501", False),
    ("  # noqa: E501, KG001", True),
])
def test_cli_noqa(tmp_path, capsys, suffix, suppressed):
    f = tmp_path / "a.py"
    f.write_text(f"import jax\nk = jax.random.key(0)\njax.random.normal(k)\njax.random.normal(k){suffix}\n")
    assert main([str(f)]) == (0 if suppressed else 1)


def test_cli_noqa_ignores_strings_and_form_feeds(tmp_path):
    f = tmp_path / "a.py"
    f.write_text('import jax\nk = jax.random.key(0)\ns = "\x0c # noqa"\njax.random.normal(k)\njax.random.normal(k)\n')
    assert main([str(f)]) == 1
    f.write_text('import jax\nk = jax.random.key(0)\ns = "\x0c"\njax.random.normal(k)\njax.random.normal(k)  # noqa\n')
    assert main([str(f)]) == 0


def test_star_import():
    src = "from jax.random import *\nk = key(0)\nnormal(k)\nnormal(k)\n"
    assert lines(src) == [4]


def test_imports_are_scoped():
    assert lines("""
    import random
    import jax
    def f(seq):
        random.choice(seq)
        random.choice(seq)
    def g(key):
        from jax import random
        random.normal(key)
    def h(random, key):
        random.choice(key)
        random.choice(key)
    """) == []


def test_long_expression(tmp_path, capsys):
    expr = " + ".join(["1"] * 3000)
    src = f"import jax\nk = jax.random.key(0)\njax.random.normal(k)\nx = {expr}\njax.random.normal(k)\n"
    assert [line for line, _, _ in check_source(src)] == [5]


def test_deep_nesting_with_break_is_fast():
    import time

    depth = 26
    src = "import jax\ndef f(key):\n"
    for i in range(depth):
        src += "    " * (i + 1) + f"for x{i} in range(2):\n"
    src += "    " * (depth + 1) + "jax.random.normal(key)\n" + "    " * (depth + 1) + "break\n"
    start = time.perf_counter()
    check_source(src)
    assert time.perf_counter() - start < 2


def test_cli_skips_directories_named_py(tmp_path):
    (tmp_path / "pkg.py").mkdir()
    (tmp_path / "pkg.py" / "a.py").write_text("x = 1\n")
    assert main([str(tmp_path)]) == 0


def test_flake8_noqa_rules(tmp_path):
    base = "import jax\nk = jax.random.key(0)\njax.random.normal(k)\njax.random.normal(k)"
    for comment, suppressed in [("#noqa", False), ("# noqa: E501 because KG001", False), ("# NoQa: kg001", False)]:
        f = tmp_path / "a.py"
        f.write_text(f"{base}  {comment}\n")
        assert main([str(f)]) == (0 if suppressed else 1), comment
