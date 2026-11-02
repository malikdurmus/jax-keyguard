import jax
import pytest

from jax_keyguard import KeyReuseError, key_reuse_guard

KEY_MAKERS = [jax.random.key, jax.random.PRNGKey]


@pytest.mark.parametrize("make_key", KEY_MAKERS)
def test_reuse_raises(make_key):
    key = make_key(0)
    with key_reuse_guard():
        jax.random.normal(key, (3,))
        with pytest.raises(KeyReuseError, match=r"normal.*test_runtime\.py:\d+"):
            jax.random.uniform(key, (3,))


@pytest.mark.parametrize("make_key", KEY_MAKERS)
def test_split_then_use_passes(make_key):
    key = make_key(0)
    with key_reuse_guard():
        for _ in range(5):
            key, sub = jax.random.split(key)
            jax.random.normal(sub, (2,))


@pytest.mark.parametrize("make_key", KEY_MAKERS)
def test_use_after_split_raises(make_key):
    key = make_key(0)
    with key_reuse_guard():
        jax.random.split(key)
        with pytest.raises(KeyReuseError, match="split"):
            jax.random.bernoulli(key)


@pytest.mark.parametrize("make_key", KEY_MAKERS)
def test_fold_in_loop_passes(make_key):
    base = make_key(0)
    with key_reuse_guard():
        for step in range(5):
            jax.random.normal(jax.random.fold_in(base, step), (2,))


@pytest.mark.parametrize("make_key", KEY_MAKERS)
def test_jit_not_affected(make_key):
    @jax.jit
    def f(key):
        return jax.random.normal(key) + jax.random.normal(key)

    key = make_key(0)
    with key_reuse_guard():
        f(key)
        f(key)


def test_originals_restored():
    normal, split = jax.random.normal, jax.random.split
    with key_reuse_guard():
        assert jax.random.normal is not normal
    assert jax.random.normal is normal and jax.random.split is split

    key = jax.random.key(0)
    with pytest.raises(KeyReuseError):
        with key_reuse_guard():
            jax.random.normal(key)
            jax.random.normal(key)
    assert jax.random.normal is normal and jax.random.split is split
    # Outside the guard, reuse is not checked.
    jax.random.normal(key)
    jax.random.normal(key)


def test_state_is_per_guard():
    key = jax.random.PRNGKey(0)
    with key_reuse_guard():
        jax.random.normal(key)
    with key_reuse_guard():
        jax.random.normal(key)


def test_enable_disable():
    from jax_keyguard import disable, enable

    key = jax.random.PRNGKey(0)
    enable()
    try:
        jax.random.normal(key)
        with pytest.raises(KeyReuseError):
            jax.random.normal(key)
    finally:
        disable()
    jax.random.normal(key)


def test_pytest_flag(pytester):
    pytester.makepyfile("""
        import jax

        def test_reuse():
            key = jax.random.PRNGKey(0)
            jax.random.normal(key)
            jax.random.normal(key)
    """)
    pytester.runpytest().assert_outcomes(passed=1)
    pytester.runpytest("--keyguard").assert_outcomes(failed=1)


@pytest.mark.parametrize("make_key", KEY_MAKERS)
def test_closed_over_key_under_tracing_is_skipped(make_key):
    import jax.numpy as jnp

    key = make_key(0)
    f = jax.jit(lambda x: x + jax.random.normal(key, x.shape))
    g = jax.grad(lambda x: x * jax.random.normal(key))
    with key_reuse_guard():
        f(jnp.ones(3))
        f(jnp.ones(4))  # retrace
        g(1.0)
        g(2.0)


def test_reset():
    from jax_keyguard import reset

    key = jax.random.PRNGKey(0)
    with key_reuse_guard():
        jax.random.normal(key)
        reset()
        jax.random.normal(key)


def test_pytest_flag_is_per_test(pytester):
    pytester.makepyfile("""
        import jax, jax_keyguard, pytest

        def test_one():
            jax.random.normal(jax.random.PRNGKey(0))

        def test_two():
            jax.random.normal(jax.random.PRNGKey(0))

        @pytest.mark.parametrize("n", [1, 2, 3])
        def test_param(n):
            jax.random.uniform(jax.random.key(42), (n,))

        def test_disables():
            jax_keyguard.disable()

        def test_still_checked():
            key = jax.random.PRNGKey(1)
            jax.random.normal(key)
            jax.random.normal(key)
    """)
    result = pytester.runpytest("--keyguard")
    result.assert_outcomes(passed=6, failed=1)
    result.stdout.fnmatch_lines(["FAILED *test_still_checked*"])


def test_dir_lists_public_api():
    import jax_keyguard

    assert {"check_source", "enable", "disable", "reset", "key_reuse_guard", "KeyReuseError"} <= set(dir(jax_keyguard))


def test_captured_names_stop_checking_after_disable():
    with key_reuse_guard():
        from jax.random import normal
    key = jax.random.key(0)
    normal(key)
    normal(key)


def test_raw_and_typed_forms_match():
    with key_reuse_guard():
        jax.random.normal(jax.random.key(3))
        with pytest.raises(KeyReuseError):
            jax.random.normal(jax.random.PRNGKey(3))
        k = jax.random.key(4)
        jax.random.normal(k)
        with pytest.raises(KeyReuseError):
            jax.random.normal(jax.random.key_data(k))
