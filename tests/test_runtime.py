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
