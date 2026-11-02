"""pytest plugin: run each test with runtime key reuse checking via --keyguard."""


def pytest_addoption(parser):
    parser.addoption("--keyguard", action="store_true", help="Raise on JAX PRNG key reuse.")


def pytest_runtest_setup(item):
    # Each test starts with no consumed keys, and with checking on even if an
    # earlier test called disable().
    if item.config.getoption("keyguard"):
        from jax_keyguard import runtime

        runtime.enable()
        runtime.reset()


def pytest_unconfigure(config):
    if config.getoption("keyguard"):
        from jax_keyguard import runtime

        runtime.disable()
