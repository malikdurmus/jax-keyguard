"""pytest plugin: run the test session with runtime key reuse checking via --keyguard."""


def pytest_addoption(parser):
    parser.addoption("--keyguard", action="store_true", help="Raise on JAX PRNG key reuse.")


def pytest_configure(config):
    if config.getoption("keyguard"):
        from jax_keyguard import runtime

        runtime.enable()


def pytest_unconfigure(config):
    if config.getoption("keyguard"):
        from jax_keyguard import runtime

        runtime.disable()
