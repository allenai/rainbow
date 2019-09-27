"""Configuration for running tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        help="Skip slow tests, useful for quick checks.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option is turned on.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
