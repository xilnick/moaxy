"""Shared test fixtures for the moaxy test suite.

Reusable building blocks (scripted adapters, response factories, and
context builders) live here so individual test files can import them
without re-declaring the same boilerplate.
"""

from tests.fixtures.fake_adapter import FakeAdapter

__all__ = ["FakeAdapter"]
