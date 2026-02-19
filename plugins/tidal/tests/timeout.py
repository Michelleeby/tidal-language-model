"""
timeout.py

Test timeout utilities for preventing hanging tests.
Provides a @timeout decorator and TimedTestCase base class
that kills any test method exceeding a configurable time limit.

Uses signal.alarm (POSIX only) — safe for single-threaded test runners.
"""

import signal
import unittest
import functools


class TestTimeoutError(Exception):
    """Raised when a test exceeds its time limit."""
    pass


def timeout(seconds):
    """Decorator that raises TestTimeoutError if a test takes longer than `seconds`.

    Usage:
        @timeout(10)
        def test_something(self):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _handler(signum, frame):
                raise TestTimeoutError(
                    f"{func.__qualname__} timed out after {seconds}s"
                )

            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


class TimedTestCase(unittest.TestCase):
    """Base test class that applies a default timeout to every test_* method.

    Subclass this instead of unittest.TestCase to get automatic 30s timeouts.
    Override `_timeout_seconds` at the class level to change the default.

    Uses `run()` wrapping so subclasses don't need to call super().setUp().
    """

    _timeout_seconds = 30

    def run(self, result=None):
        def _handler(signum, frame):
            raise TestTimeoutError(
                f"{self._testMethodName} timed out after {self._timeout_seconds}s"
            )

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self._timeout_seconds)
        try:
            return super().run(result)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
