#!/usr/bin/env python3
"""
Tests for retry utilities

Ensures retry logic works correctly with exponential backoff.
"""

import time
from unittest.mock import Mock

import pytest

from knowledge_extraction.utils.retry import (
    DATABASE_RETRY_CONFIG,
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    exponential_backoff,
    retry,
    retry_with_backoff,
)


class TestExponentialBackoff:
    """Test exponential backoff calculation"""

    def test_basic_exponential_growth(self):
        """Backoff should grow exponentially"""
        # Without jitter for predictable tests
        assert exponential_backoff(0, base_delay=1.0, jitter=False) == 1.0
        assert exponential_backoff(1, base_delay=1.0, jitter=False) == 2.0
        assert exponential_backoff(2, base_delay=1.0, jitter=False) == 4.0
        assert exponential_backoff(3, base_delay=1.0, jitter=False) == 8.0

    def test_max_delay_cap(self):
        """Backoff should not exceed max_delay"""
        delay = exponential_backoff(10, base_delay=1.0, max_delay=60.0, jitter=False)
        assert delay == 60.0

    def test_custom_base_delay(self):
        """Custom base delay should be respected"""
        assert exponential_backoff(0, base_delay=2.0, jitter=False) == 2.0
        assert exponential_backoff(1, base_delay=2.0, jitter=False) == 4.0

    def test_jitter_adds_randomness(self):
        """Jitter should add randomness to prevent thundering herd"""
        delays = [exponential_backoff(1, jitter=True) for _ in range(10)]
        # All delays should be different (with high probability)
        assert len(set(delays)) > 1

        # All delays should be within reasonable range (2.0 ± 25%)
        for delay in delays:
            assert 1.5 <= delay <= 2.5

    def test_jitter_minimum_delay(self):
        """Jitter should ensure minimum 0.1s delay"""
        delay = exponential_backoff(0, base_delay=0.01, jitter=True)
        assert delay >= 0.1


class TestRetryWithBackoff:
    """Test retry_with_backoff function"""

    def test_success_on_first_try(self):
        """Function succeeding on first try should not retry"""
        mock_func = Mock(return_value="success")

        result = retry_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_failures(self):
        """Function should retry and succeed eventually"""
        mock_func = Mock(side_effect=[
            ConnectionError("fail"),
            ConnectionError("fail"),
            "success"
        ])

        result = retry_with_backoff(
            mock_func,
            max_retries=3,
            base_delay=0.01  # Fast for testing
        )

        assert result == "success"
        assert mock_func.call_count == 3

    def test_exhausts_retries(self):
        """Should raise exception after exhausting retries"""
        mock_func = Mock(side_effect=ConnectionError("always fails"))

        with pytest.raises(ConnectionError):
            retry_with_backoff(mock_func, max_retries=2, base_delay=0.01)

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_non_retryable_exception_fails_immediately(self):
        """Non-retryable exceptions should not be retried"""
        mock_func = Mock(side_effect=ValueError("invalid input"))

        with pytest.raises(ValueError):
            retry_with_backoff(
                mock_func,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError, TimeoutError)
            )

        assert mock_func.call_count == 1  # No retries

    def test_custom_retryable_exceptions(self):
        """Should retry custom exceptions"""
        mock_func = Mock(side_effect=[
            ValueError("retry this"),
            "success"
        ])

        result = retry_with_backoff(
            mock_func,
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(ValueError,)
        )

        assert result == "success"
        assert mock_func.call_count == 2

    def test_on_retry_callback(self):
        """on_retry callback should be called on each retry"""
        mock_func = Mock(side_effect=[
            ConnectionError("fail"),
            "success"
        ])
        mock_callback = Mock()

        retry_with_backoff(
            mock_func,
            max_retries=2,
            base_delay=0.01,
            on_retry=mock_callback
        )

        # Callback called once (after first failure, before retry)
        assert mock_callback.call_count == 1

        # Check callback arguments
        call_args = mock_callback.call_args[0]
        assert isinstance(call_args[0], ConnectionError)  # exception
        assert call_args[1] == 0  # attempt number
        assert isinstance(call_args[2], float)  # delay

    def test_passes_args_and_kwargs(self):
        """Should pass arguments correctly to function"""
        mock_func = Mock(return_value="success")

        retry_with_backoff(
            mock_func,
            1,  # max_retries
            0.01,  # base_delay
            60.0,  # max_delay
            (ConnectionError, TimeoutError, OSError),  # retryable_exceptions
            None,  # on_retry
            "arg1",
            "arg2",
            kwarg1="value1"
        )

        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")


class TestRetryDecorator:
    """Test @retry decorator"""

    def test_decorator_success(self):
        """Decorated function should work normally on success"""
        @retry(max_retries=3, base_delay=0.01)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_decorator_with_failures(self):
        """Decorated function should retry on failure"""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring"""
        @retry(max_retries=1)
        def my_function():
            """This is my function"""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my function"

    def test_decorator_with_arguments(self):
        """Decorated function should handle arguments correctly"""
        @retry(max_retries=1, base_delay=0.01)
        def add(a, b, c=0):
            return a + b + c

        result = add(2, 3)
        assert result == 5

        result = add(2, 3, c=5)
        assert result == 10


class TestRetryConfig:
    """Test RetryConfig class"""

    def test_default_config(self):
        """Default config should have sensible defaults"""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0

    def test_custom_config(self):
        """Should accept custom values"""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0

    def test_retry_decorator_from_config(self):
        """Should create decorator from config"""
        config = RetryConfig(max_retries=2, base_delay=0.01)
        decorator = config.retry_decorator()

        call_count = 0

        @decorator
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2


class TestPredefinedConfigs:
    """Test predefined retry configurations"""

    def test_default_config_exists(self):
        """DEFAULT_RETRY_CONFIG should exist"""
        assert DEFAULT_RETRY_CONFIG.max_retries == 3
        assert DEFAULT_RETRY_CONFIG.base_delay == 1.0

    def test_database_config_exists(self):
        """DATABASE_RETRY_CONFIG should exist"""
        assert DATABASE_RETRY_CONFIG.max_retries == 4
        assert DATABASE_RETRY_CONFIG.base_delay == 1.0
        assert DATABASE_RETRY_CONFIG.max_delay == 30.0


class TestIntegrationWithRealDelays:
    """Integration tests with actual delays (slower tests)"""

    def test_actual_retry_timing(self):
        """Test that actual delays are applied"""
        call_times = []

        @retry(max_retries=2, base_delay=0.1)
        def timed_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ConnectionError("fail")
            return "success"

        start = time.time()
        result = timed_func()
        duration = time.time() - start

        assert result == "success"
        assert len(call_times) == 3

        # Check that retries happened (duration > 0.1s)
        # With jitter, exact timing varies but should be > 0.1s
        assert duration > 0.1

    def test_max_delay_actually_caps(self):
        """Test that max_delay actually caps the delay"""
        call_times = []

        @retry(max_retries=1, base_delay=100.0, max_delay=0.1)
        def timed_func():
            call_times.append(time.time())
            if len(call_times) < 2:
                raise ConnectionError("fail")
            return "success"

        start = time.time()
        timed_func()
        duration = time.time() - start

        # Should cap at 0.1s, not use 100s base_delay
        assert duration < 1.0  # Much less than 100s
