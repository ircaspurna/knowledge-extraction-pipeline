#!/usr/bin/env python3
"""
Retry utilities with exponential backoff

Provides decorators and functions for retrying operations that may fail
due to transient errors (network issues, rate limits, temporary failures).
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Type, TypeVar, cast

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])


# Default retryable exceptions (transient failures)
DEFAULT_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Includes network errors
)


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay.

    Formula: min(base_delay * (2 ^ attempt), max_delay)

    Args:
        attempt: Current retry attempt (0-indexed)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        jitter: Add random jitter to prevent thundering herd (default: True)

    Returns:
        Delay in seconds to wait before next retry

    Examples:
        >>> exponential_backoff(0)  # First retry
        1.0
        >>> exponential_backoff(1)  # Second retry
        2.0
        >>> exponential_backoff(2)  # Third retry
        4.0
        >>> exponential_backoff(10)  # Capped at max_delay
        60.0
    """
    delay: float = min(base_delay * (2 ** attempt), max_delay)

    if jitter:
        # Add random jitter ±25% to prevent synchronized retries
        import random
        jitter_range: float = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)
        # Ensure minimum 0.1s delay
        if delay < 0.1:
            delay = 0.1

    return delay


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[Exception, int, float], None] | None = None,
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Execute a function with retry logic and exponential backoff.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback function(exception, attempt, delay)
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Return value of func

    Raises:
        Exception: Re-raises the last exception if all retries fail

    Examples:
        >>> def flaky_api_call():
        ...     # May fail with ConnectionError
        ...     return requests.get("https://api.example.com")
        >>>
        >>> result = retry_with_backoff(flaky_api_call, max_retries=5)
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)

        except retryable_exceptions as e:
            last_exception = e

            # If this was the last attempt, re-raise
            if attempt >= max_retries:
                func_name = getattr(func, '__name__', repr(func))
                logger.error(
                    f"Function {func_name} failed after {max_retries + 1} attempts: {e}"
                )
                raise

            # Calculate backoff delay
            delay = exponential_backoff(attempt, base_delay, max_delay)

            # Log retry attempt
            func_name = getattr(func, '__name__', repr(func))
            logger.warning(
                f"Function {func_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            # Call custom retry callback if provided
            if on_retry:
                on_retry(e, attempt, delay)

            # Wait before retrying
            time.sleep(delay)

        except Exception as e:
            # Non-retryable exception - fail immediately
            func_name = getattr(func, '__name__', repr(func))
            logger.error(f"Function {func_name} failed with non-retryable error: {e}")
            raise

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop exited unexpectedly")


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[Exception, int, float], None] | None = None
) -> Callable[[F], F]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback function(exception, attempt, delay)

    Returns:
        Decorated function with retry logic

    Examples:
        >>> @retry(max_retries=5, base_delay=2.0)
        ... def fetch_data():
        ...     return requests.get("https://api.example.com")

        >>> @retry(
        ...     max_retries=3,
        ...     retryable_exceptions=(ConnectionError, TimeoutError)
        ... )
        ... def upload_file(file_path):
        ...     # Upload logic here
        ...     pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a wrapper function that calls the original with its args
            def call_func() -> Any:
                return func(*args, **kwargs)

            # Use retry_with_backoff on the wrapper
            return retry_with_backoff(
                call_func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
                on_retry=on_retry
            )
        return cast(F, wrapper)
    return decorator


class RetryConfig:
    """
    Configuration for retry behavior.

    Can be used to configure retry settings at application level.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        retryable_exceptions: tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            retryable_exceptions: Tuple of exception types to retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions

    def retry_decorator(
        self,
        on_retry: Callable[[Exception, int, float], None] | None = None
    ) -> Callable[[F], F]:
        """
        Create a retry decorator using this configuration.

        Args:
            on_retry: Optional callback function

        Returns:
            Retry decorator
        """
        return retry(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            retryable_exceptions=self.retryable_exceptions,
            on_retry=on_retry
        )


# Predefined configurations for common use cases

# Default: 3 retries, 1s base delay
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0
)

# Aggressive: More retries for critical operations
AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=120.0
)

# Quick: Fewer retries for fast-fail operations
QUICK_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=10.0
)

# Database-specific: For ChromaDB and other databases
DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=4,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
        # Add database-specific exceptions if needed
    )
)
