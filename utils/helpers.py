"""
Helper utilities for Crypto Advisor Tool
Logging, validation, error handling, and performance monitoring
"""
import logging
import time
import functools
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import pytz

from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    TIMEZONE,
    PRICE_VALIDATION,
    ERROR_RECOVERY,
)

# Configure logging
def setup_logging(name: str = __name__, level: str = LOG_LEVEL) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging(__name__)


def get_utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(pytz.UTC)


def convert_to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC"""
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(pytz.UTC)


def validate_price_data(data: Dict[str, Any]) -> bool:
    """
    Validate price data against configured thresholds
    Returns True if valid, False otherwise
    """
    try:
        price = data.get('price')
        if not price:
            logger.warning("Price data missing 'price' field")
            return False

        # Check price range
        if price < PRICE_VALIDATION['min_price']:
            logger.warning(f"Price {price} below minimum threshold")
            return False

        if price > PRICE_VALIDATION['max_price']:
            logger.warning(f"Price {price} above maximum threshold")
            return False

        # Check timestamp
        if 'timestamp' not in data:
            logger.warning("Price data missing 'timestamp' field")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating price data: {e}")
        return False


def validate_price_change(
    old_price: float,
    new_price: float,
    time_delta_hours: float
) -> bool:
    """
    Validate that price change is within reasonable bounds
    """
    if old_price <= 0:
        return False

    percent_change = abs((new_price - old_price) / old_price * 100)
    max_change = PRICE_VALIDATION['max_price_change_percent'] * time_delta_hours

    if percent_change > max_change:
        logger.warning(
            f"Price change {percent_change:.2f}% exceeds max {max_change:.2f}% "
            f"for {time_delta_hours}h period"
        )
        return False

    return True


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance
    Logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.2f}s: {e}",
                exc_info=True
            )
            raise

    return wrapper


def retry_on_failure(
    max_attempts: int = ERROR_RECOVERY['max_consecutive_failures'],
    delay: float = ERROR_RECOVERY['recovery_wait_time'],
    backoff: float = 2.0
) -> Callable:
    """
    Decorator to retry function on failure with exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}"
                    )

                    if attempt < max_attempts:
                        logger.info(f"Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts"
                        )

            raise last_exception

        return wrapper
    return decorator


def format_currency(value: float, symbol: str = "$") -> str:
    """Format currency value"""
    if value >= 1_000_000:
        return f"{symbol}{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{symbol}{value/1_000:.2f}K"
    else:
        return f"{symbol}{value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage with sign"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls: int, time_window: float):
        """
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove calls outside time window
            self.calls = [call_time for call_time in self.calls
                         if now - call_time < self.time_window]

            # Check if we can make another call
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    self.calls = []

            # Make the call
            result = func(*args, **kwargs)
            self.calls.append(time.time())
            return result

        return wrapper
    return self


class Cache:
    """Simple in-memory cache with TTL"""

    def __init__(self, ttl: int = 60):
        """
        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self):
        """Clear all cached values"""
        self.cache.clear()
        self.timestamps.clear()

    def size(self) -> int:
        """Get number of cached items"""
        return len(self.cache)
