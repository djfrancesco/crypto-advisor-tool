"""
CoinGecko API Client
Handles all API requests to CoinGecko with rate limiting and error handling
"""
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pytz

from config import (
    COINGECKO_BASE_URL,
    COINGECKO_API_KEY,
    API_CALLS_PER_MINUTE,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    API_TIMEOUT,
    CACHE_DURATION,
)
from utils.helpers import RateLimiter, Cache, retry_on_failure

logger = logging.getLogger(__name__)

# Initialize rate limiter and cache
rate_limiter = RateLimiter(API_CALLS_PER_MINUTE, 60)
api_cache = Cache(ttl=CACHE_DURATION)


class CoinGeckoClient:
    """Client for CoinGecko API"""

    def __init__(self, api_key: Optional[str] = COINGECKO_API_KEY):
        self.api_key = api_key
        self.base_url = COINGECKO_BASE_URL
        self.session = requests.Session()

        # Set headers
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        self.session.headers.update(headers)

    @rate_limiter
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to CoinGecko API with rate limiting
        """
        url = f"{self.base_url}/{endpoint}"

        # Check cache
        cache_key = f"{endpoint}:{str(params)}"
        cached_response = api_cache.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_response

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()

            # Cache successful response
            api_cache.set(cache_key, data)

            logger.debug(f"API request successful: {endpoint}")
            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)
                raise
            else:
                logger.error(f"HTTP error: {e}")
                raise

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {endpoint}")
            raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    @retry_on_failure(max_attempts=API_RETRY_ATTEMPTS, delay=API_RETRY_DELAY)
    def get_current_price(self, coin_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for multiple cryptocurrencies

        Args:
            coin_ids: List of CoinGecko coin IDs

        Returns:
            Dict with coin_id as key and price data as value
        """
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }

        data = self._make_request("simple/price", params)

        # Format response
        result = {}
        for coin_id, values in data.items():
            result[coin_id] = {
                "price": values.get("usd", 0),
                "market_cap": values.get("usd_market_cap", 0),
                "total_volume": values.get("usd_24h_vol", 0),
                "price_change_24h": values.get("usd_24h_change", 0),
                "last_updated": datetime.fromtimestamp(
                    values.get("last_updated_at", time.time()),
                    tz=pytz.UTC
                ),
            }

        return result

    @retry_on_failure(max_attempts=API_RETRY_ATTEMPTS, delay=API_RETRY_DELAY)
    def get_historical_data(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[Dict[str, Any]]:
        """
        Get historical market data for a cryptocurrency within a date range

        Args:
            coin_id: CoinGecko coin ID
            from_timestamp: Unix timestamp (start)
            to_timestamp: Unix timestamp (end)

        Returns:
            List of price data points
        """
        params = {
            "vs_currency": "usd",
            "from": from_timestamp,
            "to": to_timestamp,
        }

        data = self._make_request(f"coins/{coin_id}/market_chart/range", params)

        # Parse response into structured format
        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])

        # Create lookup dictionaries for market_cap and volume
        market_cap_dict = {int(item[0]): item[1] for item in market_caps}
        volume_dict = {int(item[0]): item[1] for item in volumes}

        result = []
        for price_data in prices:
            timestamp_ms = int(price_data[0])
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)

            result.append({
                "timestamp": timestamp,
                "price": price_data[1],
                "market_cap": market_cap_dict.get(timestamp_ms),
                "total_volume": volume_dict.get(timestamp_ms),
            })

        logger.info(
            f"Fetched {len(result)} historical data points for {coin_id} "
            f"from {datetime.fromtimestamp(from_timestamp, tz=pytz.UTC)} "
            f"to {datetime.fromtimestamp(to_timestamp, tz=pytz.UTC)}"
        )

        return result

    def get_historical_data_by_days(
        self,
        coin_id: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for the past N days

        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of history

        Returns:
            List of price data points
        """
        now = datetime.now(pytz.UTC)
        to_timestamp = int(now.timestamp())
        from_timestamp = int((now - timedelta(days=days)).timestamp())

        return self.get_historical_data(coin_id, from_timestamp, to_timestamp)

    @retry_on_failure(max_attempts=API_RETRY_ATTEMPTS, delay=API_RETRY_DELAY)
    def get_coin_details(self, coin_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a cryptocurrency

        Args:
            coin_id: CoinGecko coin ID

        Returns:
            Detailed coin information
        """
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        }

        data = self._make_request(f"coins/{coin_id}", params)

        return {
            "id": data.get("id"),
            "symbol": data.get("symbol", "").upper(),
            "name": data.get("name"),
            "description": data.get("description", {}).get("en", ""),
            "market_cap_rank": data.get("market_cap_rank"),
            "current_price": data.get("market_data", {}).get("current_price", {}).get("usd"),
            "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
            "total_volume": data.get("market_data", {}).get("total_volume", {}).get("usd"),
            "circulating_supply": data.get("market_data", {}).get("circulating_supply"),
            "total_supply": data.get("market_data", {}).get("total_supply"),
        }

    def ping(self) -> bool:
        """
        Test API connectivity

        Returns:
            True if API is reachable
        """
        try:
            self._make_request("ping")
            return True
        except Exception as e:
            logger.error(f"API ping failed: {e}")
            return False

    def clear_cache(self):
        """Clear API response cache"""
        api_cache.clear()
        logger.info("API cache cleared")


# Global client instance
_client = None


def get_client() -> CoinGeckoClient:
    """Get or create global CoinGecko client instance"""
    global _client
    if _client is None:
        _client = CoinGeckoClient()
    return _client
