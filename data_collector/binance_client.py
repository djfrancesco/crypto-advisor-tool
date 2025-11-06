"""
Binance API Client
Direct access to Binance exchange data with better rate limits than CoinGecko
No API key required for public market data
"""
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz
import time

logger = logging.getLogger(__name__)


class BinanceClient:
    """Client for Binance public API"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[List]:
        """
        Get candlestick/kline data from Binance

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 1h, 1d, etc.)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles to return (max 1000)

        Returns:
            List of klines [open_time, open, high, low, close, volume, ...]
        """
        endpoint = f"{self.base_url}/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Fetched {len(data)} klines for {symbol}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return []

    def get_historical_data(
        self,
        symbol: str,
        from_timestamp: int,
        to_timestamp: int,
        interval: str = "1h"
    ) -> List[Dict]:
        """
        Get historical price data for a symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            from_timestamp: Unix timestamp (seconds)
            to_timestamp: Unix timestamp (seconds)
            interval: Timeframe (1m, 5m, 1h, 1d)

        Returns:
            List of price data dicts
        """
        # Convert to milliseconds for Binance API
        start_ms = from_timestamp * 1000
        end_ms = to_timestamp * 1000

        all_data = []
        current_start = start_ms

        # Binance returns max 1000 candles per request
        # Calculate interval in milliseconds
        interval_ms = self._interval_to_milliseconds(interval)
        max_range = 1000 * interval_ms

        while current_start < end_ms:
            current_end = min(current_start + max_range, end_ms)

            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )

            if not klines:
                break

            # Parse klines to our format
            for kline in klines:
                timestamp = datetime.fromtimestamp(kline[0] / 1000, tz=pytz.UTC)
                close_price = float(kline[4])
                volume = float(kline[5])
                quote_volume = float(kline[7])  # Volume in USDT

                all_data.append({
                    'timestamp': timestamp,
                    'price': close_price,
                    'market_cap': None,  # Not available from exchange
                    'total_volume': quote_volume,
                })

            # Move to next batch
            if klines:
                current_start = klines[-1][0] + interval_ms
            else:
                break

            # Avoid rate limiting
            time.sleep(0.1)

        logger.info(f"Fetched {len(all_data)} data points for {symbol}")
        return all_data

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price for a symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Dict with price data or None
        """
        endpoint = f"{self.base_url}/ticker/24hr"

        params = {"symbol": symbol}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            return {
                'price': float(data['lastPrice']),
                'market_cap': None,
                'total_volume': float(data['quoteVolume']),
                'price_change_24h': float(data['priceChangePercent']),
                'last_updated': datetime.fromtimestamp(data['closeTime'] / 1000, tz=pytz.UTC),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return None

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple symbols

        Args:
            symbols: List of trading pairs

        Returns:
            Dict with symbol as key and price data as value
        """
        result = {}

        for symbol in symbols:
            price_data = self.get_current_price(symbol)
            if price_data:
                result[symbol] = price_data
            time.sleep(0.1)  # Avoid rate limiting

        return result

    def ping(self) -> bool:
        """Test API connectivity"""
        endpoint = f"{self.base_url}/ping"

        try:
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Binance API ping failed: {e}")
            return False

    def _interval_to_milliseconds(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        intervals = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }

        return intervals.get(interval, 60 * 60 * 1000)  # Default to 1h


# Coin ID to Binance symbol mapping
COIN_TO_SYMBOL = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'binancecoin': 'BNBUSDT',
    'ripple': 'XRPUSDT',
    'cardano': 'ADAUSDT',
}


def get_binance_symbol(coin_id: str) -> Optional[str]:
    """Convert CoinGecko coin ID to Binance symbol"""
    return COIN_TO_SYMBOL.get(coin_id)


# Global client instance
_client = None


def get_binance_client() -> BinanceClient:
    """Get or create global Binance client instance"""
    global _client
    if _client is None:
        _client = BinanceClient()
    return _client
