"""
Smart Data Refresher for Crypto Advisor Tool
THE HEART OF THE SYSTEM - Handles intelligent incremental data updates
Fetches ONLY new data, never duplicates, fills gaps automatically
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

from config import CRYPTOCURRENCIES, INITIAL_HISTORY_DAYS, DATA_SOURCE
from database.db_manager import (
    get_crypto_id,
    get_last_update_time,
    get_data_range,
    get_missing_data_ranges,
    insert_price_data,
    log_refresh_operation,
)
from data_collector.api_client import get_client as get_coingecko_client
from data_collector.binance_client import get_binance_client, get_binance_symbol
from utils.helpers import validate_price_data, performance_monitor

logger = logging.getLogger(__name__)


class DataRefresher:
    """Manages smart data refresh operations"""

    def __init__(self, data_source: str = DATA_SOURCE):
        """
        Initialize data refresher with specified data source

        Args:
            data_source: "coingecko", "binance", or "auto"
        """
        self.data_source = data_source
        self.coingecko_client = get_coingecko_client()
        self.binance_client = get_binance_client()

        logger.info(f"DataRefresher initialized with source: {data_source}")

    @performance_monitor
    def smart_refresh(self, coin_id: str) -> Dict[str, any]:
        """
        SMART REFRESH MECHANISM - The core innovation

        This function:
        1. Checks database for last update timestamp
        2. If no data exists, fetches initial historical data (365 days)
        3. If data exists, fetches ONLY new data since last update
        4. Identifies and fills gaps in historical data
        5. Prevents duplicates using database constraints
        6. Logs all operations for debugging

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')

        Returns:
            Dict with refresh statistics
        """
        start_time = datetime.now(pytz.UTC)
        crypto_id = get_crypto_id(coin_id)

        if not crypto_id:
            logger.error(f"Unknown cryptocurrency: {coin_id}")
            return {"status": "failed", "error": "Unknown cryptocurrency"}

        logger.info(f"Starting smart refresh for {coin_id}")

        try:
            # Step 1: Check when we last updated this coin
            last_update = get_last_update_time(crypto_id)

            if last_update is None:
                # First run - fetch initial historical data
                logger.info(f"First run for {coin_id}, fetching {INITIAL_HISTORY_DAYS} days of history")
                result = self._initial_load(crypto_id, coin_id)
            else:
                # We have data - fetch only new data since last update
                logger.info(f"Last update for {coin_id}: {last_update}")
                result = self._incremental_refresh(crypto_id, coin_id, last_update)

            # Step 2: Check for and fill gaps
            gaps_filled = self._fill_data_gaps(crypto_id, coin_id)
            result['gaps_filled'] = gaps_filled

            # Step 3: Log successful operation
            log_refresh_operation(
                operation_type=result['operation_type'],
                start_time=start_time,
                status='success',
                crypto_id=crypto_id,
                records_added=result['records_added'],
                from_timestamp=result.get('from_timestamp'),
                to_timestamp=result.get('to_timestamp'),
            )

            logger.info(
                f"Smart refresh completed for {coin_id}: "
                f"{result['records_added']} new records, "
                f"{gaps_filled} gaps filled"
            )

            result['status'] = 'success'
            return result

        except Exception as e:
            logger.error(f"Smart refresh failed for {coin_id}: {e}", exc_info=True)

            # Log failure
            log_refresh_operation(
                operation_type='refresh',
                start_time=start_time,
                status='failed',
                crypto_id=crypto_id,
                error_message=str(e),
            )

            return {
                'status': 'failed',
                'error': str(e),
                'records_added': 0,
            }

    def _fetch_historical_data(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[Dict]:
        """
        Fetch historical data from configured data source

        Args:
            coin_id: CoinGecko coin ID
            from_timestamp: Unix timestamp (seconds)
            to_timestamp: Unix timestamp (seconds)

        Returns:
            List of price data dicts
        """
        if self.data_source == "binance":
            return self._fetch_from_binance(coin_id, from_timestamp, to_timestamp)
        elif self.data_source == "coingecko":
            return self._fetch_from_coingecko(coin_id, from_timestamp, to_timestamp)
        elif self.data_source == "auto":
            # Try Binance first, fallback to CoinGecko
            try:
                return self._fetch_from_binance(coin_id, from_timestamp, to_timestamp)
            except Exception as e:
                logger.warning(f"Binance fetch failed, falling back to CoinGecko: {e}")
                return self._fetch_from_coingecko(coin_id, from_timestamp, to_timestamp)
        else:
            logger.warning(f"Unknown data source: {self.data_source}, using CoinGecko")
            return self._fetch_from_coingecko(coin_id, from_timestamp, to_timestamp)

    def _fetch_from_binance(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[Dict]:
        """Fetch data from Binance API"""
        symbol = get_binance_symbol(coin_id)
        if not symbol:
            raise ValueError(f"No Binance symbol mapping for {coin_id}")

        logger.debug(f"Fetching from Binance: {symbol}")
        return self.binance_client.get_historical_data(
            symbol,
            from_timestamp,
            to_timestamp,
            interval="1h"
        )

    def _fetch_from_coingecko(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[Dict]:
        """Fetch data from CoinGecko API"""
        logger.debug(f"Fetching from CoinGecko: {coin_id}")
        return self.coingecko_client.get_historical_data(
            coin_id,
            from_timestamp,
            to_timestamp
        )

    def _initial_load(self, crypto_id: int, coin_id: str) -> Dict:
        """
        Initial data load for first run
        Fetches INITIAL_HISTORY_DAYS of historical data
        """
        logger.info(f"Performing initial load for {coin_id}")

        now = datetime.now(pytz.UTC)
        from_date = now - timedelta(days=INITIAL_HISTORY_DAYS)
        to_date = now

        # Fetch historical data from configured source
        price_data = self._fetch_historical_data(
            coin_id,
            int(from_date.timestamp()),
            int(to_date.timestamp())
        )

        # Validate and insert data
        valid_data = [d for d in price_data if validate_price_data(d)]

        if len(valid_data) < len(price_data):
            logger.warning(
                f"Filtered out {len(price_data) - len(valid_data)} invalid records"
            )

        records_added = insert_price_data(crypto_id, valid_data)

        return {
            'operation_type': 'initial_load',
            'records_added': records_added,
            'from_timestamp': from_date,
            'to_timestamp': to_date,
        }

    def _incremental_refresh(
        self,
        crypto_id: int,
        coin_id: str,
        last_update: datetime
    ) -> Dict:
        """
        Incremental refresh - fetch ONLY new data since last update
        This is the key to avoiding duplicates!
        """
        logger.info(f"Performing incremental refresh for {coin_id} since {last_update}")

        now = datetime.now(pytz.UTC)

        # Ensure last_update is timezone-aware
        if last_update.tzinfo is None:
            last_update = pytz.UTC.localize(last_update)

        # Calculate time since last update
        time_delta = now - last_update
        logger.info(f"Time since last update: {time_delta}")

        # If last update was very recent (< 5 minutes), skip
        if time_delta < timedelta(minutes=5):
            logger.info(f"Last update was {time_delta.total_seconds():.0f}s ago, skipping refresh")
            return {
                'operation_type': 'skip',
                'records_added': 0,
                'message': 'Too recent to refresh',
            }

        # Fetch only NEW data from last_update to now
        # Add small buffer to ensure we don't miss data due to timing
        from_timestamp = int((last_update - timedelta(minutes=5)).timestamp())
        to_timestamp = int(now.timestamp())

        price_data = self._fetch_historical_data(
            coin_id,
            from_timestamp,
            to_timestamp
        )

        # Validate data
        valid_data = [d for d in price_data if validate_price_data(d)]

        # Insert with duplicate prevention (ON CONFLICT DO NOTHING)
        records_added = insert_price_data(crypto_id, valid_data)

        logger.info(
            f"Fetched {len(price_data)} records, "
            f"{len(valid_data)} valid, "
            f"{records_added} new (rest were duplicates or already existed)"
        )

        return {
            'operation_type': 'refresh',
            'records_added': records_added,
            'from_timestamp': last_update,
            'to_timestamp': now,
        }

    def _fill_data_gaps(self, crypto_id: int, coin_id: str) -> int:
        """
        Identify and fill gaps in historical data
        This ensures data completeness even if connection was lost
        """
        # Get overall data range
        earliest, latest = get_data_range(crypto_id)

        if not earliest or not latest:
            return 0

        # Look for gaps (missing data between records)
        gaps = get_missing_data_ranges(
            crypto_id,
            earliest,
            latest,
            expected_interval_hours=1
        )

        if not gaps:
            logger.info(f"No data gaps found for {coin_id}")
            return 0

        logger.info(f"Found {len(gaps)} data gaps for {coin_id}")

        total_filled = 0

        for gap_start, gap_end in gaps:
            # Only fill gaps larger than 2 hours to avoid false positives
            gap_duration = gap_end - gap_start
            if gap_duration < timedelta(hours=2):
                continue

            logger.info(f"Filling gap from {gap_start} to {gap_end} ({gap_duration})")

            try:
                # Ensure timestamps are timezone-aware
                if gap_start.tzinfo is None:
                    gap_start = pytz.UTC.localize(gap_start)
                if gap_end.tzinfo is None:
                    gap_end = pytz.UTC.localize(gap_end)

                price_data = self._fetch_historical_data(
                    coin_id,
                    int(gap_start.timestamp()),
                    int(gap_end.timestamp())
                )

                valid_data = [d for d in price_data if validate_price_data(d)]
                filled = insert_price_data(crypto_id, valid_data)
                total_filled += filled

                logger.info(f"Filled gap with {filled} records")

                # Log gap fill operation
                log_refresh_operation(
                    operation_type='gap_fill',
                    start_time=datetime.now(pytz.UTC),
                    status='success',
                    crypto_id=crypto_id,
                    records_added=filled,
                    from_timestamp=gap_start,
                    to_timestamp=gap_end,
                )

            except Exception as e:
                logger.error(f"Failed to fill gap: {e}")
                continue

        return total_filled

    def refresh_all(self) -> Dict[str, Dict]:
        """
        Refresh data for all tracked cryptocurrencies

        Returns:
            Dict with coin_id as key and refresh result as value
        """
        logger.info("Starting refresh for all cryptocurrencies")
        results = {}

        for crypto in CRYPTOCURRENCIES:
            coin_id = crypto['id']
            logger.info(f"Refreshing {coin_id}...")

            try:
                result = self.smart_refresh(coin_id)
                results[coin_id] = result
            except Exception as e:
                logger.error(f"Failed to refresh {coin_id}: {e}")
                results[coin_id] = {
                    'status': 'failed',
                    'error': str(e),
                }

        # Summary
        total_records = sum(r.get('records_added', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = len(results) - successful

        logger.info(
            f"Refresh completed: {successful} successful, {failed} failed, "
            f"{total_records} total new records"
        )

        return results

    def get_current_prices(self) -> Dict[str, Dict]:
        """
        Get current prices for all tracked cryptocurrencies
        This is useful for dashboard display without waiting for full refresh
        """
        coin_ids = [crypto['id'] for crypto in CRYPTOCURRENCIES]

        try:
            prices = self.client.get_current_prices(coin_ids)
            return prices
        except Exception as e:
            logger.error(f"Failed to get current prices: {e}")
            return {}


# Global refresher instance
_refresher = None


def get_refresher() -> DataRefresher:
    """Get or create global DataRefresher instance"""
    global _refresher
    if _refresher is None:
        _refresher = DataRefresher()
    return _refresher
