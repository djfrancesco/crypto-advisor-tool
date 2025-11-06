"""
Bulk Historical Data Loader
Downloads large historical datasets from CryptoDataDownload.com
Bypasses rate limits for fast initial data loading
"""
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
import pytz

from config import CRYPTOCURRENCIES
from database.db_manager import get_crypto_id, insert_price_data

logger = logging.getLogger(__name__)


class BulkDataLoader:
    """Load bulk historical data from CSV sources"""

    def __init__(self):
        self.base_url = "https://www.cryptodatadownload.com/cdd"
        self.exchange = "Binance"  # Using Binance exchange data

    def download_csv(self, symbol: str, interval: str = "d") -> pd.DataFrame:
        """
        Download CSV from CryptoDataDownload

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Time interval (d=daily, h=hourly, 1m=minute)

        Returns:
            DataFrame with OHLCV data
        """
        # Build URL
        filename = f"{self.exchange}_{symbol}_{interval}.csv"
        url = f"{self.base_url}/{filename}"

        logger.info(f"Downloading {filename} from {url}")

        try:
            # Download CSV
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV (skip first row which is metadata)
            from io import StringIO
            csv_data = StringIO(response.text)

            # Read CSV, skipping first row
            df = pd.read_csv(csv_data, skiprows=1)

            logger.info(f"Downloaded {len(df)} records for {symbol}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            return pd.DataFrame()

    def parse_cryptodatadownload_csv(self, df: pd.DataFrame) -> List[Dict]:
        """
        Parse CryptoDataDownload CSV format to our database format

        CSV columns: Unix, Date, Symbol, Open, High, Low, Close, Volume BTC, Volume USDT, tradecount
        """
        if df.empty:
            return []

        parsed_data = []

        for _, row in df.iterrows():
            try:
                # Try both capitalized and lowercase column names
                unix_val = row.get('Unix') or row.get('unix')
                close_val = row.get('Close') or row.get('close')
                volume_val = row.get('Volume USDT') or row.get('volume')

                if pd.isna(unix_val) or pd.isna(close_val):
                    continue

                # Parse timestamp (milliseconds to seconds)
                timestamp = datetime.fromtimestamp(int(unix_val) / 1000, tz=pytz.UTC)

                # Use close price as our price
                price = float(close_val)

                # Volume in USDT (if available)
                volume = float(volume_val) if pd.notna(volume_val) else None

                parsed_data.append({
                    'timestamp': timestamp,
                    'price': price,
                    'market_cap': None,  # Not available in CSV
                    'total_volume': volume,
                })

            except (ValueError, TypeError, KeyError):
                # Skip invalid rows silently
                continue
            except Exception as e:
                logger.debug(f"Failed to parse row: {e}")
                continue

        logger.info(f"Parsed {len(parsed_data)} records from CSV")
        return parsed_data

    def load_coin_history(self, coin_id: str, limit_days: int = None) -> int:
        """
        Load historical data for a cryptocurrency

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            limit_days: Limit to most recent N days (optional)

        Returns:
            Number of records inserted
        """
        # Map coin_id to trading symbol
        symbol_map = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'binancecoin': 'BNBUSDT',
            'ripple': 'XRPUSDT',
            'cardano': 'ADAUSDT',
        }

        symbol = symbol_map.get(coin_id)
        if not symbol:
            logger.error(f"Unknown coin_id: {coin_id}")
            return 0

        # Get crypto_id from database
        crypto_id = get_crypto_id(coin_id)
        if not crypto_id:
            logger.error(f"Crypto ID not found for {coin_id}")
            return 0

        # Download CSV
        df = self.download_csv(symbol, interval='d')  # Daily data

        if df.empty:
            logger.warning(f"No data downloaded for {coin_id}")
            return 0

        # Limit to recent days if specified
        if limit_days:
            df = df.head(limit_days)

        # Parse to our format
        price_data = self.parse_cryptodatadownload_csv(df)

        # Insert into database
        inserted = insert_price_data(crypto_id, price_data)

        logger.info(f"Loaded {inserted} records for {coin_id}")
        return inserted

    def load_all(self, limit_days: int = None) -> Dict[str, int]:
        """
        Load historical data for all tracked cryptocurrencies

        Args:
            limit_days: Limit to most recent N days (optional)

        Returns:
            Dict with coin_id as key and number of inserted records as value
        """
        logger.info("Starting bulk data load for all cryptocurrencies")
        results = {}

        for crypto in CRYPTOCURRENCIES:
            coin_id = crypto['id']
            logger.info(f"Loading {coin_id}...")

            try:
                inserted = self.load_coin_history(coin_id, limit_days)
                results[coin_id] = inserted
            except Exception as e:
                logger.error(f"Failed to load {coin_id}: {e}")
                results[coin_id] = 0

        total = sum(results.values())
        logger.info(f"Bulk load completed: {total} total records inserted")

        return results


def main():
    """CLI entry point for bulk loading"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Bulk load historical cryptocurrency data')
    parser.add_argument(
        '--coin',
        type=str,
        help='Coin ID to load (e.g., bitcoin). If not specified, loads all coins.'
    )
    parser.add_argument(
        '--days',
        type=int,
        help='Limit to most recent N days'
    )
    parser.add_argument(
        '--load-all',
        action='store_true',
        help='Load all tracked cryptocurrencies'
    )

    args = parser.parse_args()

    loader = BulkDataLoader()

    if args.load_all or args.coin is None:
        # Load all coins
        results = loader.load_all(limit_days=args.days)

        print("\n" + "="*60)
        print("BULK DATA LOAD RESULTS")
        print("="*60)
        for coin_id, count in results.items():
            print(f"{coin_id:15s}: {count:6d} records")
        print("="*60)
        print(f"TOTAL: {sum(results.values())} records")
        print("="*60)

    else:
        # Load specific coin
        count = loader.load_coin_history(args.coin, limit_days=args.days)
        print(f"\nLoaded {count} records for {args.coin}")


if __name__ == "__main__":
    main()
