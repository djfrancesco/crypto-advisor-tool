#!/usr/bin/env python
"""
Load hourly data from Binance API for all cryptocurrencies
Better than CSV for granular data
"""
import logging
from datetime import datetime, timedelta
import pytz

from config import CRYPTOCURRENCIES
from database.db_manager import get_crypto_id, insert_price_data
from data_collector.binance_client import get_binance_client, get_binance_symbol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_hourly_data(days=90):
    """Load hourly data from Binance for all coins"""
    client = get_binance_client()
    
    now = datetime.now(pytz.UTC)
    from_time = now - timedelta(days=days)
    
    print(f"\n{'='*60}")
    print(f"LOADING {days} DAYS OF HOURLY DATA FROM BINANCE")
    print(f"{'='*60}\n")
    
    results = {}
    
    for crypto in CRYPTOCURRENCIES:
        coin_id = crypto['id']
        symbol = get_binance_symbol(coin_id)
        crypto_id = get_crypto_id(coin_id)
        
        if not symbol or not crypto_id:
            logger.warning(f"Skipping {coin_id}")
            continue
        
        logger.info(f"Fetching {coin_id} ({symbol})...")
        
        try:
            data = client.get_historical_data(
                symbol,
                int(from_time.timestamp()),
                int(now.timestamp()),
                interval='1h'
            )
            
            inserted = insert_price_data(crypto_id, data)
            results[coin_id] = inserted
            logger.info(f"✓ {coin_id}: {inserted} records")
            
        except Exception as e:
            logger.error(f"✗ {coin_id}: {e}")
            results[coin_id] = 0
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for coin_id, count in results.items():
        print(f"{coin_id:15s}: {count:6d} hourly records")
    print(f"{'='*60}")
    print(f"TOTAL: {sum(results.values())} records ({days} days × 24 hours × 5 coins)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    load_hourly_data(days)
