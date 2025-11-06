"""
Database Manager for Crypto Advisor Tool
Handles all database operations with DuckDB
"""
import duckdb
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import pandas as pd

from config import DATABASE_PATH, CRYPTOCURRENCIES

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish database connection"""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        return self._connection

    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get connection, creating if necessary"""
        if self._connection is None:
            self.connect()
        return self._connection


def get_connection(db_path: str = DATABASE_PATH) -> duckdb.DuckDBPyConnection:
    """Get a database connection"""
    return duckdb.connect(db_path)


def initialize_database(db_path: str = DATABASE_PATH) -> bool:
    """
    Initialize database with schema
    Returns True if successful
    """
    try:
        conn = get_connection(db_path)

        # Read and execute schema SQL
        schema_path = Path(__file__).parent / "schemas.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Execute schema (DuckDB handles multiple statements)
        conn.execute(schema_sql)

        # Insert cryptocurrency metadata
        for crypto in CRYPTOCURRENCIES:
            conn.execute("""
                INSERT INTO cryptocurrencies (coin_id, symbol, name)
                VALUES (?, ?, ?)
                ON CONFLICT (coin_id) DO NOTHING
            """, [crypto['id'], crypto['symbol'], crypto['name']])

        conn.commit()
        conn.close()

        logger.info("Database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def get_crypto_id(coin_id: str, db_path: str = DATABASE_PATH) -> Optional[int]:
    """Get internal crypto_id from coin_id"""
    conn = get_connection(db_path)
    result = conn.execute(
        "SELECT id FROM cryptocurrencies WHERE coin_id = ?",
        [coin_id]
    ).fetchone()
    conn.close()
    return result[0] if result else None


def get_last_update_time(crypto_id: int, db_path: str = DATABASE_PATH) -> Optional[datetime]:
    """
    Get the last update timestamp for a cryptocurrency
    Critical for smart refresh logic
    """
    conn = get_connection(db_path)
    result = conn.execute("""
        SELECT MAX(timestamp) as last_update
        FROM price_history
        WHERE crypto_id = ?
    """, [crypto_id]).fetchone()
    conn.close()

    if result and result[0]:
        return result[0]
    return None


def get_data_range(crypto_id: int, db_path: str = DATABASE_PATH) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the date range of available data for a cryptocurrency"""
    conn = get_connection(db_path)
    result = conn.execute("""
        SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
        FROM price_history
        WHERE crypto_id = ?
    """, [crypto_id]).fetchone()
    conn.close()

    return (result[0], result[1]) if result else (None, None)


def get_missing_data_ranges(
    crypto_id: int,
    start_date: datetime,
    end_date: datetime,
    expected_interval_hours: int = 1,
    db_path: str = DATABASE_PATH
) -> List[Tuple[datetime, datetime]]:
    """
    Identify gaps in data where we're missing records
    Returns list of (start, end) tuples for missing ranges
    """
    conn = get_connection(db_path)

    # Get all timestamps in range
    result = conn.execute("""
        SELECT timestamp
        FROM price_history
        WHERE crypto_id = ?
        AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    """, [crypto_id, start_date, end_date]).fetchall()

    conn.close()

    if not result:
        return [(start_date, end_date)]

    timestamps = [row[0] for row in result]
    gaps = []
    expected_delta = timedelta(hours=expected_interval_hours)

    # Check for gaps larger than expected interval
    for i in range(len(timestamps) - 1):
        current = timestamps[i]
        next_time = timestamps[i + 1]
        if next_time - current > expected_delta * 2:  # Allow some tolerance
            gaps.append((current, next_time))

    return gaps


def insert_price_data(
    crypto_id: int,
    price_data: List[Dict[str, Any]],
    db_path: str = DATABASE_PATH
) -> int:
    """
    Insert price data with duplicate prevention
    Returns number of records inserted
    """
    if not price_data:
        return 0

    conn = get_connection(db_path)
    inserted = 0

    try:
        for data in price_data:
            try:
                conn.execute("""
                    INSERT INTO price_history (crypto_id, timestamp, price, market_cap, total_volume)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (crypto_id, timestamp) DO NOTHING
                """, [
                    crypto_id,
                    data['timestamp'],
                    data['price'],
                    data.get('market_cap'),
                    data.get('total_volume')
                ])
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert record at {data['timestamp']}: {e}")
                continue

        conn.commit()
        logger.info(f"Inserted {inserted} new price records for crypto_id {crypto_id}")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to insert price data: {e}")
    finally:
        conn.close()

    return inserted


def get_price_history(
    crypto_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db_path: str = DATABASE_PATH
) -> pd.DataFrame:
    """Get price history as DataFrame"""
    conn = get_connection(db_path)

    query = "SELECT timestamp, price, market_cap, total_volume FROM price_history WHERE crypto_id = ?"
    params = [crypto_id]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)

    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)

    query += " ORDER BY timestamp"

    df = conn.execute(query, params).df()
    conn.close()

    return df


def update_technical_indicators(
    crypto_id: int,
    indicators_data: List[Dict[str, Any]],
    db_path: str = DATABASE_PATH
) -> int:
    """Update technical indicators using batch insert for better performance"""
    if not indicators_data:
        return 0

    conn = get_connection(db_path)

    try:
        # Prepare batch data
        batch_data = [
            [
                crypto_id,
                data['timestamp'],
                data.get('ma_short'),
                data.get('ma_long'),
                data.get('rsi'),
                data.get('macd'),
                data.get('macd_signal'),
                data.get('macd_histogram'),
                data.get('bb_upper'),
                data.get('bb_middle'),
                data.get('bb_lower'),
                data.get('support_level'),
                data.get('resistance_level')
            ]
            for data in indicators_data
        ]

        # Use executemany for batch insert (much faster)
        conn.executemany("""
            INSERT INTO technical_indicators (
                crypto_id, timestamp, ma_short, ma_long, rsi,
                macd, macd_signal, macd_histogram,
                bb_upper, bb_middle, bb_lower,
                support_level, resistance_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (crypto_id, timestamp) DO UPDATE SET
                ma_short = EXCLUDED.ma_short,
                ma_long = EXCLUDED.ma_long,
                rsi = EXCLUDED.rsi,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                macd_histogram = EXCLUDED.macd_histogram,
                bb_upper = EXCLUDED.bb_upper,
                bb_middle = EXCLUDED.bb_middle,
                bb_lower = EXCLUDED.bb_lower,
                support_level = EXCLUDED.support_level,
                resistance_level = EXCLUDED.resistance_level
        """, batch_data)

        conn.commit()
        updated = len(indicators_data)
        logger.info(f"Updated {updated} technical indicator records")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update indicators: {e}")
        updated = 0
    finally:
        conn.close()

    return updated


def save_prediction(
    crypto_id: int,
    prediction_date: datetime,
    target_date: datetime,
    signal: str,
    confidence: float,
    predicted_price: Optional[float] = None,
    model_version: str = "v1",
    features_used: Optional[str] = None,
    db_path: str = DATABASE_PATH
) -> bool:
    """Save ML prediction"""
    conn = get_connection(db_path)

    try:
        conn.execute("""
            INSERT INTO predictions (
                crypto_id, prediction_date, target_date, signal,
                confidence, predicted_price, model_version, features_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            crypto_id, prediction_date, target_date, signal,
            confidence, predicted_price, model_version, features_used
        ])

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        conn.close()
        return False


def log_refresh_operation(
    operation_type: str,
    start_time: datetime,
    status: str,
    crypto_id: Optional[int] = None,
    records_added: int = 0,
    records_updated: int = 0,
    error_message: Optional[str] = None,
    from_timestamp: Optional[datetime] = None,
    to_timestamp: Optional[datetime] = None,
    db_path: str = DATABASE_PATH
) -> bool:
    """Log a data refresh operation"""
    conn = get_connection(db_path)

    try:
        conn.execute("""
            INSERT INTO refresh_log (
                crypto_id, operation_type, start_time, end_time,
                records_added, records_updated, status, error_message,
                from_timestamp, to_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            crypto_id, operation_type, start_time, datetime.utcnow(),
            records_added, records_updated, status, error_message,
            from_timestamp, to_timestamp
        ])

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to log refresh operation: {e}")
        conn.close()
        return False


def get_latest_predictions(
    crypto_id: int,
    limit: int = 10,
    db_path: str = DATABASE_PATH
) -> pd.DataFrame:
    """Get latest predictions for a cryptocurrency"""
    conn = get_connection(db_path)

    df = conn.execute("""
        SELECT prediction_date, target_date, signal, confidence, predicted_price
        FROM predictions
        WHERE crypto_id = ?
        ORDER BY prediction_date DESC
        LIMIT ?
    """, [crypto_id, limit]).df()

    conn.close()
    return df


def get_database_stats(db_path: str = DATABASE_PATH) -> Dict[str, Any]:
    """Get database statistics"""
    conn = get_connection(db_path)

    stats = {}

    # Total records per cryptocurrency
    result = conn.execute("""
        SELECT c.symbol, COUNT(p.id) as record_count,
               MIN(p.timestamp) as earliest_data,
               MAX(p.timestamp) as latest_data
        FROM cryptocurrencies c
        LEFT JOIN price_history p ON c.id = p.crypto_id
        GROUP BY c.id, c.symbol
    """).fetchall()

    stats['crypto_stats'] = [
        {
            'symbol': row[0],
            'records': row[1],
            'earliest': row[2],
            'latest': row[3]
        }
        for row in result
    ]

    # Recent refresh operations
    result = conn.execute("""
        SELECT operation_type, status, COUNT(*) as count
        FROM refresh_log
        WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
        GROUP BY operation_type, status
    """).fetchall()

    stats['recent_refreshes'] = [
        {'type': row[0], 'status': row[1], 'count': row[2]}
        for row in result
    ]

    conn.close()
    return stats
