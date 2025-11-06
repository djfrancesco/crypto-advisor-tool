"""
Tests for database operations
"""
import pytest
from datetime import datetime, timedelta
import pytz
import tempfile
import os

from database.db_manager import (
    initialize_database,
    get_crypto_id,
    get_last_update_time,
    insert_price_data,
    get_price_history,
    log_refresh_operation,
)


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.duckdb') as f:
        db_path = f.name

    # Initialize database
    initialize_database(db_path)

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


def test_database_initialization(test_db):
    """Test database initialization"""
    # Should not raise any errors
    assert os.path.exists(test_db)


def test_get_crypto_id(test_db):
    """Test retrieving cryptocurrency ID"""
    crypto_id = get_crypto_id('bitcoin', test_db)
    assert crypto_id is not None
    assert isinstance(crypto_id, int)


def test_insert_and_retrieve_price_data(test_db):
    """Test inserting and retrieving price data"""
    crypto_id = get_crypto_id('bitcoin', test_db)

    # Insert test data
    now = datetime.now(pytz.UTC)
    test_data = [
        {
            'timestamp': now - timedelta(hours=2),
            'price': 50000.0,
            'market_cap': 1000000000.0,
            'total_volume': 500000000.0,
        },
        {
            'timestamp': now - timedelta(hours=1),
            'price': 51000.0,
            'market_cap': 1020000000.0,
            'total_volume': 520000000.0,
        },
    ]

    inserted = insert_price_data(crypto_id, test_data, test_db)
    assert inserted == 2

    # Retrieve data
    df = get_price_history(crypto_id, db_path=test_db)
    assert len(df) == 2
    assert df.iloc[0]['price'] == 50000.0
    assert df.iloc[1]['price'] == 51000.0


def test_duplicate_prevention(test_db):
    """Test that duplicate entries are prevented"""
    crypto_id = get_crypto_id('bitcoin', test_db)

    now = datetime.now(pytz.UTC)
    test_data = [
        {
            'timestamp': now,
            'price': 50000.0,
            'market_cap': 1000000000.0,
            'total_volume': 500000000.0,
        },
    ]

    # Insert first time
    inserted1 = insert_price_data(crypto_id, test_data, test_db)
    assert inserted1 == 1

    # Insert same data again (should be ignored)
    inserted2 = insert_price_data(crypto_id, test_data, test_db)
    assert inserted2 == 0  # No new records

    # Verify only one record exists
    df = get_price_history(crypto_id, db_path=test_db)
    assert len(df) == 1


def test_get_last_update_time(test_db):
    """Test retrieving last update time"""
    crypto_id = get_crypto_id('ethereum', test_db)

    # Should be None initially
    last_update = get_last_update_time(crypto_id, test_db)
    assert last_update is None

    # Insert data
    now = datetime.now(pytz.UTC)
    test_data = [{'timestamp': now, 'price': 3000.0}]
    insert_price_data(crypto_id, test_data, test_db)

    # Should return the timestamp
    last_update = get_last_update_time(crypto_id, test_db)
    assert last_update is not None


def test_refresh_log(test_db):
    """Test logging refresh operations"""
    crypto_id = get_crypto_id('bitcoin', test_db)
    start_time = datetime.now(pytz.UTC)

    success = log_refresh_operation(
        operation_type='test',
        start_time=start_time,
        status='success',
        crypto_id=crypto_id,
        records_added=10,
        db_path=test_db,
    )

    assert success is True
