"""
Configuration file for Crypto Advisor Tool
Contains all application settings and constants
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "crypto_data.duckdb"))

# CoinGecko API Configuration
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")

# Rate Limiting (CoinGecko free tier: 10-30 calls/minute)
API_CALLS_PER_MINUTE = 25
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2  # seconds
API_TIMEOUT = 30  # seconds
CACHE_DURATION = 60  # seconds

# Cryptocurrencies to Track
CRYPTOCURRENCIES = [
    {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    {"id": "binancecoin", "symbol": "BNB", "name": "Binance Coin"},
    {"id": "ripple", "symbol": "XRP", "name": "XRP"},
    {"id": "cardano", "symbol": "ADA", "name": "Cardano"},
]

# Data Collection Settings
REFRESH_INTERVAL_MINUTES = int(os.getenv("REFRESH_INTERVAL_MINUTES", "15"))
INITIAL_HISTORY_DAYS = 365  # Days of historical data to fetch on first run
MIN_DATA_POINTS = 30  # Minimum data points required for analysis

# Technical Indicators Parameters
TECHNICAL_INDICATORS = {
    "moving_averages": {
        "short_period": 7,
        "long_period": 30,
    },
    "rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    },
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2,
    },
}

# Machine Learning Settings
ML_CONFIG = {
    "prediction_days": 3,  # Days ahead to predict
    "training_window_days": 90,  # Days of historical data for training
    "min_training_samples": 60,
    "test_split_ratio": 0.2,
    "random_state": 42,
    "n_estimators": 100,  # Random Forest trees
    "max_depth": 10,
    "min_samples_split": 5,
}

# Feature Engineering
FEATURE_PERIODS = [1, 3, 7, 30]  # Days for price change features

# Prediction Thresholds
PREDICTION_THRESHOLDS = {
    "strong_buy": 0.75,
    "buy": 0.60,
    "hold": 0.40,
    "sell": 0.25,
    "strong_sell": 0.0,
}

# Streamlit Dashboard Settings
DASHBOARD_CONFIG = {
    "page_title": "Crypto Advisor",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "refresh_interval": 60,  # seconds
    "chart_height": 500,
    "sparkline_height": 100,
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Timezone
TIMEZONE = os.getenv("TIMEZONE", "UTC")

# Database Configuration
DB_CONFIG = {
    "read_only": False,
    "access_mode": "automatic",
}

# Performance Settings
BATCH_SIZE = 1000  # Batch size for bulk inserts
CONNECTION_POOL_SIZE = 5

# Data Validation
PRICE_VALIDATION = {
    "min_price": 0.000001,
    "max_price": 10000000,
    "max_price_change_percent": 100,  # Maximum % change per hour
}

# Error Recovery
ERROR_RECOVERY = {
    "max_consecutive_failures": 3,
    "recovery_wait_time": 300,  # seconds
    "notification_threshold": 5,  # Send notification after N failures
}
