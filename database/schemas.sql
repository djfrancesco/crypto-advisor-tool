-- Crypto Advisor Tool Database Schema
-- DuckDB SQL for creating all required tables

-- Cryptocurrencies Metadata
CREATE TABLE IF NOT EXISTS cryptocurrencies (
    id INTEGER PRIMARY KEY,
    coin_id VARCHAR NOT NULL UNIQUE,  -- CoinGecko ID (e.g., 'bitcoin')
    symbol VARCHAR NOT NULL,          -- Symbol (e.g., 'BTC')
    name VARCHAR NOT NULL,            -- Full name (e.g., 'Bitcoin')
    last_updated TIMESTAMP,           -- Last time data was fetched
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Price History (Core data table)
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY,
    crypto_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price DOUBLE NOT NULL,
    market_cap DOUBLE,
    total_volume DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crypto_id) REFERENCES cryptocurrencies(id),
    UNIQUE(crypto_id, timestamp)  -- Prevent duplicate entries
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_price_history_crypto_time
ON price_history(crypto_id, timestamp DESC);

-- Technical Indicators (Calculated metrics)
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY,
    crypto_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    ma_short DOUBLE,              -- Short-term moving average
    ma_long DOUBLE,               -- Long-term moving average
    rsi DOUBLE,                   -- Relative Strength Index
    macd DOUBLE,                  -- MACD line
    macd_signal DOUBLE,           -- MACD signal line
    macd_histogram DOUBLE,        -- MACD histogram
    bb_upper DOUBLE,              -- Bollinger Band upper
    bb_middle DOUBLE,             -- Bollinger Band middle
    bb_lower DOUBLE,              -- Bollinger Band lower
    support_level DOUBLE,         -- Support price level
    resistance_level DOUBLE,      -- Resistance price level
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crypto_id) REFERENCES cryptocurrencies(id),
    UNIQUE(crypto_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_technical_indicators_crypto_time
ON technical_indicators(crypto_id, timestamp DESC);

-- ML Predictions (Buy/Sell/Hold signals)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    crypto_id INTEGER NOT NULL,
    prediction_date TIMESTAMP NOT NULL,  -- When prediction was made
    target_date TIMESTAMP NOT NULL,      -- Date being predicted
    signal VARCHAR NOT NULL,             -- BUY, SELL, HOLD
    confidence DOUBLE NOT NULL,          -- Confidence score (0-1)
    predicted_price DOUBLE,              -- Predicted price
    actual_price DOUBLE,                 -- Actual price (filled later)
    accuracy_score DOUBLE,               -- Accuracy when actual is known
    model_version VARCHAR,               -- Model version identifier
    features_used TEXT,                  -- JSON of features used
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crypto_id) REFERENCES cryptocurrencies(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_crypto_target
ON predictions(crypto_id, target_date DESC);

-- Refresh Log (Track all data updates)
CREATE TABLE IF NOT EXISTS refresh_log (
    id INTEGER PRIMARY KEY,
    crypto_id INTEGER,               -- NULL for system-wide operations
    operation_type VARCHAR NOT NULL, -- 'initial_load', 'refresh', 'gap_fill'
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    records_added INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    status VARCHAR NOT NULL,         -- 'success', 'partial', 'failed'
    error_message TEXT,
    from_timestamp TIMESTAMP,        -- Data range start
    to_timestamp TIMESTAMP,          -- Data range end
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crypto_id) REFERENCES cryptocurrencies(id)
);

CREATE INDEX IF NOT EXISTS idx_refresh_log_time
ON refresh_log(created_at DESC);

-- Data Quality Metrics (Monitor data health)
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id INTEGER PRIMARY KEY,
    crypto_id INTEGER NOT NULL,
    check_date TIMESTAMP NOT NULL,
    total_records INTEGER,
    missing_periods INTEGER,
    duplicate_records INTEGER,
    outlier_records INTEGER,
    data_completeness_percent DOUBLE,
    quality_score DOUBLE,           -- Overall quality (0-100)
    issues_found TEXT,              -- JSON array of issues
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crypto_id) REFERENCES cryptocurrencies(id)
);

-- Model Performance Tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY,
    model_version VARCHAR NOT NULL,
    evaluation_date TIMESTAMP NOT NULL,
    crypto_id INTEGER,              -- NULL for overall metrics
    accuracy DOUBLE,
    precision_score DOUBLE,
    recall_score DOUBLE,
    f1_score DOUBLE,
    total_predictions INTEGER,
    correct_predictions INTEGER,
    training_samples INTEGER,
    test_samples INTEGER,
    hyperparameters TEXT,           -- JSON of parameters
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crypto_id) REFERENCES cryptocurrencies(id)
);

-- System Configuration (Store app settings)
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
