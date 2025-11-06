# 📈 Crypto Advisor Tool

AI-powered cryptocurrency analysis and trading advisor with intelligent data management, technical indicators, and machine learning predictions.

## 🌟 Features

- **Smart Data Refresh**: Intelligent incremental updates that fetch only new data, preventing duplicates
- **Real-time Tracking**: Monitor top 5 cryptocurrencies (BTC, ETH, BNB, XRP, ADA)
- **Technical Analysis**: Calculate moving averages, RSI, MACD, Bollinger Bands
- **ML Predictions**: Random Forest model for BUY/SELL/HOLD signals
- **Interactive Dashboard**: Beautiful Streamlit interface with 4 pages
- **Portfolio Simulator**: Test investment strategies and allocation
- **Automated Gap Filling**: Automatically identifies and fills missing data
- **Comprehensive Logging**: Track all operations for debugging

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/djfrancesco/crypto-advisor-tool.git
cd crypto-advisor-tool
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment (optional)**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Initialize database and load historical data**
```bash
# Initialize database
python -c "from database.db_manager import initialize_database; initialize_database()"

# Load hourly data from Binance (RECOMMENDED - ~2 min for 90 days)
python load_hourly_data.py 90

# Alternative: Daily data from CSV (faster but less granular)
python -m data_collector.bulk_loader --load-all --days 365 --interval d
```

6. **Run the dashboard**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📁 Project Structure

```
crypto-advisor-tool/
├── database/
│   ├── __init__.py
│   ├── schemas.sql           # Database schema
│   └── db_manager.py          # Database operations
├── data_collector/
│   ├── __init__.py
│   ├── api_client.py          # CoinGecko API wrapper
│   └── data_refresher.py      # Smart refresh logic ⭐
├── analysis/
│   ├── __init__.py
│   ├── technical_indicators.py # Technical analysis
│   └── ml_predictor.py         # Machine learning
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Utility functions
├── tests/
│   ├── __init__.py
│   └── test_database.py        # Test suite
├── models/                     # ML models (generated)
├── app.py                      # Streamlit dashboard
├── config.py                   # Configuration
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🎯 Usage

### Dashboard Pages

#### 1. Overview
- Real-time prices for all tracked cryptocurrencies
- 24-hour price changes
- ML trading signals with confidence scores
- Mini sparkline charts
- Quick refresh button

#### 2. Detailed Analysis
- Interactive price charts with technical indicators
- Moving averages (7-day and 30-day)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Trading volume charts
- Historical ML predictions

#### 3. Portfolio Simulator
- Input investment amount
- Allocate percentages to each cryptocurrency
- View predicted returns based on ML signals
- Visualize allocation with pie chart
- Calculate risk metrics

#### 4. Data Management
- Manual data refresh for all cryptocurrencies
- Retrain ML models
- View database statistics
- Export data to CSV
- Clear caches

### Data Sources

The tool supports **multiple data sources** for maximum flexibility:

#### Option 1: Binance API - Hourly Data (RECOMMENDED)
```bash
# Load 90 days of hourly data (~2 minutes for 10,800 records)
python load_hourly_data.py 90

# Customize days
python load_hourly_data.py 30   # Last 30 days
```
- **Granularity**: Hourly (24 records per day)
- **Source**: Binance exchange API
- **Speed**: ~2 minutes for 90 days × 5 coins
- **Rate Limit**: 1200 req/min
- **Best for**: Detailed analysis and ML predictions

#### Option 2: Bulk CSV Loading - Daily Data
```bash
# Load 1 year of daily data in ~20 seconds
python -m data_collector.bulk_loader --load-all --days 365 --interval d

# Load specific coin
python -m data_collector.bulk_loader --coin bitcoin --days 90 --interval d
```
- **Granularity**: Daily (1 record per day)
- **Source**: CryptoDataDownload.com
- **Speed**: Instant bulk downloads (no rate limits)
- **Best for**: Quick setup, long-term trends

#### Option 3: Binance API - Incremental Updates (Default)
- **Rate Limit**: 1200 requests/minute (48x better than CoinGecko)
- **Data**: Direct from Binance exchange
- **Best for**: Ongoing incremental updates
- Configured via `DATA_SOURCE="binance"` in config.py (default)

#### Option 4: CoinGecko API
- **Rate Limit**: 25-30 requests/minute (free tier)
- **Data**: Aggregated from multiple sources
- **Best for**: Backup/fallback option
- Configure via `DATA_SOURCE="coingecko"` in config.py

#### Auto Mode
Set `DATA_SOURCE="auto"` to try Binance first, fallback to CoinGecko on failure.

### Programmatic Usage

```python
# Initialize database
from database.db_manager import initialize_database
initialize_database()

# Bulk load historical data (fast!)
from data_collector.bulk_loader import BulkDataLoader
loader = BulkDataLoader()
loader.load_all(limit_days=365)

# Refresh data for all cryptocurrencies (uses configured source)
from data_collector.data_refresher import get_refresher
refresher = get_refresher()
results = refresher.refresh_all()

# Calculate technical indicators
from analysis.technical_indicators import get_analyzer
analyzer = get_analyzer()
analyzer.batch_calculate_all()

# Generate ML predictions
from analysis.ml_predictor import get_predictor
predictor = get_predictor()
predictions = predictor.batch_predict_all()
```

## 🧠 Smart Refresh Mechanism

The **Smart Refresh** is the heart of this system. It ensures data integrity while minimizing API calls:

### How It Works

1. **First Run**: Fetches 365 days of historical data
2. **Subsequent Runs**: Fetches ONLY new data since last update
3. **Gap Detection**: Automatically identifies missing data periods
4. **Gap Filling**: Fetches data to fill any gaps
5. **Duplicate Prevention**: Database constraints prevent duplicate entries
6. **Operation Logging**: All operations logged for debugging

### Key Features

- ✅ Never fetches data that already exists
- ✅ Automatically fills gaps from connection losses
- ✅ Respects API rate limits
- ✅ Handles timezone conversions properly
- ✅ Validates all incoming data
- ✅ Provides detailed logging

## 📊 Technical Indicators

### Calculated Indicators

- **Moving Averages**: Short-term (7-day) and long-term (30-day)
- **RSI**: Relative Strength Index (14-day period)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **Support/Resistance**: Price levels identification

### Signal Generation

Each indicator generates signals:
- **BULLISH**: Suggests buying opportunity
- **BEARISH**: Suggests selling opportunity
- **NEUTRAL**: No strong signal

## 🤖 Machine Learning

### Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Price changes, technical indicators, volume patterns
- **Labels**: BUY (0), HOLD (1), SELL (2)
- **Training**: 80% train, 20% test split
- **Validation**: 5-fold cross-validation

### Prediction Process

1. **Feature Engineering**: Extract 15+ features from price data
2. **Model Training**: Train Random Forest on historical data
3. **Prediction**: Generate signal for next 3 days
4. **Confidence**: Probability score for the prediction
5. **Storage**: Save prediction to database

### Model Performance

- Target accuracy: >60%
- Includes precision, recall, and F1 scores
- Cross-validation for robustness
- Retrainable with latest data

## ⚙️ Configuration

Edit `config.py` or use environment variables:

```python
# API Configuration
COINGECKO_API_KEY = ""  # Optional for free tier

# Data Refresh
REFRESH_INTERVAL_MINUTES = 15
INITIAL_HISTORY_DAYS = 365

# Technical Indicators
TECHNICAL_INDICATORS = {
    "moving_averages": {"short_period": 7, "long_period": 30},
    "rsi": {"period": 14},
    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
}

# Machine Learning
ML_CONFIG = {
    "prediction_days": 3,
    "training_window_days": 90,
    "n_estimators": 100,
    "max_depth": 10,
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_database.py
```

## 📈 Database Schema

### Main Tables

- **cryptocurrencies**: Metadata for tracked coins
- **price_history**: Historical price data (with duplicate prevention)
- **technical_indicators**: Calculated indicators
- **predictions**: ML predictions and signals
- **refresh_log**: Operation tracking
- **data_quality_metrics**: Data health monitoring

### Key Constraints

- `UNIQUE(crypto_id, timestamp)` on price_history prevents duplicates
- Foreign key relationships ensure data integrity
- Indexes on timestamp columns for fast queries

## 🔧 Troubleshooting

### No data appearing

1. Check database initialization: `ls *.duckdb`
2. Run manual refresh in Data Management page
3. Check logs for API errors

### API rate limits

- Free tier: 10-30 calls/minute
- System includes rate limiting and retry logic
- Upgrade to paid tier for higher limits

### ML model errors

- Ensure sufficient historical data (90+ days)
- Retrain models from Data Management page
- Check logs for specific errors

### Duplicate data

- Should never happen due to database constraints
- If occurs, check `refresh_log` table
- Report as bug with logs

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- CoinGecko for cryptocurrency data API
- Streamlit for the dashboard framework
- scikit-learn for machine learning tools
- DuckDB for fast analytics database

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/djfrancesco/crypto-advisor-tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/djfrancesco/crypto-advisor-tool/discussions)

## 🗺️ Roadmap

- [ ] Add more cryptocurrencies
- [ ] Implement alerting system
- [ ] Add backtesting framework
- [ ] Support multiple fiat currencies
- [ ] Add sentiment analysis
- [ ] Create mobile app
- [ ] Add real-time WebSocket updates
- [ ] Implement paper trading

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Cryptocurrency trading involves significant risk. Always do your own research and consult with financial advisors before making investment decisions. Past performance does not guarantee future results.

---

**Built with ❤️ for the crypto community**
