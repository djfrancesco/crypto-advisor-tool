"""
Technical Indicators Calculator
Calculates various technical indicators for cryptocurrency analysis
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from config import TECHNICAL_INDICATORS, CRYPTOCURRENCIES
from database.db_manager import (
    get_crypto_id,
    get_price_history,
    update_technical_indicators,
)
from utils.helpers import performance_monitor

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Calculate technical indicators for cryptocurrency price data"""

    def __init__(self):
        self.config = TECHNICAL_INDICATORS

    @performance_monitor
    def calculate_all_indicators(self, crypto_id: int, coin_id: str) -> int:
        """
        Calculate all technical indicators for a cryptocurrency

        Returns:
            Number of indicator records updated
        """
        logger.info(f"Calculating technical indicators for {coin_id}")

        # Get price history
        df = get_price_history(crypto_id)

        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {coin_id}, need at least 30 records")
            return 0

        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp')

        # Calculate all indicators
        df = self._calculate_moving_averages(df)
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_bollinger_bands(df)
        df = self._identify_support_resistance(df)

        # Prepare data for database
        indicators_data = []
        for _, row in df.iterrows():
            indicators_data.append({
                'timestamp': row['timestamp'],
                'ma_short': row.get('ma_short'),
                'ma_long': row.get('ma_long'),
                'rsi': row.get('rsi'),
                'macd': row.get('macd'),
                'macd_signal': row.get('macd_signal'),
                'macd_histogram': row.get('macd_histogram'),
                'bb_upper': row.get('bb_upper'),
                'bb_middle': row.get('bb_middle'),
                'bb_lower': row.get('bb_lower'),
                'support_level': row.get('support_level'),
                'resistance_level': row.get('resistance_level'),
            })

        # Update database
        updated = update_technical_indicators(crypto_id, indicators_data)
        logger.info(f"Updated {updated} technical indicator records for {coin_id}")

        return updated

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate short and long-term moving averages"""
        short_period = self.config['moving_averages']['short_period']
        long_period = self.config['moving_averages']['long_period']

        df['ma_short'] = df['price'].rolling(window=short_period, min_periods=1).mean()
        df['ma_long'] = df['price'].rolling(window=long_period, min_periods=1).mean()

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI)"""
        period = self.config['rsi']['period']

        # Calculate price changes
        delta = df['price'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        df['rsi'] = 100 - (100 / (1 + rs))

        # Handle edge cases
        df['rsi'] = df['rsi'].replace([np.inf, -np.inf], np.nan)
        df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI for missing values

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        fast_period = self.config['macd']['fast_period']
        slow_period = self.config['macd']['slow_period']
        signal_period = self.config['macd']['signal_period']

        # Calculate EMAs
        ema_fast = df['price'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['price'].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow

        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        period = self.config['bollinger_bands']['period']
        std_dev = self.config['bollinger_bands']['std_dev']

        # Calculate middle band (SMA)
        df['bb_middle'] = df['price'].rolling(window=period, min_periods=1).mean()

        # Calculate standard deviation
        rolling_std = df['price'].rolling(window=period, min_periods=1).std()

        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)

        return df

    def _identify_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Identify support and resistance levels
        Uses local minima/maxima in a rolling window
        """
        # Find local minima (support)
        df['local_min'] = df['price'].rolling(
            window=window,
            center=True,
            min_periods=1
        ).min()

        # Find local maxima (resistance)
        df['local_max'] = df['price'].rolling(
            window=window,
            center=True,
            min_periods=1
        ).max()

        # Identify support levels (where price equals local minimum)
        df['support_level'] = df.apply(
            lambda row: row['local_min'] if row['price'] == row['local_min'] else np.nan,
            axis=1
        )

        # Identify resistance levels (where price equals local maximum)
        df['resistance_level'] = df.apply(
            lambda row: row['local_max'] if row['price'] == row['local_max'] else np.nan,
            axis=1
        )

        # Forward fill to maintain current support/resistance
        df['support_level'] = df['support_level'].fillna(method='ffill')
        df['resistance_level'] = df['resistance_level'].fillna(method='ffill')

        # Clean up temporary columns
        df = df.drop(['local_min', 'local_max'], axis=1)

        return df

    def get_latest_indicators(self, crypto_id: int) -> Optional[Dict]:
        """
        Get the most recent technical indicators for a cryptocurrency

        Returns:
            Dict with latest indicator values or None
        """
        try:
            # Get price history with indicators
            df = get_price_history(crypto_id)

            if df.empty:
                return None

            # Calculate indicators if needed
            if 'rsi' not in df.columns or df['rsi'].isna().all():
                # Recalculate
                coin_id = None
                for crypto in CRYPTOCURRENCIES:
                    if get_crypto_id(crypto['id']) == crypto_id:
                        coin_id = crypto['id']
                        break

                if coin_id:
                    self.calculate_all_indicators(crypto_id, coin_id)
                    df = get_price_history(crypto_id)

            # Get latest row
            latest = df.iloc[-1]

            return {
                'timestamp': latest['timestamp'],
                'price': latest['price'],
                'ma_short': latest.get('ma_short'),
                'ma_long': latest.get('ma_long'),
                'rsi': latest.get('rsi'),
                'macd': latest.get('macd'),
                'macd_signal': latest.get('macd_signal'),
                'macd_histogram': latest.get('macd_histogram'),
                'bb_upper': latest.get('bb_upper'),
                'bb_middle': latest.get('bb_middle'),
                'bb_lower': latest.get('bb_lower'),
                'support_level': latest.get('support_level'),
                'resistance_level': latest.get('resistance_level'),
            }

        except Exception as e:
            logger.error(f"Failed to get latest indicators: {e}")
            return None

    def batch_calculate_all(self) -> Dict[str, int]:
        """
        Calculate indicators for all tracked cryptocurrencies

        Returns:
            Dict with coin_id as key and number of updated records as value
        """
        logger.info("Batch calculating indicators for all cryptocurrencies")
        results = {}

        for crypto in CRYPTOCURRENCIES:
            coin_id = crypto['id']
            crypto_id = get_crypto_id(coin_id)

            if not crypto_id:
                logger.warning(f"Unknown cryptocurrency: {coin_id}")
                continue

            try:
                updated = self.calculate_all_indicators(crypto_id, coin_id)
                results[coin_id] = updated
            except Exception as e:
                logger.error(f"Failed to calculate indicators for {coin_id}: {e}")
                results[coin_id] = 0

        total_updated = sum(results.values())
        logger.info(f"Batch calculation completed: {total_updated} total records updated")

        return results

    def analyze_signals(self, crypto_id: int) -> Dict[str, str]:
        """
        Analyze technical indicators to generate trading signals

        Returns:
            Dict with signal types and their values (BULLISH/BEARISH/NEUTRAL)
        """
        indicators = self.get_latest_indicators(crypto_id)

        if not indicators:
            return {}

        signals = {}

        # RSI Signal
        rsi = indicators.get('rsi')
        if rsi:
            if rsi > self.config['rsi']['overbought']:
                signals['rsi'] = 'BEARISH'
            elif rsi < self.config['rsi']['oversold']:
                signals['rsi'] = 'BULLISH'
            else:
                signals['rsi'] = 'NEUTRAL'

        # Moving Average Crossover
        ma_short = indicators.get('ma_short')
        ma_long = indicators.get('ma_long')
        if ma_short and ma_long:
            if ma_short > ma_long:
                signals['ma_crossover'] = 'BULLISH'
            else:
                signals['ma_crossover'] = 'BEARISH'

        # MACD Signal
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd and macd_signal:
            if macd > macd_signal:
                signals['macd'] = 'BULLISH'
            else:
                signals['macd'] = 'BEARISH'

        # Bollinger Bands
        price = indicators.get('price')
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        if price and bb_upper and bb_lower:
            if price >= bb_upper:
                signals['bollinger'] = 'BEARISH'  # Overbought
            elif price <= bb_lower:
                signals['bollinger'] = 'BULLISH'  # Oversold
            else:
                signals['bollinger'] = 'NEUTRAL'

        return signals


# Global analyzer instance
_analyzer = None


def get_analyzer() -> TechnicalAnalyzer:
    """Get or create global TechnicalAnalyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = TechnicalAnalyzer()
    return _analyzer
