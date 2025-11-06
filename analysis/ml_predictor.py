"""
Machine Learning Predictor for Cryptocurrency Trading Signals
Uses Random Forest to predict BUY/SELL/HOLD signals
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from config import (
    ML_CONFIG,
    FEATURE_PERIODS,
    PREDICTION_THRESHOLDS,
    CRYPTOCURRENCIES,
    PROJECT_ROOT,
)
from database.db_manager import (
    get_crypto_id,
    get_price_history,
    save_prediction,
)
from utils.helpers import performance_monitor

logger = logging.getLogger(__name__)


class MLPredictor:
    """Machine Learning predictor for crypto trading signals"""

    def __init__(self):
        self.config = ML_CONFIG
        self.models = {}  # Store model for each cryptocurrency
        self.model_dir = PROJECT_ROOT / "models"
        self.model_dir.mkdir(exist_ok=True)

    @performance_monitor
    def train_model(self, crypto_id: int, coin_id: str) -> Dict[str, any]:
        """
        Train ML model for a cryptocurrency

        Returns:
            Dict with training statistics
        """
        logger.info(f"Training model for {coin_id}")

        # Get historical data
        df = get_price_history(crypto_id)

        if df.empty or len(df) < self.config['min_training_samples']:
            logger.warning(
                f"Insufficient data for {coin_id}, "
                f"need at least {self.config['min_training_samples']} records"
            )
            return {"status": "failed", "error": "Insufficient data"}

        # Prepare features and labels
        X, y = self._prepare_features(df)

        if len(X) < self.config['min_training_samples']:
            logger.warning(f"Not enough valid samples for {coin_id}")
            return {"status": "failed", "error": "Not enough valid samples"}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_split_ratio'],
            random_state=self.config['random_state'],
            shuffle=False  # Important for time series!
        )

        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            random_state=self.config['random_state'],
            n_jobs=-1  # Use all CPU cores
        )

        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        logger.info(
            f"Model trained for {coin_id}: "
            f"Accuracy={accuracy:.3f}, F1={f1:.3f}, CV={cv_scores.mean():.3f}"
        )

        # Save model
        self.models[crypto_id] = model
        self._save_model(model, crypto_id, coin_id)

        return {
            "status": "success",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X.shape[1],
        }

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from price data

        Features include:
        - Price changes over multiple periods
        - Technical indicators (if available)
        - Volume changes
        - Market cap changes

        Labels:
        - 0 = SELL (price drops > 2%)
        - 1 = HOLD (price changes within ±2%)
        - 2 = BUY (price rises > 2%)
        """
        df = df.copy()
        df = df.sort_values('timestamp')

        # Feature 1: Price changes over different periods
        for period in FEATURE_PERIODS:
            df[f'price_change_{period}d'] = df['price'].pct_change(periods=period) * 100

        # Feature 2: Moving averages ratio
        df['ma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        df['ma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
        df['ma_ratio'] = df['ma_7'] / df['ma_30']

        # Feature 3: Volatility
        df['volatility'] = df['price'].rolling(window=7, min_periods=1).std()

        # Feature 4: Volume changes
        if 'total_volume' in df.columns:
            for period in [1, 3, 7]:
                df[f'volume_change_{period}d'] = df['total_volume'].pct_change(periods=period) * 100

        # Feature 5: RSI (if not present, calculate)
        if 'rsi' not in df.columns or df['rsi'].isna().all():
            df = self._calculate_simple_rsi(df)

        # Create labels based on future price movement
        prediction_days = self.config['prediction_days']
        df['future_price'] = df['price'].shift(-prediction_days)
        df['future_return'] = ((df['future_price'] - df['price']) / df['price']) * 100

        # Label classification
        df['label'] = 1  # Default: HOLD
        df.loc[df['future_return'] > 2, 'label'] = 2  # BUY
        df.loc[df['future_return'] < -2, 'label'] = 0  # SELL

        # Select features for training
        feature_cols = [
            col for col in df.columns
            if col.startswith('price_change_') or
               col.startswith('volume_change_') or
               col in ['ma_ratio', 'volatility', 'rsi']
        ]

        # Remove rows with NaN values
        df = df[feature_cols + ['label']].dropna()

        X = df[feature_cols].values
        y = df['label'].values

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")

        return X, y

    def _calculate_simple_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI if not present"""
        delta = df['price'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()

        rs = avg_gains / avg_losses
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].replace([np.inf, -np.inf], np.nan).fillna(50)

        return df

    def predict_signal(
        self,
        crypto_id: int,
        coin_id: str
    ) -> Optional[Dict[str, any]]:
        """
        Predict trading signal for a cryptocurrency

        Returns:
            Dict with prediction details or None
        """
        # Load or get model
        if crypto_id not in self.models:
            model = self._load_model(crypto_id, coin_id)
            if model is None:
                logger.info(f"No trained model for {coin_id}, training new model...")
                result = self.train_model(crypto_id, coin_id)
                if result['status'] != 'success':
                    logger.error(f"Failed to train model for {coin_id}")
                    return None
                model = self.models[crypto_id]
            else:
                self.models[crypto_id] = model

        # Get recent data
        df = get_price_history(crypto_id)

        if df.empty:
            logger.warning(f"No data available for {coin_id}")
            return None

        # Prepare features for latest data point
        X, _ = self._prepare_features(df)

        if len(X) == 0:
            logger.warning(f"Could not prepare features for {coin_id}")
            return None

        # Get latest features
        latest_features = X[-1].reshape(1, -1)

        # Make prediction
        model = self.models[crypto_id]
        prediction = model.predict(latest_features)[0]
        probabilities = model.predict_proba(latest_features)[0]

        # Map prediction to signal
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = signal_map[prediction]
        confidence = probabilities[prediction]

        # Get predicted price (simple estimate based on historical patterns)
        latest_price = df.iloc[-1]['price']
        if signal == "BUY":
            predicted_price = latest_price * 1.02  # +2%
        elif signal == "SELL":
            predicted_price = latest_price * 0.98  # -2%
        else:
            predicted_price = latest_price

        logger.info(
            f"Prediction for {coin_id}: {signal} "
            f"(confidence={confidence:.2%}, price=${latest_price:.2f})"
        )

        # Save prediction to database
        now = datetime.utcnow()
        target_date = now + timedelta(days=self.config['prediction_days'])

        save_prediction(
            crypto_id=crypto_id,
            prediction_date=now,
            target_date=target_date,
            signal=signal,
            confidence=confidence,
            predicted_price=predicted_price,
            model_version="v1",
            features_used=json.dumps({"feature_count": X.shape[1]}),
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "current_price": latest_price,
            "prediction_date": now,
            "target_date": target_date,
            "probabilities": {
                "SELL": probabilities[0],
                "HOLD": probabilities[1],
                "BUY": probabilities[2],
            }
        }

    def batch_predict_all(self) -> Dict[str, Dict]:
        """
        Generate predictions for all tracked cryptocurrencies

        Returns:
            Dict with coin_id as key and prediction as value
        """
        logger.info("Batch predicting signals for all cryptocurrencies")
        results = {}

        for crypto in CRYPTOCURRENCIES:
            coin_id = crypto['id']
            crypto_id = get_crypto_id(coin_id)

            if not crypto_id:
                logger.warning(f"Unknown cryptocurrency: {coin_id}")
                continue

            try:
                prediction = self.predict_signal(crypto_id, coin_id)
                if prediction:
                    results[coin_id] = prediction
                else:
                    results[coin_id] = {"status": "failed", "error": "Prediction failed"}
            except Exception as e:
                logger.error(f"Failed to predict for {coin_id}: {e}")
                results[coin_id] = {"status": "failed", "error": str(e)}

        logger.info(f"Batch prediction completed: {len(results)} cryptocurrencies")
        return results

    def _save_model(self, model, crypto_id: int, coin_id: str):
        """Save trained model to disk"""
        model_path = self.model_dir / f"{coin_id}_model.pkl"

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved for {coin_id}")
        except Exception as e:
            logger.error(f"Failed to save model for {coin_id}: {e}")

    def _load_model(self, crypto_id: int, coin_id: str):
        """Load trained model from disk"""
        model_path = self.model_dir / f"{coin_id}_model.pkl"

        if not model_path.exists():
            return None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded for {coin_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model for {coin_id}: {e}")
            return None

    def retrain_all_models(self) -> Dict[str, Dict]:
        """
        Retrain models for all cryptocurrencies

        Returns:
            Dict with training results for each coin
        """
        logger.info("Retraining all models")
        results = {}

        for crypto in CRYPTOCURRENCIES:
            coin_id = crypto['id']
            crypto_id = get_crypto_id(coin_id)

            if not crypto_id:
                continue

            try:
                result = self.train_model(crypto_id, coin_id)
                results[coin_id] = result
            except Exception as e:
                logger.error(f"Failed to train model for {coin_id}: {e}")
                results[coin_id] = {"status": "failed", "error": str(e)}

        return results


# Global predictor instance
_predictor = None


def get_predictor() -> MLPredictor:
    """Get or create global MLPredictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
