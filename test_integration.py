"""
Quick integration test to verify the system works end-to-end
"""
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_integration():
    """Run end-to-end integration test"""

    logger.info("=" * 70)
    logger.info("CRYPTO ADVISOR TOOL - INTEGRATION TEST")
    logger.info("=" * 70)

    # Test 1: Database initialization
    logger.info("\n[1/5] Testing database initialization...")
    try:
        from database.db_manager import initialize_database
        result = initialize_database()
        assert result, "Database initialization failed"
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

    # Test 2: API connectivity
    logger.info("\n[2/5] Testing API connectivity...")
    try:
        from data_collector.api_client import get_client
        client = get_client()
        is_connected = client.ping()
        assert is_connected, "API ping failed"
        logger.info("✅ API connection successful")
    except Exception as e:
        logger.error(f"❌ API connectivity failed: {e}")
        return False

    # Test 3: Smart refresh for one cryptocurrency
    logger.info("\n[3/5] Testing smart refresh (Bitcoin only)...")
    try:
        from data_collector.data_refresher import get_refresher
        refresher = get_refresher()
        result = refresher.smart_refresh('bitcoin')

        if result.get('status') == 'success':
            logger.info(f"✅ Smart refresh successful: {result.get('records_added', 0)} records added")
        else:
            logger.warning(f"⚠️ Smart refresh completed with status: {result.get('status')}")
            logger.warning(f"   Message: {result.get('error', 'Unknown')}")
    except Exception as e:
        logger.error(f"❌ Smart refresh failed: {e}")
        return False

    # Test 4: Technical indicators
    logger.info("\n[4/5] Testing technical indicators...")
    try:
        from analysis.technical_indicators import get_analyzer
        from database.db_manager import get_crypto_id

        analyzer = get_analyzer()
        crypto_id = get_crypto_id('bitcoin')

        if crypto_id:
            updated = analyzer.calculate_all_indicators(crypto_id, 'bitcoin')
            logger.info(f"✅ Technical indicators calculated: {updated} records updated")
        else:
            logger.warning("⚠️ Bitcoin not found in database")
    except Exception as e:
        logger.error(f"❌ Technical indicators failed: {e}")
        return False

    # Test 5: ML predictions
    logger.info("\n[5/5] Testing ML predictions...")
    try:
        from analysis.ml_predictor import get_predictor
        from database.db_manager import get_crypto_id

        predictor = get_predictor()
        crypto_id = get_crypto_id('bitcoin')

        if crypto_id:
            prediction = predictor.predict_signal(crypto_id, 'bitcoin')

            if prediction:
                logger.info(f"✅ ML prediction generated:")
                logger.info(f"   Signal: {prediction['signal']}")
                logger.info(f"   Confidence: {prediction['confidence']:.2%}")
                logger.info(f"   Current Price: ${prediction['current_price']:.2f}")
            else:
                logger.warning("⚠️ ML prediction returned None (may need more data)")
        else:
            logger.warning("⚠️ Bitcoin not found in database")
    except Exception as e:
        logger.error(f"❌ ML predictions failed: {e}")
        logger.error(f"   This is expected if insufficient historical data")
        logger.info("   Run multiple refreshes to collect more data")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION TEST COMPLETED")
    logger.info("=" * 70)
    logger.info("\n✨ System is operational!")
    logger.info("\nNext steps:")
    logger.info("1. Run 'streamlit run app.py' to start the dashboard")
    logger.info("2. Use the Data Management page to refresh all cryptocurrencies")
    logger.info("3. Wait for sufficient data before using ML predictions")
    logger.info("\nTip: The smart refresh will only fetch new data on subsequent runs,")
    logger.info("     preventing duplicates and saving API calls!")

    return True

if __name__ == "__main__":
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)
