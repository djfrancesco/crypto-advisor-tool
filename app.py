"""
Crypto Advisor Tool - Streamlit Dashboard
Multi-page application for cryptocurrency analysis and trading recommendations
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz

from config import DASHBOARD_CONFIG, CRYPTOCURRENCIES
from database.db_manager import (
    initialize_database,
    get_crypto_id,
    get_price_history,
    get_database_stats,
    get_latest_predictions,
)
from data_collector.data_refresher import get_refresher
from analysis.technical_indicators import get_analyzer
from analysis.ml_predictor import get_predictor
from utils.helpers import (
    setup_logging,
    format_currency,
    format_percentage,
    calculate_percentage_change,
)

# Setup logging
logger = setup_logging(__name__)

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG['page_title'],
    page_icon=DASHBOARD_CONFIG['page_icon'],
    layout=DASHBOARD_CONFIG['layout'],
    initial_sidebar_state=DASHBOARD_CONFIG['initial_sidebar_state'],
)

# Initialize database
@st.cache_resource
def init_database():
    """Initialize database on first run"""
    return initialize_database()

init_database()

# Sidebar navigation
st.sidebar.title("📈 Crypto Advisor")
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Detailed Analysis", "Portfolio Simulator", "Data Management"]
)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    st.sidebar.info("Dashboard will auto-refresh every 60 seconds")


# ============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================================

def render_overview_page():
    """Render the overview dashboard with current prices and signals"""
    st.title("📊 Cryptocurrency Overview")

    # Refresh data button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"Last updated: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    with col2:
        if st.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Refreshing data..."):
                refresher = get_refresher()
                results = refresher.refresh_all()
                st.success(f"Refreshed {len(results)} cryptocurrencies")
                st.rerun()
    with col3:
        if st.button("🤖 Update Predictions"):
            with st.spinner("Generating predictions..."):
                predictor = get_predictor()
                results = predictor.batch_predict_all()
                st.success(f"Generated {len(results)} predictions")
                st.rerun()

    # Get data for all cryptocurrencies
    refresher = get_refresher()
    predictor = get_predictor()

    # Display cryptocurrency cards
    for crypto in CRYPTOCURRENCIES:
        coin_id = crypto['id']
        crypto_id = get_crypto_id(coin_id)

        if not crypto_id:
            continue

        # Get latest data
        df = get_price_history(crypto_id)

        if df.empty:
            st.warning(f"No data available for {crypto['name']}")
            continue

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        # Calculate 24h change
        price_change = calculate_percentage_change(previous['price'], latest['price'])

        # Get prediction
        try:
            prediction = predictor.predict_signal(crypto_id, coin_id)
        except:
            prediction = None

        # Create card
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

            with col1:
                st.subheader(f"{crypto['symbol']} - {crypto['name']}")
                st.write(f"💰 {format_currency(latest['price'])}")

            with col2:
                change_color = "green" if price_change >= 0 else "red"
                st.metric(
                    "24h Change",
                    format_percentage(price_change),
                    delta=format_percentage(price_change),
                )

            with col3:
                if 'market_cap' in df.columns and latest.get('market_cap'):
                    st.write("**Market Cap**")
                    st.write(format_currency(latest['market_cap']))

            with col4:
                if prediction:
                    signal = prediction['signal']
                    confidence = prediction['confidence']

                    signal_color = {
                        'BUY': '🟢',
                        'HOLD': '🟡',
                        'SELL': '🔴',
                    }.get(signal, '⚪')

                    st.write(f"**Signal:** {signal_color} **{signal}**")
                    st.write(f"Confidence: {confidence:.1%}")

            # Mini sparkline chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                line=dict(color='#1f77b4', width=1),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
            ))
            fig.update_layout(
                height=100,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, width='stretch', key=f"sparkline_{coin_id}")

        st.divider()


# ============================================================================
# PAGE 2: DETAILED ANALYSIS
# ============================================================================

def render_detailed_analysis():
    """Render detailed technical analysis for a selected cryptocurrency"""
    st.title("🔍 Detailed Analysis")

    # Cryptocurrency selector
    crypto_options = {f"{c['symbol']} - {c['name']}": c['id'] for c in CRYPTOCURRENCIES}
    selected = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    coin_id = crypto_options[selected]
    crypto_id = get_crypto_id(coin_id)

    if not crypto_id:
        st.error("Cryptocurrency not found")
        return

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.slider("Days of history", 7, 365, 90)

    # Get data
    df = get_price_history(crypto_id)

    if df.empty:
        st.warning(f"No data available for {selected}")
        return

    # Filter by date range
    cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days_back)
    df = df[df['timestamp'] >= cutoff_date]

    # Calculate technical indicators
    with st.spinner("Calculating technical indicators..."):
        analyzer = get_analyzer()
        analyzer.calculate_all_indicators(crypto_id, coin_id)

    # Refresh data to get indicators
    df = get_price_history(crypto_id)
    df = df[df['timestamp'] >= cutoff_date]

    # Main price chart with technical indicators
    st.subheader("📈 Price Chart with Technical Indicators")

    fig = go.Figure()

    # Candlestick would require OHLC data, using line for now
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        name="Price",
        line=dict(color='#1f77b4', width=2),
    ))

    # Add moving averages if available
    if 'ma_short' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ma_short'],
            name="MA Short (7d)",
            line=dict(color='orange', width=1, dash='dash'),
        ))

    if 'ma_long' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ma_long'],
            name="MA Long (30d)",
            line=dict(color='red', width=1, dash='dash'),
        ))

    # Add Bollinger Bands if available
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bb_upper'],
            name="BB Upper",
            line=dict(color='gray', width=1, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bb_lower'],
            name="BB Lower",
            line=dict(color='gray', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)',
        ))

    fig.update_layout(
        height=DASHBOARD_CONFIG['chart_height'],
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
    )

    st.plotly_chart(fig, width='stretch')

    # RSI Chart
    if 'rsi' in df.columns:
        st.subheader("📊 Relative Strength Index (RSI)")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['rsi'],
            name="RSI",
            line=dict(color='purple', width=2),
        ))

        # Add overbought/oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

        fig_rsi.update_layout(
            height=200,
            yaxis_title="RSI",
            yaxis_range=[0, 100],
            hovermode='x unified',
        )

        st.plotly_chart(fig_rsi, width='stretch')

    # MACD Chart
    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        st.subheader("📉 MACD")

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['macd'],
            name="MACD",
            line=dict(color='blue', width=2),
        ))
        fig_macd.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['macd_signal'],
            name="Signal",
            line=dict(color='orange', width=2),
        ))
        fig_macd.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['macd_histogram'],
            name="Histogram",
            marker_color='gray',
        ))

        fig_macd.update_layout(
            height=200,
            yaxis_title="MACD",
            hovermode='x unified',
        )

        st.plotly_chart(fig_macd, width='stretch')

    # Volume Chart
    if 'total_volume' in df.columns:
        st.subheader("📊 Trading Volume")

        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['total_volume'],
            name="Volume",
            marker_color='lightblue',
        ))

        fig_volume.update_layout(
            height=200,
            yaxis_title="Volume (USD)",
            hovermode='x unified',
        )

        st.plotly_chart(fig_volume, width='stretch')

    # ML Predictions
    st.subheader("🤖 ML Predictions")

    predictions_df = get_latest_predictions(crypto_id, limit=5)

    if not predictions_df.empty:
        st.dataframe(predictions_df, width='stretch')
    else:
        st.info("No predictions available yet. Click 'Update Predictions' to generate.")


# ============================================================================
# PAGE 3: PORTFOLIO SIMULATOR
# ============================================================================

def render_portfolio_simulator():
    """Render portfolio simulation and allocation recommendations"""
    st.title("💼 Portfolio Simulator")

    st.write("Simulate your cryptocurrency portfolio allocation and potential returns.")

    # Investment amount
    investment = st.number_input(
        "Total Investment Amount (USD)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=100.0,
    )

    # Allocation sliders
    st.subheader("Portfolio Allocation")

    allocations = {}
    remaining = 100.0

    for i, crypto in enumerate(CRYPTOCURRENCIES):
        max_allocation = remaining if i < len(CRYPTOCURRENCIES) - 1 else remaining
        allocation = st.slider(
            f"{crypto['symbol']} - {crypto['name']}",
            0.0,
            max_allocation,
            min(20.0, max_allocation),
            step=1.0,
            key=f"alloc_{crypto['id']}"
        )
        allocations[crypto['id']] = allocation
        remaining -= allocation

    st.write(f"**Unallocated:** {remaining:.1f}%")

    if abs(remaining) > 0.01:
        st.warning("Please allocate 100% of your portfolio")
        return

    # Calculate portfolio
    st.subheader("📊 Portfolio Summary")

    predictor = get_predictor()
    total_value = investment
    predicted_total = 0

    portfolio_data = []

    for crypto in CRYPTOCURRENCIES:
        coin_id = crypto['id']
        crypto_id = get_crypto_id(coin_id)
        allocation_pct = allocations[coin_id]

        if allocation_pct == 0:
            continue

        amount = investment * (allocation_pct / 100)

        # Get current price
        df = get_price_history(crypto_id)
        if df.empty:
            continue

        current_price = df.iloc[-1]['price']
        quantity = amount / current_price

        # Get prediction
        try:
            prediction = predictor.predict_signal(crypto_id, coin_id)
            if prediction:
                predicted_price = prediction['predicted_price']
                predicted_value = quantity * predicted_price
                predicted_return = ((predicted_value - amount) / amount) * 100
            else:
                predicted_price = current_price
                predicted_value = amount
                predicted_return = 0
        except:
            predicted_price = current_price
            predicted_value = amount
            predicted_return = 0

        predicted_total += predicted_value

        portfolio_data.append({
            'Cryptocurrency': f"{crypto['symbol']} - {crypto['name']}",
            'Allocation': f"{allocation_pct:.1f}%",
            'Amount (USD)': f"${amount:,.2f}",
            'Quantity': f"{quantity:,.6f}",
            'Current Price': f"${current_price:,.2f}",
            'Predicted Price': f"${predicted_price:,.2f}",
            'Predicted Value': f"${predicted_value:,.2f}",
            'Expected Return': format_percentage(predicted_return),
        })

    # Display portfolio table
    portfolio_df = pd.DataFrame(portfolio_data)
    st.dataframe(portfolio_df, width='stretch')

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Investment", format_currency(investment))

    with col2:
        st.metric("Predicted Value", format_currency(predicted_total))

    with col3:
        total_return = ((predicted_total - investment) / investment) * 100
        st.metric("Expected Return", format_percentage(total_return), delta=format_percentage(total_return))

    # Allocation pie chart
    st.subheader("📊 Allocation Chart")

    alloc_data = []
    for crypto in CRYPTOCURRENCIES:
        coin_id = crypto['id']
        allocation_pct = allocations[coin_id]
        if allocation_pct > 0:
            alloc_data.append({
                'Cryptocurrency': crypto['symbol'],
                'Allocation': allocation_pct
            })

    if alloc_data:
        fig = px.pie(
            alloc_data,
            values='Allocation',
            names='Cryptocurrency',
            title='Portfolio Allocation'
        )
        st.plotly_chart(fig, width='stretch')


# ============================================================================
# PAGE 4: DATA MANAGEMENT
# ============================================================================

def render_data_management():
    """Render data management and database statistics"""
    st.title("⚙️ Data Management")

    # Manual refresh controls
    st.subheader("🔄 Data Refresh")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Refresh All Data", type="primary"):
            with st.spinner("Refreshing all cryptocurrency data..."):
                refresher = get_refresher()
                results = refresher.refresh_all()

                success = sum(1 for r in results.values() if r.get('status') == 'success')
                total_records = sum(r.get('records_added', 0) for r in results.values())

                st.success(f"✅ Refreshed {success}/{len(results)} cryptocurrencies ({total_records} new records)")

                for coin_id, result in results.items():
                    if result.get('status') == 'success':
                        st.info(f"{coin_id}: +{result.get('records_added', 0)} records")
                    else:
                        st.error(f"{coin_id}: {result.get('error', 'Failed')}")

    with col2:
        if st.button("Retrain All ML Models"):
            with st.spinner("Retraining machine learning models..."):
                predictor = get_predictor()
                results = predictor.retrain_all_models()

                success = sum(1 for r in results.values() if r.get('status') == 'success')

                st.success(f"✅ Trained {success}/{len(results)} models")

                for coin_id, result in results.items():
                    if result.get('status') == 'success':
                        accuracy = result.get('accuracy', 0)
                        st.info(f"{coin_id}: Accuracy = {accuracy:.2%}")

    # Database statistics
    st.subheader("📊 Database Statistics")

    stats = get_database_stats()

    if stats.get('crypto_stats'):
        st.write("**Data Availability:**")

        stats_data = []
        for crypto_stat in stats['crypto_stats']:
            stats_data.append({
                'Symbol': crypto_stat['symbol'],
                'Records': crypto_stat['records'],
                'Earliest Data': crypto_stat['earliest'].strftime('%Y-%m-%d') if crypto_stat['earliest'] else 'N/A',
                'Latest Data': crypto_stat['latest'].strftime('%Y-%m-%d') if crypto_stat['latest'] else 'N/A',
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, width='stretch')

    # Export functionality
    st.subheader("📥 Export Data")

    export_crypto = st.selectbox(
        "Select cryptocurrency to export",
        [f"{c['symbol']} - {c['name']}" for c in CRYPTOCURRENCIES]
    )

    if st.button("Export to CSV"):
        crypto_id = get_crypto_id(export_crypto.split(' - ')[0].lower())
        df = get_price_history(crypto_id)

        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{export_crypto.split(' - ')[0]}_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data available for export")

    # Clear cache
    st.subheader("🗑️ Cache Management")

    if st.button("Clear All Caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ All caches cleared")
        st.rerun()


# ============================================================================
# MAIN APP
# ============================================================================

# Render selected page
if page == "Overview":
    render_overview_page()
elif page == "Detailed Analysis":
    render_detailed_analysis()
elif page == "Portfolio Simulator":
    render_portfolio_simulator()
elif page == "Data Management":
    render_data_management()

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()
