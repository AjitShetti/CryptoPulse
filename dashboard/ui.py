"""
CryptoPulse — Streamlit Dashboard
Real-time cryptocurrency price prediction dashboard.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ingestion.fetch_prices import BinanceFetcher
from data.processing import FeatureEngineer
from config.settings import API_BASE_URL, SYMBOL, INTERVAL, REFRESH_INTERVAL


# ── Interval helper ──────────────────────────────────────────────────────
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
}


def candles_in_24h(interval: str) -> int:
    """Return the number of candles that span 24 hours for the given interval."""
    minutes = INTERVAL_MINUTES.get(interval, 60)
    return max(1, int(24 * 60 / minutes))


# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoPulse Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-up {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .prediction-down {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: 800;
    }
    .header-title {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e30, #252540);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper Functions ─────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_candle_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch candle data from Binance."""
    fetcher = BinanceFetcher(symbol=symbol, interval=interval)
    candles = fetcher.fetch_klines(limit=limit)
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


@st.cache_data(ttl=REFRESH_INTERVAL)
def get_prediction(symbol: str, interval: str) -> dict | None:
    """Call the FastAPI /predict endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"symbol": symbol, "interval": interval},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


@st.cache_data(ttl=300)
def get_model_info() -> dict | None:
    """Call the FastAPI /model-info endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def get_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def add_indicators_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for charting."""
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator

    close = df["close"]
    df["sma_7"] = SMAIndicator(close, window=7).sma_indicator()
    df["sma_25"] = SMAIndicator(close, window=25).sma_indicator()
    df["ema_7"] = EMAIndicator(close, window=7).ema_indicator()

    macd_obj = MACD(close)
    df["macd_line"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    df["rsi"] = RSIIndicator(close, window=14).rsi()
    return df


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ CryptoPulse")
    st.markdown("---")

    symbol = st.selectbox(
        "Symbol",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
        index=0,
    )
    interval = st.selectbox(
        "Interval",
        ["15m", "30m", "1h", "4h", "1d"],
        index=2,
    )
    candle_count = st.slider("Candles", min_value=50, max_value=500, value=200, step=50)

    st.markdown("---")

    # API Status
    api_healthy = get_api_health()
    if api_healthy:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")
        st.caption("Start the API server to enable predictions.")

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ── Header ────────────────────────────────────────────────────────────
st.markdown('<p class="header-title">⚡ CryptoPulse Dashboard</p>', unsafe_allow_html=True)
st.markdown(f"Real-time {symbol} price analysis & ML prediction • `{interval}` candles")

# ── Fetch Data ───────────────────────────────────────────────────────
try:
    df = fetch_candle_data(symbol, interval, candle_count)
    df = add_indicators_to_df(df)
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()


# ── Top Metrics Row ──────────────────────────────────────────────────
current_price = df["close"].iloc[-1]
prev_price = df["close"].iloc[-2]
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

# Use interval-aware candle count for 24h aggregates
n_24h = candles_in_24h(interval)
volume_24h = df["volume"].tail(n_24h).sum() if len(df) >= n_24h else df["volume"].sum()
high_24h = df["high"].tail(n_24h).max() if len(df) >= n_24h else df["high"].max()
low_24h = df["low"].tail(n_24h).min() if len(df) >= n_24h else df["low"].min()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Current Price", f"${current_price:,.2f}", f"{price_change_pct:+.2f}%")
with col2:
    st.metric("24h High", f"${high_24h:,.2f}")
with col3:
    st.metric("24h Low", f"${low_24h:,.2f}")
with col4:
    st.metric("24h Volume", f"{volume_24h:,.0f}")
with col5:
    rsi_val = df["rsi"].iloc[-1] if pd.notna(df["rsi"].iloc[-1]) else 0
    rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
    st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)


# ── Prediction Section ───────────────────────────────────────────────
st.markdown("---")
pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    st.markdown("### 🤖 ML Prediction")
    if api_healthy:
        prediction = get_prediction(symbol, interval)
        if prediction:
            pred_class = "prediction-up" if prediction["prediction"] == "UP" else "prediction-down"
            pred_emoji = "🚀" if prediction["prediction"] == "UP" else "📉"
            pred_color = "#10b981" if prediction["prediction"] == "UP" else "#ef4444"

            st.markdown(
                f"""
                <div class="{pred_class}">
                    <div class="prediction-text" style="color: {pred_color}">
                        {pred_emoji} {prediction["prediction"]}
                    </div>
                    <div style="font-size: 1.2rem; color: #e2e8f0; margin-top: 8px;">
                        Confidence: {prediction["confidence"]*100:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                        Price: ${prediction["current_price"]:,.2f}
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                        Features: {prediction["features_used"]} indicators
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Could not get prediction. Is the model trained?")
    else:
        st.info("Start the API server to see predictions.")

with pred_col2:
    st.markdown("### 📊 Model Info")
    if api_healthy:
        model_info = get_model_info()
        if model_info:
            metrics = model_info.get("metrics", {})
            mi_col1, mi_col2, mi_col3, mi_col4 = st.columns(4)
            with mi_col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with mi_col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            with mi_col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            with mi_col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")

            st.caption(
                f"Trained: {model_info.get('trained_at', 'N/A')} • "
                f"Features: {model_info.get('feature_count', 0)} • "
                f"Model: {model_info.get('model_type', 'XGBClassifier')}"
            )

            # Top features
            top_features = metrics.get("top_features", [])
            if top_features:
                feat_df = pd.DataFrame(top_features[:8])
                st.bar_chart(feat_df.set_index("name")["importance"])
        else:
            st.warning("Model info unavailable.")
    else:
        st.info("Start the API server to see model metrics.")


# ── Price Chart ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Price Chart")

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("Candlestick + Moving Averages", "MACD", "RSI"),
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444",
    ),
    row=1,
    col=1,
)

# Moving Averages
for col_name, color, label in [
    ("sma_7", "#fbbf24", "SMA 7"),
    ("sma_25", "#3b82f6", "SMA 25"),
    ("ema_7", "#a78bfa", "EMA 7"),
]:
    if col_name in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[col_name],
                name=label,
                line=dict(color=color, width=1),
            ),
            row=1,
            col=1,
        )

# MACD
if "macd_line" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["macd_line"], name="MACD",
            line=dict(color="#3b82f6", width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["macd_signal"], name="Signal",
            line=dict(color="#f97316", width=1.5),
        ),
        row=2, col=1,
    )
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=df["datetime"], y=df["macd_hist"], name="MACD Hist",
            marker_color=colors, opacity=0.6,
        ),
        row=2, col=1,
    )

# RSI
if "rsi" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["rsi"], name="RSI",
            line=dict(color="#a78bfa", width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.5, row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(139,92,246,0.05)", line_width=0, row=3, col=1)

fig.update_layout(
    template="plotly_dark",
    height=800,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,26,0.8)",
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

st.plotly_chart(fig, use_container_width=True)


# ── Volume Chart ─────────────────────────────────────────────────────
st.markdown("### 📊 Volume")
vol_colors = ["#10b981" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef4444" for i in range(len(df))]

vol_fig = go.Figure(
    go.Bar(x=df["datetime"], y=df["volume"], marker_color=vol_colors, opacity=0.7)
)
vol_fig.update_layout(
    template="plotly_dark",
    height=200,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,26,0.8)",
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Volume"),
)
st.plotly_chart(vol_fig, use_container_width=True)


# ── Raw Data Table ───────────────────────────────────────────────────
with st.expander("📋 Raw Candle Data"):
    display_df = df[["datetime", "open", "high", "low", "close", "volume"]].tail(50)
    display_df = display_df.sort_values("datetime", ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "⚡ CryptoPulse v1.0 • Built with Streamlit, FastAPI & XGBoost • "
    "Data from Binance Public API • Not Financial Advice"
    "</div>",
    unsafe_allow_html=True,
)
