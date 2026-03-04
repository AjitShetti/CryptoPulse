"""
CryptoPulse — Feature Engineering (Enhanced)
Loads raw OHLCV data from DB, computes technical indicators, lag features,
rolling statistics, and cross-signal features for improved ML accuracy.
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

from data.db import get_session, PriceHistory
from config.settings import SYMBOL
from core.logger import logging

# Minimum price change (%) to label as UP/DOWN — ignore noise in between
TARGET_THRESHOLD = 0.001  # 0.1%


class FeatureEngineer:
    """
    Loads OHLCV data from the database and engineers technical indicator
    features for the ML classification model.
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol

    def load_from_db(self) -> pd.DataFrame:
        """Load raw price history from database into a DataFrame."""
        session = get_session()
        try:
            records = (
                session.query(PriceHistory)
                .filter_by(symbol=self.symbol)
                .order_by(PriceHistory.open_time.asc())
                .all()
            )
            if not records:
                raise ValueError(f"No price data found for {self.symbol}")

            data = [
                {
                    "open_time": r.open_time,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "close_time": r.close_time,
                    "quote_volume": r.quote_volume,
                    "trades": r.trades,
                }
                for r in records
            ]
            df = pd.DataFrame(data)
            logging.info(f"Loaded {len(df)} rows from DB for {self.symbol}")
            return df
        finally:
            session.close()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators and add as new columns."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        # ── Trend Indicators ─────────────────────────────────────────────
        df["sma_7"] = SMAIndicator(close, window=7).sma_indicator()
        df["sma_25"] = SMAIndicator(close, window=25).sma_indicator()
        df["sma_99"] = SMAIndicator(close, window=99).sma_indicator()
        df["ema_7"] = EMAIndicator(close, window=7).ema_indicator()
        df["ema_25"] = EMAIndicator(close, window=25).ema_indicator()
        df["ema_50"] = EMAIndicator(close, window=50).ema_indicator()

        macd = MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # ADX — trend strength
        adx = ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # ── Momentum Indicators ──────────────────────────────────────────
        df["rsi_14"] = RSIIndicator(close, window=14).rsi()
        df["rsi_7"] = RSIIndicator(close, window=7).rsi()

        stoch = StochasticOscillator(high, low, close)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

        # ── Volatility Indicators ────────────────────────────────────────
        bb = BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        df["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

        # ── Volume Indicators ────────────────────────────────────────────
        df["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df["mfi"] = MFIIndicator(high, low, close, volume, window=14).money_flow_index()

        # ── Price-derived features ───────────────────────────────────────
        df["price_change"] = close.pct_change()
        df["price_change_3"] = close.pct_change(periods=3)
        df["price_change_7"] = close.pct_change(periods=7)
        df["price_change_14"] = close.pct_change(periods=14)
        df["high_low_ratio"] = high / (low + 1e-10)
        df["close_open_ratio"] = close / (open_ + 1e-10)
        df["upper_shadow"] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
        df["lower_shadow"] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
        df["body_ratio"] = np.abs(close - open_) / (high - low + 1e-10)
        df["volume_change"] = volume.pct_change()
        df["volume_sma_20"] = SMAIndicator(volume, window=20).sma_indicator()
        df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)

        logging.info(f"Added {len(df.columns)} columns after feature engineering")
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged price and indicator features to capture temporal patterns."""
        # Lagged close returns
        for lag in [1, 2, 3, 5, 10]:
            df[f"return_lag_{lag}"] = df["close"].pct_change().shift(lag)

        # Lagged indicators
        for lag in [1, 2, 3]:
            df[f"rsi_lag_{lag}"] = df["rsi_14"].shift(lag)
            df[f"macd_diff_lag_{lag}"] = df["macd_diff"].shift(lag)

        # RSI rate of change
        df["rsi_change"] = df["rsi_14"].diff()
        df["macd_diff_change"] = df["macd_diff"].diff()

        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        close = df["close"]
        returns = close.pct_change()

        # Rolling volatility
        df["volatility_5"] = returns.rolling(5).std()
        df["volatility_10"] = returns.rolling(10).std()
        df["volatility_20"] = returns.rolling(20).std()

        # Rolling mean returns
        df["mean_return_5"] = returns.rolling(5).mean()
        df["mean_return_10"] = returns.rolling(10).mean()
        df["mean_return_20"] = returns.rolling(20).mean()

        # Rolling skewness and kurtosis (distribution shape of returns)
        df["skew_20"] = returns.rolling(20).skew()
        df["kurt_20"] = returns.rolling(20).kurt()

        # Rolling min/max as a ratio of current price (support/resistance proxy)
        df["high_20_ratio"] = close / close.rolling(20).max()
        df["low_20_ratio"] = close / close.rolling(20).min()
        df["high_50_ratio"] = close / close.rolling(50).max()
        df["low_50_ratio"] = close / close.rolling(50).min()

        # Volume rolling stats
        vol = df["volume"]
        df["vol_std_10"] = vol.rolling(10).std()
        df["vol_mean_ratio_10"] = vol / (vol.rolling(10).mean() + 1e-10)

        return df

    def add_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average crossover signals as numeric features."""
        # Price relative to MAs
        df["close_above_sma7"] = (df["close"] > df["sma_7"]).astype(int)
        df["close_above_sma25"] = (df["close"] > df["sma_25"]).astype(int)
        df["close_above_ema25"] = (df["close"] > df["ema_25"]).astype(int)

        # MA crossovers
        df["sma7_above_sma25"] = (df["sma_7"] > df["sma_25"]).astype(int)
        df["ema7_above_ema25"] = (df["ema_7"] > df["ema_25"]).astype(int)

        # Distance from MAs (normalized)
        df["dist_from_sma25"] = (df["close"] - df["sma_25"]) / (df["sma_25"] + 1e-10)
        df["dist_from_sma99"] = (df["close"] - df["sma_99"]) / (df["sma_99"] + 1e-10)
        df["dist_from_bb_mid"] = (df["close"] - df["bb_middle"]) / (df["bb_middle"] + 1e-10)

        # RSI zones
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

        # MACD signal crossover
        df["macd_above_signal"] = (df["macd"] > df["macd_signal"]).astype(int)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features from the open_time timestamp."""
        if "open_time" not in df.columns:
            return df

        dt = pd.to_datetime(df["open_time"], unit="ms")
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek

        # Cyclical encoding (sine/cosine to preserve continuity)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Drop raw time features (keep only cyclical)
        df = df.drop(columns=["hour", "day_of_week"], errors="ignore")
        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary classification target with threshold filtering.
        1 = next candle closes higher by > threshold (UP)
        0 = next candle closes lower by > threshold (DOWN)
        Rows with change < threshold are dropped (noise).
        """
        future_return = (df["close"].shift(-1) - df["close"]) / df["close"]
        df["target"] = np.where(
            future_return > TARGET_THRESHOLD, 1,
            np.where(future_return < -TARGET_THRESHOLD, 0, np.nan)
        )
        return df

    def prepare_features(self) -> pd.DataFrame:
        """
        Full pipeline: load → add indicators → lag → rolling → crossovers
        → time features → create target → clean NaNs.
        Returns a clean DataFrame ready for training.
        """
        df = self.load_from_db()

        # Add time features BEFORE dropping open_time
        df = self.add_time_features(df)
        df = self.add_technical_indicators(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_crossover_signals(df)
        df = self.create_target(df)

        # Drop non-feature columns
        drop_cols = ["open_time", "close_time"]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Drop rows with NaN (from indicator warmup, lags, target shift, threshold)
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        after = len(df)

        # Ensure target is integer
        df["target"] = df["target"].astype(int)

        logging.info(f"Dropped {before - after} NaN rows. {after} rows remaining.")
        print(f"   Target distribution: UP={int((df['target']==1).sum())}, "
              f"DOWN={int((df['target']==0).sum())}")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of feature column names (excludes target)."""
        exclude = {"target"}
        return [col for col in df.columns if col not in exclude]

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Time-series aware train/test split (no shuffle).
        Returns X_train, X_test, y_train, y_test.
        """
        feature_cols = self.get_feature_columns(df)
        split_idx = int(len(df) * (1 - test_size))

        X_train = df.iloc[:split_idx][feature_cols]
        X_test = df.iloc[split_idx:][feature_cols]
        y_train = df.iloc[:split_idx]["target"]
        y_test = df.iloc[split_idx:]["target"]

        logging.info(
            f"Split: train={len(X_train)} rows, test={len(X_test)} rows"
        )
        return X_train, X_test, y_train, y_test


# Minimum candles required for indicator warmup (SMA-99 needs at least 99 rows)
MIN_CANDLES = 100

REQUIRED_CANDLE_KEYS = {"open", "high", "low", "close", "volume", "open_time"}


def prepare_features_from_candles(candles: list[dict]) -> pd.DataFrame:
    """
    Prepare features from a list of candle dicts (for prediction-time use).
    Returns a single-row DataFrame of the latest candle's features.

    Raises:
        ValueError: If candles are missing required keys or count is insufficient.
    """
    if not candles:
        raise ValueError("No candles provided.")

    # Validate required keys on first candle
    sample_keys = set(candles[0].keys())
    missing_keys = REQUIRED_CANDLE_KEYS - sample_keys
    if missing_keys:
        raise ValueError(
            f"Candle dicts are missing required keys: {sorted(missing_keys)}"
        )

    if len(candles) < MIN_CANDLES:
        raise ValueError(
            f"Need at least {MIN_CANDLES} candles for indicator warmup, "
            f"got {len(candles)}."
        )

    df = pd.DataFrame(candles)
    fe = FeatureEngineer()

    # Add time features BEFORE dropping open_time
    df = fe.add_time_features(df)
    df = fe.add_technical_indicators(df)
    df = fe.add_lag_features(df)
    df = fe.add_rolling_features(df)
    df = fe.add_crossover_signals(df)

    drop_cols = ["open_time", "close_time"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna()

    return df.iloc[[-1]] if len(df) > 0 else df
