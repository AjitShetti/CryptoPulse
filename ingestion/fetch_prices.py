"""
CryptoPulse — Binance Data Ingestion
Fetches historical OHLCV kline data from the Binance public API.
"""
import requests
from typing import Optional

from config.settings import BINANCE_BASE_URL, SYMBOL, INTERVAL, FETCH_LIMIT
from core.logger import logging
from core.custonException import CustomException
import sys


class BinanceFetcher:
    """Fetches cryptocurrency kline (candlestick) data from Binance public API."""

    KLINES_ENDPOINT = "/api/v3/klines"

    def __init__(
        self,
        symbol: str = SYMBOL,
        interval: str = INTERVAL,
        base_url: str = BINANCE_BASE_URL,
    ):
        self.symbol = symbol
        self.interval = interval
        self.base_url = base_url
        logging.info(
            f"BinanceFetcher initialized: symbol={symbol}, interval={interval}"
        )

    def fetch_klines(
        self,
        limit: int = FETCH_LIMIT,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch kline/candlestick data from Binance.

        Args:
            limit: Number of candles to fetch (max 1000 per request).
            start_time: Start time in milliseconds (optional).
            end_time: End time in milliseconds (optional).

        Returns:
            List of candle dicts with keys: open_time, open, high, low, close,
            volume, close_time, quote_volume, trades.
        """
        try:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": min(limit, 1000),
            }
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time

            url = f"{self.base_url}{self.KLINES_ENDPOINT}"
            logging.info(f"Fetching klines: {url} | params={params}")

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            raw_data = response.json()
            candles = self._parse_klines(raw_data)

            logging.info(f"Fetched {len(candles)} candles for {self.symbol}")
            return candles

        except requests.RequestException as e:
            logging.error(f"Binance API request failed: {e}")
            raise CustomException(f"Binance API request failed: {e}", sys)

    def fetch_all_klines(self, total: int = FETCH_LIMIT) -> list[dict]:
        """
        Fetch more than 1000 candles by paginating through the API.

        Args:
            total: Total number of candles to fetch.

        Returns:
            List of all candle dicts.
        """
        all_candles = []
        remaining = total
        end_time = None

        while remaining > 0:
            batch_size = min(remaining, 1000)
            candles = self.fetch_klines(limit=batch_size, end_time=end_time)

            if not candles:
                break

            all_candles = candles + all_candles  # prepend (older data first)
            end_time = candles[0]["open_time"] - 1  # move window backwards
            remaining -= len(candles)

            logging.info(
                f"Pagination: fetched {len(candles)}, total so far: {len(all_candles)}"
            )

        return all_candles

    def fetch_latest_candle(self) -> Optional[dict]:
        """Fetch the most recent completed candle. Returns None if unavailable."""
        candles = self.fetch_klines(limit=2)
        if len(candles) >= 2:
            return candles[-2]  # second-to-last is the latest COMPLETED candle
        return candles[0] if candles else None

    @staticmethod
    def _parse_klines(raw_data: list) -> list[dict]:
        """
        Parse raw Binance kline response into structured dicts.

        Binance kline format:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        candles = []
        for k in raw_data:
            candles.append(
                {
                    "open_time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                    "quote_volume": float(k[7]),
                    "trades": int(k[8]),
                }
            )
        return candles

    def get_current_price(self) -> float:
        """Fetch the current ticker price for the symbol."""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = requests.get(
                url, params={"symbol": self.symbol}, timeout=10
            )
            response.raise_for_status()
            return float(response.json()["price"])
        except requests.RequestException as e:
            logging.error(f"Failed to fetch current price: {e}")
            raise CustomException(f"Failed to fetch current price: {e}", sys)
