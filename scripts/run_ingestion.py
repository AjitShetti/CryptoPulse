"""
CryptoPulse — Ingestion Runner
Fetches price data from Binance and stores it in the database.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import create_tables, get_session, bulk_upsert_candles
from ingestion.fetch_prices import BinanceFetcher
from config.settings import SYMBOL, INTERVAL, FETCH_LIMIT
from core.logger import logging


def main():
    print("=" * 50)
    print("  CryptoPulse — Data Ingestion")
    print("=" * 50)
    print(f"  Symbol:   {SYMBOL}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Limit:    {FETCH_LIMIT}")
    print("=" * 50)

    try:
        # 1. Ensure DB tables exist
        create_tables()

        # 2. Fetch candles from Binance
        fetcher = BinanceFetcher()
        print(f"\n📡 Fetching {FETCH_LIMIT} candles from Binance...")
        candles = fetcher.fetch_all_klines(total=FETCH_LIMIT)
        print(f"✅ Fetched {len(candles)} candles")

        # 3. Store in database
        session = get_session()
        try:
            print(f"\n💾 Storing candles in database...")
            inserted, skipped = bulk_upsert_candles(session, SYMBOL, candles)
            print(f"✅ Inserted: {inserted} | Skipped (duplicates): {skipped}")
        finally:
            session.close()

        # 4. Verify
        session = get_session()
        try:
            from data.db import PriceHistory

            total = session.query(PriceHistory).filter_by(symbol=SYMBOL).count()
            print(f"\n📊 Total records in DB for {SYMBOL}: {total}")
        finally:
            session.close()

        print("\n🎉 Ingestion complete!")
        logging.info(f"Ingestion complete: {inserted} new, {skipped} skipped, {total} total")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        logging.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
