"""
CryptoPulse — Database Layer
SQLAlchemy ORM model for price history and DB session management.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    BigInteger,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DATABASE_URL, DATABASE_PATH
from core.logger import logging

import os

# Ensure data directory exists (guard against empty dirname for bare filenames)
_db_dir = os.path.dirname(DATABASE_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

# ── Engine & Session ────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


# ── ORM Model ──────────────────────────────────────────────────────────
class PriceHistory(Base):
    """OHLCV candle data from Binance."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    open_time = Column(BigInteger, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_time = Column(BigInteger, nullable=False)
    quote_volume = Column(Float, nullable=False)
    trades = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("symbol", "open_time", name="uix_symbol_open_time"),
    )

    def __repr__(self):
        return f"<PriceHistory(symbol={self.symbol}, open_time={self.open_time}, close={self.close})>"


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(engine)
    logging.info("Database tables created successfully.")


def get_session():
    """Get a new database session."""
    return SessionLocal()


def bulk_upsert_candles(session, symbol: str, candles: list[dict]):
    """
    Insert candles, skipping duplicates based on (symbol, open_time).
    Uses a bulk existence check to avoid N+1 queries.
    """
    if not candles:
        return 0, 0

    # Bulk existence check: get all open_times already in DB for this symbol
    all_open_times = [c["open_time"] for c in candles]
    existing_times = set()

    # Query in batches of 500 to avoid SQLite variable limits
    for i in range(0, len(all_open_times), 500):
        batch = all_open_times[i : i + 500]
        rows = (
            session.query(PriceHistory.open_time)
            .filter(PriceHistory.symbol == symbol, PriceHistory.open_time.in_(batch))
            .all()
        )
        existing_times.update(r[0] for r in rows)

    inserted = 0
    skipped = 0

    for candle in candles:
        if candle["open_time"] in existing_times:
            skipped += 1
            continue

        record = PriceHistory(symbol=symbol, **candle)
        session.add(record)
        inserted += 1

    session.commit()
    logging.info(f"Bulk upsert: {inserted} inserted, {skipped} skipped for {symbol}")
    return inserted, skipped
