"""
CryptoPulse — Database Initialization Script
Creates all tables in the SQLite database.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import create_tables
from core.logger import logging


def main():
    print("=" * 50)
    print("  CryptoPulse — Database Initialization")
    print("=" * 50)

    try:
        create_tables()
        print("✅ Database tables created successfully!")
        logging.info("Database initialized via init_db.py")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        logging.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
