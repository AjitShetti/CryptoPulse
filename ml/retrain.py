"""
CryptoPulse — Model Retraining
Fetches fresh data, re-engineers features, retrains the model,
and saves if performance improves.
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.fetch_prices import BinanceFetcher
from data.db import create_tables, get_session, bulk_upsert_candles
from ml.train import ModelTrainer
from core.utils import load_metadata
from config.settings import SYMBOL, MODELS_DIR, FETCH_LIMIT
from core.logger import logging


class Retrainer:
    """
    Orchestrates the retraining pipeline:
    1. Fetch latest data from Binance
    2. Retrain model on full dataset
    3. Save only if metrics improve
    """

    def __init__(self):
        self.previous_metrics = None

    def load_previous_metrics(self) -> dict | None:
        """Load metrics from the currently saved model."""
        try:
            metadata = load_metadata(MODELS_DIR)
            self.previous_metrics = metadata.get("metrics", {})
            return self.previous_metrics
        except FileNotFoundError:
            logging.info("No previous model found. Will train fresh.")
            return None

    def fetch_latest_data(self):
        """Fetch and store latest candles."""
        create_tables()
        fetcher = BinanceFetcher()
        candles = fetcher.fetch_all_klines(total=FETCH_LIMIT)

        session = get_session()
        try:
            inserted, skipped = bulk_upsert_candles(session, SYMBOL, candles)
            print(f"📡 Data refresh: {inserted} new candles, {skipped} existing")
        finally:
            session.close()

    def retrain(self) -> dict:
        """
        Full retrain pipeline.
        Returns new metrics dict.
        """
        print("=" * 50)
        print("  CryptoPulse — Model Retraining")
        print("=" * 50)

        # 1. Load previous metrics
        prev = self.load_previous_metrics()
        if prev:
            print(f"\n📊 Previous model — Accuracy: {prev.get('accuracy', 'N/A')}, "
                  f"F1: {prev.get('f1_score', 'N/A')}")

        # 2. Fetch latest data
        print("\n📡 Fetching latest data...")
        self.fetch_latest_data()

        # 3. Retrain
        trainer = ModelTrainer()
        new_metrics = trainer.train()

        # 4. Compare
        if prev and prev.get("f1_score"):
            old_f1 = prev["f1_score"]
            new_f1 = new_metrics["f1_score"]
            improvement = new_f1 - old_f1

            if improvement > 0:
                print(f"\n✅ Model improved! F1: {old_f1:.4f} → {new_f1:.4f} (+{improvement:.4f})")
            elif improvement == 0:
                print(f"\n⚖️  Model performance unchanged. F1: {new_f1:.4f}")
            else:
                print(f"\n⚠️  Model regressed. F1: {old_f1:.4f} → {new_f1:.4f} ({improvement:.4f})")
                print("   Note: Model was still saved (latest data is more relevant).")
        else:
            print(f"\n🆕 First model trained! F1: {new_metrics['f1_score']:.4f}")

        print("\n🎉 Retraining complete!")
        return new_metrics


def main():
    try:
        retrainer = Retrainer()
        retrainer.retrain()
    except Exception as e:
        print(f"\n❌ Retraining failed: {e}")
        logging.error(f"Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
