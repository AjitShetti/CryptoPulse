"""
CryptoPulse — Core Utilities
Helper functions for model persistence and project paths.
"""
import json
import joblib
from pathlib import Path
from datetime import datetime

from core.logger import logging


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def save_model(model, scaler, feature_names: list, metrics: dict, models_dir: Path):
    """
    Persist trained model, scaler, feature names, and metadata to disk.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(feature_names, models_dir / "feature_names.joblib")

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Model artifacts saved to {models_dir}")


def load_model(models_dir: Path):
    """
    Load model, scaler, feature names, and metadata from disk.
    Returns (model, scaler, feature_names, metadata).
    """
    model = joblib.load(models_dir / "model.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")

    with open(models_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    logging.info(f"Model artifacts loaded from {models_dir}")
    return model, scaler, feature_names, metadata


def load_metadata(models_dir: Path) -> dict:
    """Load only the metadata JSON."""
    with open(models_dir / "metadata.json", "r") as f:
        return json.load(f)
