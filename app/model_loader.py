"""
CryptoPulse — Model Loader
Thread-safe singleton loader for the trained XGBoost model, scaler, and feature names.
"""
import threading
import numpy as np
from pathlib import Path

from core.utils import load_model, load_metadata
from config.settings import MODELS_DIR
from core.logger import logging


class ModelLoader:
    """
    Loads and holds the trained model artifacts in memory.
    Provides a predict() method for inference.
    Thread-safe singleton via double-checked locking.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = False
                    cls._instance._load_lock = threading.Lock()
        return cls._instance

    def load(self):
        """Load model artifacts from disk (thread-safe)."""
        with self._load_lock:
            try:
                self.model, self.scaler, self.feature_names, self.metadata = load_model(
                    MODELS_DIR
                )
                self._loaded = True
                logging.info("Model loaded successfully into ModelLoader")
            except Exception as e:
                self._loaded = False
                logging.error(f"Failed to load model: {e}")
                raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: dict) -> tuple[str, float]:
        """
        Make a prediction from a features dictionary.

        Args:
            features: Dict mapping feature_name → value.

        Returns:
            Tuple of (prediction_label: "UP"/"DOWN", confidence: float).

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If required features are missing.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Check for missing features
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            logging.warning(
                f"Missing {len(missing)} features (will default to 0): {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )

        # Build feature array in correct order
        feature_array = np.array(
            [[features.get(name, 0) for name in self.feature_names]]
        )

        # Scale
        feature_array_scaled = self.scaler.transform(feature_array)

        # Predict
        prediction = self.model.predict(feature_array_scaled)[0]
        probabilities = self.model.predict_proba(feature_array_scaled)[0]

        label = "UP" if prediction == 1 else "DOWN"
        confidence = float(probabilities[prediction])

        logging.info(f"Prediction: {label} (confidence={confidence:.4f})")
        return label, confidence

    def get_metadata(self) -> dict:
        """Return model metadata."""
        if not self._loaded:
            raise RuntimeError("Model not loaded.")
        return self.metadata


# Global instance
model_loader = ModelLoader()
