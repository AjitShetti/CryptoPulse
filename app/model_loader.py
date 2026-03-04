"""
CryptoPulse — Model Loader
Singleton loader for the trained XGBoost model, scaler, and feature names.
"""
import numpy as np
from pathlib import Path

from core.utils import load_model, load_metadata
from config.settings import MODELS_DIR
from core.logger import logging


class ModelLoader:
    """
    Loads and holds the trained model artifacts in memory.
    Provides a predict() method for inference.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        """Load model artifacts from disk."""
        try:
            self.model, self.scaler, self.feature_names, self.metadata = load_model(
                MODELS_DIR
            )
            self._loaded = True
            logging.info("Model loaded successfully into ModelLoader")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self._loaded = False
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
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

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
