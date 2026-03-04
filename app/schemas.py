"""
CryptoPulse — Pydantic Schemas
Request/response models for the FastAPI prediction API.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request body for /predict endpoint."""
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    interval: str = Field(default="1h", description="Candle interval")


class PredictionResponse(BaseModel):
    """Response body for /predict endpoint."""
    symbol: str
    prediction: str  # "UP" or "DOWN"
    confidence: float
    current_price: float
    timestamp: str
    features_used: int


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response body for /model-info endpoint."""
    trained_at: str
    feature_count: int
    feature_names: list[str]
    metrics: dict
    model_type: str = "XGBClassifier"


class RetrainResponse(BaseModel):
    """Response body for /retrain endpoint."""
    status: str
    metrics: dict
    timestamp: str
