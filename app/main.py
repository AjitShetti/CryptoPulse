"""
CryptoPulse — FastAPI Prediction API
Serves real-time cryptocurrency price direction predictions.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.model_loader import model_loader
from ingestion.fetch_prices import BinanceFetcher
from data.processing import prepare_features_from_candles
from config.settings import API_HOST, API_PORT
from core.logger import logging


# ── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="CryptoPulse API",
    description="Real-time cryptocurrency price direction prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Event ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load the trained model on application startup."""
    try:
        model_loader.load()
        logging.info("Model loaded on startup")
    except Exception as e:
        logging.warning(f"Could not load model on startup: {e}")
        logging.warning("API will start but /predict will be unavailable")


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about the currently loaded model."""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = model_loader.get_metadata()
    return ModelInfoResponse(
        trained_at=metadata.get("trained_at", "unknown"),
        feature_count=metadata.get("feature_count", 0),
        feature_names=metadata.get("feature_names", []),
        metrics=metadata.get("metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest = PredictionRequest()):
    """
    Make a price direction prediction.

    Fetches the latest candles from Binance, engineers features,
    and runs the model to predict UP or DOWN.
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        # 1. Fetch recent candles for feature calculation
        fetcher = BinanceFetcher(symbol=request.symbol, interval=request.interval)
        candles = fetcher.fetch_klines(limit=200)  # enough for indicator warmup

        if not candles:
            raise HTTPException(status_code=502, detail="Could not fetch candle data from Binance")

        # 2. Get current price
        current_price = candles[-1]["close"]

        # 3. Engineer features from candles
        features_df = prepare_features_from_candles(candles)

        if features_df.empty:
            raise HTTPException(status_code=500, detail="Feature engineering produced no data")

        # 4. Convert to dict for prediction
        features_dict = features_df.iloc[0].to_dict()

        # 5. Predict
        prediction, confidence = model_loader.predict(features_dict)

        return PredictionResponse(
            symbol=request.symbol,
            prediction=prediction,
            confidence=round(confidence, 4),
            current_price=current_price,
            timestamp=datetime.now().isoformat(),
            features_used=len(model_loader.feature_names),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Run directly ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
