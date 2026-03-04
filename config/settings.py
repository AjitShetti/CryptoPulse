"""
CryptoPulse — Centralized Configuration
Loads settings from .env file with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


# ── Database ────────────────────────────────────────────────────────────
DATABASE_PATH = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "cryptopulse.db"))
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


# ── Binance API ─────────────────────────────────────────────────────────
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1h")
FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "1000"))


# ── Model ───────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))


# ── FastAPI ─────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


# ── Streamlit Dashboard ────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "60"))  # seconds
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
