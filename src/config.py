"""Runtime configuration shared by the API, training code, and dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_ARTIFACT_PATH = Path(
    os.getenv("MODEL_ARTIFACT_PATH", str(PROJECT_ROOT / "models" / "churnguard.joblib"))
)
if not MODEL_ARTIFACT_PATH.is_absolute():
    MODEL_ARTIFACT_PATH = PROJECT_ROOT / MODEL_ARTIFACT_PATH

METRICS_PATH = MODEL_ARTIFACT_PATH.parent / "metrics.json"
METADATA_PATH = MODEL_ARTIFACT_PATH.parent / "metadata.json"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./artifacts/churnguard.db")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

if not 0 < DECISION_THRESHOLD < 1:
    raise ValueError("DECISION_THRESHOLD must be between 0 and 1.")
if not 1 <= MAX_BATCH_SIZE <= 5000:
    raise ValueError("MAX_BATCH_SIZE must be between 1 and 5000.")
