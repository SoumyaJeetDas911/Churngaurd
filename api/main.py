"""FastAPI application for ChurnGuard predictions and history."""

from __future__ import annotations

import logging
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.database import (
    analytics_summary,
    build_engine,
    get_prediction,
    get_session_factory,
    initialize_database,
    list_predictions,
)
from api.schemas import (
    BatchPredictionInput,
    BatchPredictionResult,
    HistoryPage,
    HistoryRecord,
    PredictionInput,
    PredictionResult,
)
from api.services import create_prediction
from src.config import MAX_BATCH_SIZE, PROJECT_ROOT
from src.predict import ChurnPredictor, PredictionValidationError


logger = logging.getLogger("churnguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.engine = build_engine()
    app.state.session_factory = get_session_factory(app.state.engine)
    app.state.history_ready = False
    try:
        initialize_database(app.state.engine)
        app.state.history_ready = True
    except (SQLAlchemyError, OSError):
        logger.exception("Prediction history is unavailable; predictions remain enabled.")
    try:
        app.state.predictor = ChurnPredictor()
    except (FileNotFoundError, ValueError, OSError, ModuleNotFoundError, EOFError):
        logger.exception("Model artifact could not be loaded.")
    yield
    app.state.engine.dispose()


app = FastAPI(
    title="ChurnGuard API",
    description="Local customer churn risk scoring and prediction history.",
    version="1.0.0",
    lifespan=lifespan,
)


def _predictor(request: Request) -> ChurnPredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Train the model with `python -m src.train`.",
        )
    return predictor


def _session(request: Request) -> Session | None:
    if not request.app.state.history_ready:
        return None
    return request.app.state.session_factory()


def _as_result(prediction: dict[str, Any], record, warning: str | None) -> dict[str, Any]:
    return {
        **prediction,
        "prediction_id": record.id if record is not None else None,
        "history_saved": record is not None,
        "history_warning": warning if record is None else None,
    }


@app.get("/health")
def health(request: Request):
    predictor = request.app.state.predictor
    result = {
        "status": "ready" if predictor is not None else "model_unavailable",
        "model_ready": predictor is not None,
        "history_ready": request.app.state.history_ready,
    }
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result
        )
    return result


@app.get("/model/info")
def model_info(request: Request):
    predictor = _predictor(request)
    metadata = predictor.metadata
    return {
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "trained_at": metadata["trained_at"],
        "selection_metric": metadata["selection_metric"],
        "validation_roc_auc": metadata["validation_roc_auc"],
        "decision_threshold": predictor.threshold,
        "training_metadata": metadata,
    }


@app.get("/metrics")
def model_metrics(request: Request):
    return _predictor(request).metrics


@app.get("/features")
def features(request: Request):
    predictor = _predictor(request)
    return {
        "features": [
            {"name": name, **spec}
            for name, spec in predictor.feature_schema.items()
        ],
        "target_column": predictor.metadata["data"]["target_column"],
        "excluded_columns": predictor.metadata["data"]["excluded_columns"],
    }


@app.post("/predict", response_model=PredictionResult)
def predict(body: PredictionInput, request: Request):
    predictor = _predictor(request)
    session = _session(request)
    try:
        try:
            prediction, record, warning = create_prediction(
                predictor, session, body.features, body.customer_label
            )
        except PredictionValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _as_result(prediction, record, warning)
    finally:
        if session is not None:
            session.close()


@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(body: BatchPredictionInput, request: Request):
    if len(body.records) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch exceeds the configured limit of {MAX_BATCH_SIZE} records.",
        )
    predictor = _predictor(request)
    for index, item in enumerate(body.records):
        try:
            predictor.validate_features(item.features)
        except PredictionValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Record {index}: {exc}") from exc

    results = []
    for item in body.records:
        session = _session(request)
        try:
            try:
                prediction, record, warning = create_prediction(
                    predictor, session, item.features, item.customer_label
                )
            except PredictionValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            results.append(_as_result(prediction, record, warning))
        finally:
            if session is not None:
                session.close()
    return {"results": results, "count": len(results)}


@app.get("/history", response_model=HistoryPage)
def history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    if not request.app.state.history_ready:
        raise HTTPException(status_code=503, detail="Prediction history is unavailable.")
    session = _session(request)
    try:
        records, total = list_predictions(session, limit, offset)
        return {"items": records, "total": total, "limit": limit, "offset": offset}
    except SQLAlchemyError as exc:
        logger.exception("Failed to read prediction history.")
        raise HTTPException(status_code=503, detail="Prediction history query failed.") from exc
    finally:
        session.close()


@app.get("/history/{prediction_id}", response_model=HistoryRecord)
def history_detail(prediction_id: int, request: Request):
    if not request.app.state.history_ready:
        raise HTTPException(status_code=503, detail="Prediction history is unavailable.")
    session = _session(request)
    try:
        record = get_prediction(session, prediction_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Prediction record not found.")
        return record
    except SQLAlchemyError as exc:
        logger.exception("Failed to read prediction history record.")
        raise HTTPException(status_code=503, detail="Prediction history query failed.") from exc
    finally:
        session.close()


@app.get("/analytics/summary")
def analytics(request: Request):
    if not request.app.state.history_ready:
        raise HTTPException(status_code=503, detail="Prediction history is unavailable.")
    session = _session(request)
    try:
        return analytics_summary(session)
    except SQLAlchemyError as exc:
        logger.exception("Failed to aggregate prediction history.")
        raise HTTPException(status_code=503, detail="Prediction analytics unavailable.") from exc
    finally:
        session.close()


@app.get("/analytics/training")
def training_analytics():
    profile_path = PROJECT_ROOT / "artifacts" / "eda" / "data_profile.json"
    if not profile_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training-data profile unavailable. Run `python -m src.inspect_data`.",
        )
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Training data profile could not be loaded.")
        raise HTTPException(
            status_code=503, detail="Training-data profile is unreadable."
        ) from exc
    return {
        "descriptive_observations": profile.get("descriptive_observations", []),
        "segment_churn": profile.get("segment_churn", {}),
        "shape": profile.get("shape"),
        "target_counts": profile.get("target_counts"),
    }