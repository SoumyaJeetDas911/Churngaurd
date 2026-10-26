"""Pydantic request and response models for the REST API."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]
    customer_label: str | None = Field(default=None, max_length=128)

    @field_validator("features")
    @classmethod
    def features_size_limited(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, default=str)) > 64_000:
            raise ValueError("Feature payload must not exceed 64 KB.")
        return value


class BatchPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[PredictionInput] = Field(min_length=1)


class PredictionResult(BaseModel):
    prediction_id: int | None = None
    probability: float = Field(ge=0, le=1)
    predicted_class: str
    risk_category: str
    threshold: float
    model_version: str
    explanation: dict[str, Any] | None = None
    explanation_warning: str | None = None
    retention_suggestions: list[str]
    history_saved: bool
    history_warning: str | None = None


class BatchPredictionResult(BaseModel):
    results: list[PredictionResult]
    count: int


class HistoryRecord(BaseModel):
    id: int
    timestamp: datetime
    customer_label: str | None
    probability: float
    predicted_class: str
    risk_category: str
    threshold: float
    model_version: str
    customer_features: dict[str, Any]


class HistoryPage(BaseModel):
    items: list[HistoryRecord]
    total: int
    limit: int
    offset: int
