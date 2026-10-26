"""SQLite-backed prediction history."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String, create_engine, func, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import DATABASE_URL, PROJECT_ROOT


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    customer_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    risk_category: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


def build_engine(database_url: str = DATABASE_URL) -> Engine:
    url = make_url(database_url)
    connect_args: dict[str, Any] = {}
    if url.get_backend_name() == "sqlite":
        database = url.database
        if database and database != ":memory:":
            database_path = Path(database)
            if not database_path.is_absolute():
                database_path = PROJECT_ROOT / database_path
            database_path.parent.mkdir(parents=True, exist_ok=True)
            url = url.set(database=str(database_path))
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def initialize_database(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def add_prediction(session: Session, values: dict[str, Any]) -> PredictionRecord:
    record = PredictionRecord(**values)
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def list_predictions(session: Session, limit: int, offset: int) -> tuple[list[PredictionRecord], int]:
    total = int(session.scalar(select(func.count()).select_from(PredictionRecord)) or 0)
    records = list(
        session.scalars(
            select(PredictionRecord)
            .order_by(PredictionRecord.timestamp.desc(), PredictionRecord.id.desc())
            .offset(offset)
            .limit(limit)
        )
    )
    return records, total


def get_prediction(session: Session, prediction_id: int) -> PredictionRecord | None:
    return session.get(PredictionRecord, prediction_id)


def analytics_summary(session: Session) -> dict[str, Any]:
    total = int(session.scalar(select(func.count()).select_from(PredictionRecord)) or 0)
    risks = {
        risk: int(count)
        for risk, count in session.execute(
            select(PredictionRecord.risk_category, func.count()).group_by(
                PredictionRecord.risk_category
            )
        )
    }
    classes = {
        label: int(count)
        for label, count in session.execute(
            select(PredictionRecord.predicted_class, func.count()).group_by(
                PredictionRecord.predicted_class
            )
        )
    }
    average_probability = session.scalar(select(func.avg(PredictionRecord.probability)))
    return {
        "total_predictions": total,
        "risk_counts": risks,
        "predicted_class_counts": classes,
        "average_probability": (
            float(average_probability) if average_probability is not None else None
        ),
    }
