from sqlalchemy import String, Integer, Float, DateTime, Text, func, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Human-friendly identifier like "exp_0043"
    exp_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)

    model_name: Mapped[str] = mapped_column(String(128), index=True)  # e.g., "unet_resnet34"
    dataset: Mapped[str] = mapped_column(String(128), index=True)     # e.g., "cityscapes"
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    dice: Mapped[float | None] = mapped_column(Float, nullable=True)
    iou: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

class ExperimentArtifact(Base):
    __tablename__ = "experiment_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)

    # e.g. "metrics.json", "config.yaml", "train.log"
    name: Mapped[str] = mapped_column(String(128), index=True)

    # e.g. "json", "yaml", "log", "text"
    kind: Mapped[str] = mapped_column(String(32), index=True)

    # original path on disk (useful for traceability)
    path: Mapped[str] = mapped_column(String(512))

    # raw contents (toy version: store it directly in DB)
    content: Mapped[str] = mapped_column(Text)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    experiment: Mapped["Experiment"] = relationship(backref="artifacts")

class TextChunk(Base):
    __tablename__ = "text_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    artifact_id: Mapped[int] = mapped_column(ForeignKey("experiment_artifacts.id"), index=True)

    chunk_index: Mapped[int] = mapped_column(Integer)  # 0,1,2...
    content: Mapped[str] = mapped_column(Text)

    # We store the FAISS vector position for this chunk
    faiss_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())