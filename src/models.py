from sqlalchemy import String, Integer, Float, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
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