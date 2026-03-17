from __future__ import annotations

import json
from pathlib import Path

import yaml
from sqlalchemy import select

from .db import SessionLocal
from .models import Experiment, ExperimentArtifact

def read_text(path: Path, max_chars: int = 2_000_000) -> str:
    # Keep it safe for giant logs in a toy project
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[TRUNCATED]..."
    return text


def ingest_experiment_folder(folder: Path) -> None:
    """
    Expects:
      folder name like exp_0043
      optional: metrics.json, config.yaml, train.log
    """
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    exp_key = folder.name  # "exp_0043"

    metrics_path = folder / "metrics.json"
    config_path = folder / "config.yaml"
    log_path = folder / "train.log"

    metrics = {}
    config = {}

    if metrics_path.exists():
        metrics = json.loads(read_text(metrics_path))

    if config_path.exists():
        config = yaml.safe_load(read_text(config_path)) or {}

    # Try to extract core fields from config (fallbacks included)
    model_name = str(config.get("model_name", "unknown_model"))
    dataset = str(config.get("dataset", "unknown_dataset"))

    accuracy = metrics.get("accuracy")
    dice = metrics.get("dice")
    iou = metrics.get("iou")

    db = SessionLocal()

    # Upsert-ish behavior: if exp_key exists, update; else create
    exp = db.scalar(select(Experiment).where(Experiment.exp_key == exp_key))
    if exp is None:
        exp = Experiment(
            exp_key=exp_key,
            model_name=model_name,
            dataset=dataset,
            accuracy=accuracy,
            dice=dice,
            iou=iou,
            notes=f"Ingested from {folder.as_posix()}",
        )
        db.add(exp)
        db.flush()  # ensures exp.id is available
    else:
        exp.model_name = model_name
        exp.dataset = dataset
        exp.accuracy = accuracy
        exp.dice = dice
        exp.iou = iou

    # helper to add/update artifacts by (experiment_id, name)
    def upsert_artifact(name: str, kind: str, path: Path) -> None:
        if not path.exists():
            return
        existing = db.scalar(
            select(ExperimentArtifact).where(
                ExperimentArtifact.experiment_id == exp.id,
                ExperimentArtifact.name == name,
            )
        )
        content = read_text(path)
        if existing is None:
            db.add(
                ExperimentArtifact(
                    experiment_id=exp.id,
                    name=name,
                    kind=kind,
                    path=str(path),
                    content=content,
                )
            )
        else:
            existing.kind = kind
            existing.path = str(path)
            existing.content = content

    upsert_artifact("metrics.json", "json", metrics_path)
    upsert_artifact("config.yaml", "yaml", config_path)
    upsert_artifact("train.log", "log", log_path)

    db.commit()
    db.close()

    print(f"Ingested {exp_key} (model={model_name}, dataset={dataset}).")


def main():
    # Example usage: python -m src.ingest_experiment data/exp_0043
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.ingest_experiment <path-to-experiment-folder>")

    folder = Path(sys.argv[1]).resolve()
    ingest_experiment_folder(folder)


if __name__ == "__main__":
    main()