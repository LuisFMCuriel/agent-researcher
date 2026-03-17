from __future__ import annotations

from sqlalchemy import delete

from .db import SessionLocal
from .models import Base, Experiment, ExperimentArtifact
from .db import engine


def build_synthetic_experiments():
    return [
        {
            "exp_key": "exp_001",
            "dataset": "CIFAR-10",
            "model_name": "ResNet18",
            "accuracy": 0.842,
            "dice": 0.0,
            "iou": 0.0,
            "notes": "Baseline run. Stable training but moderate overfitting reduced validation performance.",
            "lr": 0.001,
            "augmentation": "none",
            "conclusion": "Baseline run. Stable training but moderate overfitting reduced validation performance."
        },
        {
            "exp_key": "exp_002",
            "dataset": "CIFAR-10",
            "model_name": "ResNet18",
            "accuracy": 0.871,
            "dice": 0.0,
            "iou": 0.0,
            "notes": "Data augmentation improved generalization and reduced overfitting relative to the baseline.",
            "lr": 0.001,
            "augmentation": "random crop + horizontal flip",
            "conclusion": "Data augmentation improved generalization and reduced overfitting relative to the baseline."
        },
    ]


def build_full_report(item: dict) -> str:
    return f"""Experiment summary for {item['exp_key']}.
Experiment key: {item['exp_key']}
Dataset: {item['dataset']}
Model name: {item['model_name']}
Learning rate: {item['lr']}
Augmentation: {item['augmentation']}

Metrics report for {item['exp_key']}.
Accuracy: {item['accuracy']:.3f}
Dice: {item['dice']:.3f}
IoU: {item['iou']:.3f}
Final validation accuracy for experiment {item['exp_key']} is {item['accuracy']:.3f}.

Training notes for {item['exp_key']}.
Model {item['model_name']} was trained on {item['dataset']} with learning rate {item['lr']}.
Augmentation strategy used: {item['augmentation']}.
Observed behavior: training was stable, and validation performance depended strongly on augmentation and model capacity.

Conclusion for {item['exp_key']}.
{item['conclusion']}
"""


def main():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    db.execute(delete(ExperimentArtifact))
    db.execute(delete(Experiment))
    db.commit()

    synthetic = build_synthetic_experiments()

    for item in synthetic:
        exp = Experiment(
            exp_key=item["exp_key"],
            model_name=item["model_name"],
            dataset=item["dataset"],
            notes=item["notes"],
            accuracy=item["accuracy"],
            dice=item["dice"],
            iou=item["iou"],
        )
        db.add(exp)
        db.flush()

        artifact = ExperimentArtifact(
            experiment_id=exp.id,
            name="synthetic_report.txt",
            kind="text",
            path=f"/synthetic/{item['exp_key']}.txt",
            content=build_full_report(item),
        )
        db.add(artifact)

    db.commit()
    db.close()

    print("Synthetic experiments and artifacts inserted.")
    print("Now run: python -m src.build_faiss --rebuild")


if __name__ == "__main__":
    main()