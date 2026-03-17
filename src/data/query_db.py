from sqlalchemy import select, desc
from .db import SessionLocal
from .models import Experiment


def main():
    db = SessionLocal()

    # Get one experiment
    exp_key = "exp_0043"
    exp = db.scalar(select(Experiment).where(Experiment.exp_key == exp_key))
    print("\n=== EXPERIMENT LOOKUP ===")
    print(exp_key, "->", None if exp is None else {
        "model": exp.model_name,
        "dataset": exp.dataset,
        "dice": exp.dice,
        "iou": exp.iou,
        "acc": exp.accuracy,
        "notes": exp.notes,
    })

    # Best dice
    best = db.scalar(select(Experiment).order_by(desc(Experiment.dice)))
    print("\n=== BEST BY DICE ===")
    print(None if best is None else (best.exp_key, best.model_name, best.dice))

    db.close()


if __name__ == "__main__":
    main()