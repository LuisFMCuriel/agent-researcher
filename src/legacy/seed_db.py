from .db import SessionLocal
from .models import Experiment


def main():
    db = SessionLocal()

    examples = [
        Experiment(exp_key="exp_0043", model_name="unet_resnet34", dataset="cityscapes",
                   dice=0.81, iou=0.72, accuracy=0.91, notes="baseline UNet + aug v1"),
        Experiment(exp_key="exp_0044", model_name="unet_resnet34", dataset="cityscapes",
                   dice=0.83, iou=0.74, accuracy=0.92, notes="lr schedule cosine"),
        Experiment(exp_key="exp_0045", model_name="swin_unet", dataset="cityscapes",
                   dice=0.85, iou=0.76, accuracy=0.93, notes="transformer backbone"),
    ]

    for e in examples:
        db.add(e)

    db.commit()
    db.close()
    print("Seeded example experiments.")


if __name__ == "__main__":
    main()