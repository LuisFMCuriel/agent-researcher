from .db import engine, Base
from . import models  # noqa: F401 (ensures models are imported)


def main():
    Base.metadata.create_all(bind=engine)
    print("DB initialized: experiments.db created (if it didn't exist).")


if __name__ == "__main__":
    main()