"""Entry point to fine-tune T5 for headline generation."""

import pandas as pd

from src.t5.config import (
    MODEL_NAME,
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE,
    MAX_SOURCE_TEXT_LENGTH,
    MAX_TARGET_TEXT_LENGTH,
    SEED,
    DATA_PATH,
    OUTPUT_DIR,
)
from src.t5.train.trainer import T5Trainer


def main():
    """Load data and launch T5 fine-tuning."""
    df = pd.read_csv(DATA_PATH)

    model_params = {
        "MODEL": MODEL_NAME,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "VALID_BATCH_SIZE": VALID_BATCH_SIZE,
        "TRAIN_EPOCHS": TRAIN_EPOCHS,
        "VAL_EPOCHS": 1,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_SOURCE_TEXT_LENGTH": MAX_SOURCE_TEXT_LENGTH,
        "MAX_TARGET_TEXT_LENGTH": MAX_TARGET_TEXT_LENGTH,
        "SEED": SEED,
    }

    T5Trainer(
        df[:5000],
        source_text="Description",
        target_text="Headlines",
        model_params=model_params,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
