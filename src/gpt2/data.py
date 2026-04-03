"""Data loading and splitting utilities for GPT-2 fine-tuning."""

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.gpt2.config import BATCH_SIZE, MAX_LENGTH, SEED, DATA_PATH
from src.gpt2.dataset.custom_dataset import CustomDataset


def load_data(path=DATA_PATH):
    """Load and clean the headlines CSV.

    Args:
        path: Path to the CSV file. Defaults to DATA_PATH from config.

    Returns:
        A cleaned pandas DataFrame with the 'Time' column dropped and
        duplicate rows removed.
    """
    df = pd.read_csv(path, delimiter=",")
    df.drop("Time", inplace=True, axis=1)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def split_data(df):
    """Split a DataFrame of headlines into train and validation lists.

    Args:
        df: DataFrame containing a 'Headlines' column.

    Returns:
        A tuple of (train_headlines, val_headlines) as Python lists.
    """
    df_train, df_val = train_test_split(df, test_size=0.20, random_state=SEED)

    train_headlines = df_train["Headlines"].reset_index(drop=True).tolist()
    val_headlines = df_val["Headlines"].reset_index(drop=True).tolist()

    return train_headlines, val_headlines


def build_dataloaders(train_headlines, val_headlines, tokenizer):
    """Build train and validation DataLoaders from headline lists.

    Args:
        train_headlines: Training portion of 'Headlines' text data.
        val_headlines: Validation portion of 'Headlines' text data.
        tokenizer: GPT2Tokenizer from Hugging Face.

    Returns:
        A tuple of (train_loader, val_loader) as PyTorch DataLoaders.
    """
    train_set = CustomDataset(train_headlines, tokenizer, MAX_LENGTH)
    val_set = CustomDataset(val_headlines, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_set.collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_set.collate_fn
    )

    return train_loader, val_loader
