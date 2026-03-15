"""Helpers Module"""

import os

import pandas as pd
import numpy as np


import torch
from torch.utils.data import DataLoader


from transformers import GPT2Tokenizer, GPT2LMHeadModel


from src.gpt2.config import (
    BATCH_SIZE,
    MODEL_NAME,
    MAX_LENGTH,
    SEED,
    DATA_PATH,
    OUTPUT_DIR,
    DEVICE,
)


from src.gpt2.dataset.custom_dataset import CustomDataset


from sklearn.model_selection import train_test_split

# --- Load Data ---


def load_data(path=DATA_PATH):
    """Load data"""
    df = pd.read_csv(path, delimiter=",")
    df.drop("Time", inplace=True, axis=1)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --- Split Data ---
def split_data(df):
    """Split dataframe into training and testing split"""

    df_train, df_val = train_test_split(df, test_size=0.20, random_state=SEED)

    train_headlines = df_train["Headlines"].reset_index(drop=True).tolist()
    val_headlines = df_val["Headlines"].reset_index(drop=True).tolist()

    return train_headlines, val_headlines


# --- Set Seed ---


def set_seed():
    """Set Seed for project"""
    torch.manual_seed(SEED)
    np.random.rand(SEED)

    torch.backends.cudnn.deterministic = True


# --Build Model --


def build_model(device=DEVICE):
    """Build model"""

    # load gpt2 tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # add pad token

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    # gpt2 model

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # add pad token to model
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    return tokenizer, model


# --- Create Dataloaders ---


def build_dataloaders(train_headlines, val_headlines, tokenizer):
    """ "Builds dataloaders from CustomDataSet class.

    Args:
        train_headlines: Training portion of 'headlines' text data.
        val_headlines: Validation portion of 'headlines' text_data.
        tokenizer: GPT2Tokenizer from Hugging Face.

    Returns:
        train_loader:
            - Training portion of data with torch.utils.data.DataLoader
            wrapper.

        val_loader:
            - Validation portion of data with torch.utils.data.DataLoader
            wrapper.
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


# --- Save Model ---


def save_model(model, tokenizer, output_dir=OUTPUT_DIR):
    """Save model in JSON format after training."""

    os.makedirs(output_dir, exist_ok=True)

    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"models saved to: {output_dir}")
    except OSError:
        print("Output Directory Does Not Exist")
        raise
