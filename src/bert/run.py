"""Main script to fine-tune BERT for sentiment classification."""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from src.bert.config import bert_config
from src.bert.dataset.custom_dataset import get_dataloader
from src.bert.model.bert import BertClassifier
from src.bert.utils.processing import polarity, sentiment
from src.bert.eval.train_eval_epoch import training_epoch, eval_model


def main():
    """Train and evaluate the BERT sentiment classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv(bert_config.get("data_path"), delimiter=",")
    df["polarity"] = df["Headlines"].apply(polarity)
    df["sentiment"] = df["polarity"].apply(sentiment)
    df['sentiment'] = df['sentiment'].map({"Negative": 0, "Neutral": 1, "Positive": 2})
    
    # drop polarity and drop duplicates
    
    df.drop(["Time", "polarity"], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Train/val/test split
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=50, random_state=42)

    # Tokenizer and dataloaders
    tokenizer = BertTokenizer.from_pretrained(bert_config.get("model_name"))
    max_len = bert_config.get("max_len", 80)
    batch_size = bert_config.get("batch_size", 12)

    train_dataloader = get_dataloader(df_train, tokenizer, max_len, batch_size, shuffle=True)
    val_dataloader = get_dataloader(df_val, tokenizer, max_len, batch_size)
    test_dataloader = get_dataloader(df_test, tokenizer, max_len, batch_size)

    # Model, loss, optimizer
    model = BertClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    epochs = bert_config.get("epochs", 3)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_acc, train_loss = training_epoch(
            model, train_dataloader, loss_fn, optimizer, device, len(df_train)
        )

        val_acc, val_loss = eval_model(
            model, val_dataloader, loss_fn, device, len(df_val)
        )

        print(f"  Train Accuracy: {train_acc * 100:.2f}%  |  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy:   {val_acc * 100:.2f}%  |  Val Loss:   {val_loss:.4f}")

    # Final test evaluation
    test_acc, test_loss = eval_model(
        model, test_dataloader, loss_fn, device, len(df_test)
    )
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%  |  Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()