"""Custom PyTorch Dataset for BERT sentiment classification."""

import torch


class CustomDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset that tokenizes headlines for BERT sentiment classification.

    Attributes:
        headlines: Numpy array of headline strings.
        targets: Numpy array of sentiment labels.
        max_len: Maximum token length for sequences.
        tokenizer: HuggingFace tokenizer (e.g., BertTokenizer).
    """

    def __init__(self, headlines, targets, max_len, tokenizer):
        """Initialize the dataset.

        Args:
            headlines: Numpy array of headline strings.
            targets: Numpy array of sentiment labels.
            max_len: Maximum token length for sequences.
            tokenizer: HuggingFace tokenizer.
        """
        self.headlines = headlines
        self.targets = targets
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.headlines)

    def __getitem__(self, idx):
        """Tokenize and return a single headline-target pair.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask for the input.
                - targets: Sentiment label as a tensor.
        """
        headline = str(self.headlines[idx])
        target = self.targets[idx]

        encoding = self.tokenizer(
            headline,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


def get_dataloader(df, tokenizer, max_len, batch_size, shuffle=False):
    """Create a DataLoader from a DataFrame.

    Args:
        df: Pandas DataFrame with 'Headlines' and 'sentiment' columns.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum token length for sequences.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.

    Returns:
        A PyTorch DataLoader.
    """
    dataset = CustomDataset(
        headlines=df["Headlines"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
