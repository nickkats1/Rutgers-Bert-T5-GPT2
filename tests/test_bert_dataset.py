"""Tests for the BERT CustomDataset and DataLoader factory."""

import numpy as np
import pytest
from tests.conftest import requires_network

import pandas as pd


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased not available (no network)")


@requires_network
def test_dataset_length(tokenizer):
    from src.bert.dataset.custom_dataset import CustomDataset
    headlines = np.array(["headline one", "headline two", "headline three"])
    targets = np.array([0, 1, 2])
    ds = CustomDataset(headlines, targets, max_len=32, tokenizer=tokenizer)
    assert len(ds) == 3


@requires_network
def test_dataset_item_keys(tokenizer):
    from src.bert.dataset.custom_dataset import CustomDataset
    headlines = np.array(["breaking news today"])
    targets = np.array([1])
    ds = CustomDataset(headlines, targets, max_len=32, tokenizer=tokenizer)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "targets"}


@requires_network
def test_dataset_input_ids_shape(tokenizer):
    from src.bert.dataset.custom_dataset import CustomDataset
    headlines = np.array(["short headline"])
    targets = np.array([0])
    ds = CustomDataset(headlines, targets, max_len=32, tokenizer=tokenizer)
    item = ds[0]
    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)


@requires_network
def test_dataset_target_value(tokenizer):
    from src.bert.dataset.custom_dataset import CustomDataset
    headlines = np.array(["positive news"])
    targets = np.array([2])
    ds = CustomDataset(headlines, targets, max_len=16, tokenizer=tokenizer)
    item = ds[0]
    assert item["targets"].item() == 2


@requires_network
def test_get_dataloader_returns_batches(tokenizer):
    from src.bert.dataset.custom_dataset import get_dataloader
    df = pd.DataFrame(
        {
            "Headlines": ["headline one", "headline two", "headline three", "headline four"],
            "sentiment": [0, 1, 2, 1],
        }
    )
    loader = get_dataloader(df, tokenizer, max_len=32, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
