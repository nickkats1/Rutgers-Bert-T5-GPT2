"""Tests for the T5 CustomDataset."""

import pytest
import pandas as pd
from tests.conftest import requires_network


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained("t5-small")
    except Exception:
        pytest.skip("t5-small not available (no network)")


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Description": ["Company reports record profits this quarter"],
            "Headlines": ["Record profits reported"],
        }
    )


@requires_network
def test_dataset_length(df, tokenizer):
    from src.t5.dataset.custom_dataset import CustomDataset
    ds = CustomDataset(
        df, tokenizer, source_len=64, target_len=32,
        source_col="Description", target_col="Headlines",
    )
    assert len(ds) == 1


@requires_network
def test_dataset_item_keys(df, tokenizer):
    from src.t5.dataset.custom_dataset import CustomDataset
    ds = CustomDataset(
        df, tokenizer, source_len=64, target_len=32,
        source_col="Description", target_col="Headlines",
    )
    item = ds[0]
    assert set(item.keys()) == {"source_ids", "source_mask", "target_ids"}


@requires_network
def test_source_ids_shape(df, tokenizer):
    from src.t5.dataset.custom_dataset import CustomDataset
    ds = CustomDataset(
        df, tokenizer, source_len=64, target_len=32,
        source_col="Description", target_col="Headlines",
    )
    item = ds[0]
    assert item["source_ids"].shape == (64,)
    assert item["source_mask"].shape == (64,)
    assert item["target_ids"].shape == (32,)


@requires_network
def test_multiple_rows(tokenizer):
    from src.t5.dataset.custom_dataset import CustomDataset
    df_multi = pd.DataFrame(
        {
            "Description": ["First description", "Second description", "Third description"],
            "Headlines": ["First headline", "Second headline", "Third headline"],
        }
    )
    ds = CustomDataset(
        df_multi, tokenizer, source_len=32, target_len=16,
        source_col="Description", target_col="Headlines",
    )
    assert len(ds) == 3
    for i in range(3):
        item = ds[i]
        assert item["source_ids"].shape == (32,)
        assert item["target_ids"].shape == (16,)
