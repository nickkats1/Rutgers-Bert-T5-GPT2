"""Tests for GPT-2 data loading and splitting utilities."""

import pytest
import pandas as pd
from src.gpt2.data import load_data, split_data
from src.gpt2.utils import set_seed


def test_load_data_returns_dataframe(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text(
        "Time,Headlines\n"
        "2024-01-01,Test headline one\n"
        "2024-01-02,Another headline\n"
        "2024-01-03,Third headline here\n"
    )
    df = load_data(path=str(csv))
    assert isinstance(df, pd.DataFrame)
    assert "Headlines" in df.columns
    assert "Time" not in df.columns


def test_load_data_drops_duplicates(tmp_path):
    csv = tmp_path / "dupes.csv"
    csv.write_text(
        "Time,Headlines\n"
        "2024-01-01,Duplicate headline\n"
        "2024-01-01,Duplicate headline\n"
        "2024-01-02,Unique headline\n"
    )
    df = load_data(path=str(csv))
    assert len(df) == 2


def test_split_data_sizes():
    df = pd.DataFrame({"Headlines": [f"Headline {i}" for i in range(100)]})
    train, val = split_data(df)
    assert len(train) + len(val) == 100
    assert len(val) == pytest.approx(20, abs=2)


def test_split_data_returns_lists():
    df = pd.DataFrame({"Headlines": ["headline one", "headline two", "headline three"]})
    train, val = split_data(df)
    assert isinstance(train, list)
    assert isinstance(val, list)


def test_split_data_no_overlap():
    df = pd.DataFrame({"Headlines": [f"Headline {i}" for i in range(50)]})
    train, val = split_data(df)
    assert len(set(train) & set(val)) == 0


def test_set_seed_runs_without_error():
    set_seed()
