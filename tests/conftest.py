"""Shared pytest fixtures for the Rutgers-NLP test suite."""

import pytest
import pandas as pd


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_network: mark test as requiring network access to download model weights",
    )


def _is_network_available() -> bool:
    """Return True if HuggingFace Hub is reachable."""
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        return True
    except Exception:
        return False


requires_network = pytest.mark.skipif(
    not _is_network_available(),
    reason="Skipped: no network access (HuggingFace model download required)",
)


@pytest.fixture
def sample_headlines_df():
    """Small DataFrame of synthetic news headlines."""
    return pd.DataFrame(
        {
            "Headlines": [
                "Stock market surges to record high",
                "Tech company announces layoffs",
                "Scientists discover new species",
                "Government passes new climate bill",
            ]
        }
    )


@pytest.fixture
def sample_predictions():
    """Short list of synthetic generated headlines."""
    return ["The market is up today", "Tech firm cuts jobs"]


@pytest.fixture
def sample_actuals():
    """Matching list of reference headlines."""
    return [
        "Stock market surges to record high",
        "Tech company announces layoffs",
    ]
