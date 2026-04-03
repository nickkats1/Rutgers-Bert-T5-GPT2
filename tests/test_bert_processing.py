"""Tests for BERT text-processing utilities."""

from src.bert.utils.processing import polarity, sentiment


def test_polarity_returns_float():
    result = polarity("The market surged today")
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


def test_sentiment_positive():
    assert sentiment(0.5) == "Positive"


def test_sentiment_negative():
    assert sentiment(-0.3) == "Negative"


def test_sentiment_neutral():
    assert sentiment(0.0) == "Neutral"


def test_polarity_negative_text():
    assert polarity("This is a terrible disaster") < 0


def test_polarity_positive_text():
    assert polarity("This is a wonderful, excellent day") > 0
