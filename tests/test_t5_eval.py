"""Tests for the T5 ROUGE evaluation utilities."""

import pytest
from src.t5.evaluate.eval import compute_rouge


def test_compute_rouge_returns_all_keys(sample_predictions, sample_actuals):
    results = compute_rouge(sample_predictions, sample_actuals)
    assert "rouge1" in results
    assert "rouge2" in results
    assert "rougeL" in results


def test_compute_rouge_perfect_match():
    texts = ["the cat sat on the mat"]
    results = compute_rouge(texts, texts)
    assert results["rouge1"] == pytest.approx(1.0, abs=0.01)
    assert results["rougeL"] == pytest.approx(1.0, abs=0.01)


def test_compute_rouge_scores_are_between_0_and_1(sample_predictions, sample_actuals):
    results = compute_rouge(sample_predictions, sample_actuals)
    for key, val in results.items():
        assert 0.0 <= val <= 1.0, f"{key} score {val} out of [0, 1]"


def test_compute_rouge_zero_overlap():
    preds = ["aaa bbb ccc"]
    refs = ["xxx yyy zzz"]
    results = compute_rouge(preds, refs)
    assert results["rouge1"] == pytest.approx(0.0, abs=0.01)


def test_compute_rouge_returns_floats(sample_predictions, sample_actuals):
    results = compute_rouge(sample_predictions, sample_actuals)
    for val in results.values():
        assert isinstance(val, float)
