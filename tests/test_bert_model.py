"""Tests for the BertClassifier model."""

import torch
import pytest
from tests.conftest import requires_network


@requires_network
def test_classifier_output_shape():
    """BERT classifier should output (batch_size, 3) logits."""
    from src.bert.model.bert import BertClassifier
    model = BertClassifier()
    model.eval()
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    assert logits.shape == (batch_size, 3)


@requires_network
def test_classifier_output_is_not_softmax():
    """Output should be raw logits, not probabilities (sum ≠ 1)."""
    from src.bert.model.bert import BertClassifier
    model = BertClassifier()
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 16))
    attention_mask = torch.ones(1, 16, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    assert not torch.allclose(logits.sum(), torch.tensor(1.0), atol=0.1)


@requires_network
def test_classifier_different_inputs_give_different_outputs():
    """Two different inputs should produce different logits."""
    from src.bert.model.bert import BertClassifier
    model = BertClassifier()
    model.eval()
    ids_a = torch.randint(0, 500, (1, 16))
    ids_b = torch.randint(500, 1000, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.long)
    with torch.no_grad():
        out_a = model(ids_a, mask)
        out_b = model(ids_b, mask)
    assert not torch.allclose(out_a, out_b)
