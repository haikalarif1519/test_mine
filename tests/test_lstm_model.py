"""Tests for lstm_model module (PyTorch)."""

import numpy as np
import pytest
import torch

from lstm_model import (
    SystemStateLSTM,
    build_model,
    predict_state,
    predict_state_probabilities,
)
import config


class TestBuildModel:
    def test_is_pytorch_module(self):
        model = build_model(num_features=10, num_classes=3)
        assert isinstance(model, torch.nn.Module)

    def test_forward_output_shape(self):
        model = build_model(num_features=10, num_classes=3)
        x = torch.randn(4, 5, 10)  # batch=4, seq=5, features=10
        out = model(x)
        assert out.shape == (4, 3)


class TestPredict:
    @pytest.fixture()
    def model(self):
        return build_model(num_features=10, num_classes=3)

    def test_probabilities_sum_to_one(self, model):
        seq = np.random.rand(5, 10).astype(np.float32)
        probs = predict_state_probabilities(model, seq)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_state_returns_valid(self, model):
        seq = np.random.rand(5, 10).astype(np.float32)
        state, probs = predict_state(model, seq)
        assert state in config.STATE_NAMES.values()
        assert probs.shape == (3,)
