"""Tests for lstm_model module."""

import numpy as np
import pytest

from lstm_model import build_model, predict_state, predict_state_probabilities
import config


class TestBuildModel:
    def test_output_shape(self):
        model = build_model(seq_length=5, num_features=7, num_classes=3)
        assert model.output_shape == (None, 3)

    def test_input_shape(self):
        model = build_model(seq_length=5, num_features=7, num_classes=3)
        assert model.input_shape == (None, 5, 7)


class TestPredict:
    @pytest.fixture()
    def model(self):
        return build_model(seq_length=5, num_features=7, num_classes=3)

    def test_probabilities_sum_to_one(self, model):
        seq = np.random.rand(5, 7).astype(np.float32)
        probs = predict_state_probabilities(model, seq)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_state_returns_valid(self, model):
        seq = np.random.rand(5, 7).astype(np.float32)
        state, probs = predict_state(model, seq)
        assert state in config.STATE_NAMES.values()
        assert probs.shape == (3,)
