"""Tests for fuzzy_logic module."""

import pytest

from fuzzy_logic import build_fuzzy_system, evaluate_sleep_decision


class TestFuzzyLogic:
    @pytest.fixture()
    def fuzzy_sim(self):
        return build_fuzzy_system()

    def test_high_idle_many_consecutive_triggers_sleep(self, fuzzy_sim):
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 0.95, 10)
        assert should_sleep is True
        assert score > 0.5

    def test_low_idle_no_sleep(self, fuzzy_sim):
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 0.1, 0)
        assert should_sleep is False

    def test_medium_idle_few_consecutive_no_sleep(self, fuzzy_sim):
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 0.5, 1)
        assert should_sleep is False

    def test_clamping_out_of_range(self, fuzzy_sim):
        """Values outside the defined range should be clamped, not crash."""
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 1.5, 20)
        assert isinstance(should_sleep, bool)
        assert 0.0 <= score <= 1.0

    def test_boundary_idle_high_consecutive(self, fuzzy_sim):
        """High idle probability with intermediate consecutive count."""
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 0.9, 5)
        # Should produce a score but not necessarily trigger sleep
        assert isinstance(should_sleep, bool)
        assert 0.0 <= score <= 1.0
