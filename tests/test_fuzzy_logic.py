"""Tests for fuzzy_logic module."""

import pytest

from fuzzy_logic import build_fuzzy_system, evaluate_sleep_decision
import config


class TestFuzzyLogic:
    @pytest.fixture()
    def fuzzy_sim(self):
        return build_fuzzy_system()

    def test_high_idle_many_consecutive_triggers_sleep(self, fuzzy_sim):
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 0.95, 15)
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
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 1.5, 25)
        assert isinstance(should_sleep, bool)
        assert 0.0 <= score <= 1.0

    def test_boundary_idle_high_consecutive(self, fuzzy_sim):
        """High idle probability with intermediate consecutive count."""
        should_sleep, score = evaluate_sleep_decision(fuzzy_sim, 0.9, 10)
        # Should produce a score but not necessarily trigger sleep
        assert isinstance(should_sleep, bool)
        assert 0.0 <= score <= 1.0

    # ── 10-minute minimum guard ─────────────────────────────────────────

    def test_below_min_consec_never_sleeps(self, fuzzy_sim):
        """Even with very high idle probability, the system must NOT sleep
        if consecutive_idle < MIN_CONSEC_IDLE_FOR_SLEEP (default 10)."""
        for consec in range(config.MIN_CONSEC_IDLE_FOR_SLEEP):
            should_sleep, score = evaluate_sleep_decision(
                fuzzy_sim, 0.99, consec,
            )
            assert should_sleep is False, (
                f"Sleep triggered at consecutive_idle={consec}, "
                f"which is below minimum {config.MIN_CONSEC_IDLE_FOR_SLEEP}"
            )

    def test_at_min_consec_with_high_idle_can_sleep(self, fuzzy_sim):
        """At exactly MIN_CONSEC_IDLE_FOR_SLEEP with high idle prob, the
        hard guard no longer blocks — the fuzzy system decides."""
        should_sleep, score = evaluate_sleep_decision(
            fuzzy_sim, 0.99, config.MIN_CONSEC_IDLE_FOR_SLEEP,
        )
        # We don't assert should_sleep is True because the fuzzy system
        # might still not trigger sleep; we just verify the guard didn't
        # block (score > 0 means fuzzy was evaluated).
        assert score > 0.0
