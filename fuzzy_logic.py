"""
Fuzzy Logic Controller for Sleep Decision.

Uses the LSTM's idle-state probability and a count of consecutive idle
predictions to decide whether it is safe to put the system to sleep.
This prevents *catastrophic sleep* — putting the system to sleep while a
user or process is still active.
"""

import logging

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import config

logger = logging.getLogger(__name__)


def build_fuzzy_system():
    """
    Build and return a fuzzy-logic control system for the sleep decision.

    Inputs
    ------
    idle_probability : float in [0, 1]
        Probability of the system being in the *idle* state (from LSTM).
    consecutive_idle : int in [0, ``FUZZY_CONSEC_HIGH``]
        Number of consecutive time-steps the system was predicted idle.

    Output
    ------
    sleep_decision : float in [0, 1]
        Value > ``FUZZY_SLEEP_THRESHOLD`` → put the system to sleep.
    """
    # ── Antecedent (input) variables ────────────────────────────────────
    idle_prob = ctrl.Antecedent(
        np.linspace(config.FUZZY_IDLE_PROB_LOW, config.FUZZY_IDLE_PROB_HIGH, 100),
        "idle_probability",
    )
    consec = ctrl.Antecedent(
        np.arange(config.FUZZY_CONSEC_LOW, config.FUZZY_CONSEC_HIGH + 1, 1),
        "consecutive_idle",
    )

    # ── Consequent (output) variable ────────────────────────────────────
    sleep = ctrl.Consequent(
        np.linspace(config.FUZZY_SLEEP_LOW, config.FUZZY_SLEEP_HIGH, 100),
        "sleep_decision",
    )

    # ── Membership functions: idle_probability ──────────────────────────
    idle_prob["low"] = fuzz.trimf(idle_prob.universe, [0.0, 0.0, 0.4])
    idle_prob["medium"] = fuzz.trimf(idle_prob.universe, [0.3, 0.5, 0.7])
    idle_prob["high"] = fuzz.trimf(idle_prob.universe, [0.6, 1.0, 1.0])

    # ── Membership functions: consecutive_idle ──────────────────────────
    consec["few"] = fuzz.trimf(consec.universe, [0, 0, 3])
    consec["some"] = fuzz.trimf(consec.universe, [2, 5, 7])
    consec["many"] = fuzz.trimf(consec.universe, [6, 10, 10])

    # ── Membership functions: sleep_decision ────────────────────────────
    sleep["no_sleep"] = fuzz.trimf(sleep.universe, [0.0, 0.0, 0.3])
    sleep["maybe"] = fuzz.trimf(sleep.universe, [0.2, 0.5, 0.8])
    sleep["sleep"] = fuzz.trimf(sleep.universe, [0.7, 1.0, 1.0])

    # ── Rules ───────────────────────────────────────────────────────────
    # High idle probability AND many consecutive idle → sleep
    rule1 = ctrl.Rule(
        idle_prob["high"] & consec["many"], sleep["sleep"]
    )
    # High idle probability AND some consecutive idle → maybe
    rule2 = ctrl.Rule(
        idle_prob["high"] & consec["some"], sleep["maybe"]
    )
    # High idle probability AND few consecutive idle → no sleep
    rule3 = ctrl.Rule(
        idle_prob["high"] & consec["few"], sleep["no_sleep"]
    )
    # Medium idle probability AND many consecutive idle → maybe
    rule4 = ctrl.Rule(
        idle_prob["medium"] & consec["many"], sleep["maybe"]
    )
    # Medium idle probability AND some/few → no sleep
    rule5 = ctrl.Rule(
        idle_prob["medium"] & consec["some"], sleep["no_sleep"]
    )
    rule6 = ctrl.Rule(
        idle_prob["medium"] & consec["few"], sleep["no_sleep"]
    )
    # Low idle probability → always no sleep
    rule7 = ctrl.Rule(
        idle_prob["low"] & consec["many"], sleep["no_sleep"]
    )
    rule8 = ctrl.Rule(
        idle_prob["low"] & consec["some"], sleep["no_sleep"]
    )
    rule9 = ctrl.Rule(
        idle_prob["low"] & consec["few"], sleep["no_sleep"]
    )

    system = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
    ])
    return ctrl.ControlSystemSimulation(system)


def evaluate_sleep_decision(fuzzy_sim, idle_probability, consecutive_idle):
    """
    Run the fuzzy controller and return a sleep decision.

    Parameters
    ----------
    fuzzy_sim : ctrl.ControlSystemSimulation
        The pre-built fuzzy simulation object.
    idle_probability : float
        LSTM-predicted probability that the system is idle (0-1).
    consecutive_idle : int
        Number of consecutive idle predictions so far.

    Returns
    -------
    should_sleep : bool
        ``True`` if the system should be put to sleep.
    sleep_score : float
        The raw defuzzified output (0-1).
    """
    # Clamp inputs to valid ranges
    idle_probability = float(np.clip(idle_probability, 0.0, 1.0))
    consecutive_idle = int(np.clip(
        consecutive_idle, config.FUZZY_CONSEC_LOW, config.FUZZY_CONSEC_HIGH
    ))

    fuzzy_sim.input["idle_probability"] = idle_probability
    fuzzy_sim.input["consecutive_idle"] = consecutive_idle

    try:
        fuzzy_sim.compute()
        score = fuzzy_sim.output["sleep_decision"]
    except Exception:
        logger.warning("Fuzzy computation failed – defaulting to no-sleep.")
        return False, 0.0

    should_sleep = bool(score >= config.FUZZY_SLEEP_THRESHOLD)
    logger.debug(
        "Fuzzy: idle_prob=%.3f, consec=%d → score=%.3f, sleep=%s",
        idle_probability, consecutive_idle, score, should_sleep,
    )
    return should_sleep, float(score)
