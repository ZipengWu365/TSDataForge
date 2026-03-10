from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.rng import as_rng
from ..dynamics.causal import CausalTreatmentOutcome, CausalVARX
from ..dynamics.control import EventTriggeredController, SecondOrderTracking
from ..dynamics.policy import PolicyControlledStateSpace
from ..dynamics.regime import RegimeSwitch
from ..dynamics.state_space import JointServoMIMO, LinearStateSpace
from ..interventions import InterventionSpec
from ..policies import ConstantPolicy, LinearFeedbackPolicy, PiecewiseConstantPolicy
from ..operators import Add, Stack
from ..primitives.control import PiecewiseConstantInput, StepReference, WaypointTrajectory
from ..primitives.events import BurstyPulseTrain
from ..primitives.noise import AR1Noise, RandomWalkProcess, WhiteGaussianNoise
from ..primitives.seasonal import MultiSineSeasonality, QuasiPeriodicSeasonality, SineSeasonality
from ..primitives.trend import LinearTrend, PiecewiseLinearTrend
from ..specs import SeriesSpec

RecipeFn = Callable[[np.random.Generator], SeriesSpec]


def _white_noise(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=WhiteGaussianNoise(std=float(rng.uniform(0.15, 0.9))),
        structure_id="white_noise",
        tags=("noise", "white_noise"),
        name="white_noise",
    )


def _colored_noise(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=AR1Noise(phi=float(rng.uniform(0.45, 0.95)), sigma=float(rng.uniform(0.05, 0.35))),
        structure_id="colored_noise",
        tags=("noise", "colored_noise"),
        name="colored_noise",
    )


def _random_walk(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=RandomWalkProcess(sigma=float(rng.uniform(0.05, 0.25))),
        structure_id="random_walk",
        tags=("random_walk", "nonstationary"),
        name="random_walk",
    )


def _linear_trend(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=LinearTrend(slope=float(rng.uniform(-0.03, 0.03)), intercept=float(rng.uniform(-1.0, 1.0))),
        structure_id="linear_trend",
        tags=("trend", "linear_trend"),
        name="linear_trend",
    )


def _piecewise_trend(rng: np.random.Generator) -> SeriesSpec:
    k1 = float(rng.uniform(0.25, 0.45))
    k2 = float(rng.uniform(0.6, 0.85))
    slopes = (
        float(rng.uniform(-0.03, 0.03)),
        float(rng.uniform(-0.03, 0.03)),
        float(rng.uniform(-0.03, 0.03)),
    )
    return SeriesSpec(
        latent=PiecewiseLinearTrend(knots=(k1, k2), slopes=slopes, intercept=float(rng.uniform(-1.0, 1.0))),
        structure_id="piecewise_trend",
        tags=("trend", "piecewise"),
        name="piecewise_trend",
    )


def _sine_periodic(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=SineSeasonality(freq=float(rng.choice((12.0, 24.0, 48.0))), amp=float(rng.uniform(0.5, 1.5))),
        structure_id="sine_periodic",
        tags=("seasonal", "single_periodic"),
        name="sine_periodic",
    )


def _multi_periodic(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=MultiSineSeasonality(
            freqs=(24.0, 168.0),
            amps=(float(rng.uniform(0.5, 1.4)), float(rng.uniform(0.1, 0.6))),
        ),
        structure_id="multi_periodic",
        tags=("seasonal", "multi_periodic"),
        name="multi_periodic",
    )


def _quasi_periodic(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=QuasiPeriodicSeasonality(
            freq=float(rng.choice((12.0, 24.0, 48.0))),
            amp=float(rng.uniform(0.5, 1.5)),
            jitter=float(rng.uniform(0.02, 0.12)),
        ),
        structure_id="quasi_periodic",
        tags=("seasonal", "quasi_periodic"),
        name="quasi_periodic",
    )


def _trend_seasonal_noise(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=Add(
            (
                LinearTrend(slope=float(rng.uniform(-0.02, 0.02)), intercept=float(rng.uniform(-0.5, 0.5))),
                MultiSineSeasonality(
                    freqs=(24.0, 168.0),
                    amps=(float(rng.uniform(0.6, 1.6)), float(rng.uniform(0.1, 0.5))),
                ),
                AR1Noise(phi=float(rng.uniform(0.25, 0.85)), sigma=float(rng.uniform(0.03, 0.15))),
            )
        ),
        structure_id="trend_seasonal_noise",
        tags=("trend", "seasonal", "noise", "composite"),
        name="trend_seasonal_noise",
    )


def _regime_switch(rng: np.random.Generator) -> SeriesSpec:
    regime_a = Add(
        (
            LinearTrend(slope=float(rng.uniform(-0.01, 0.01)), intercept=float(rng.uniform(-0.5, 0.2))),
            SineSeasonality(freq=24.0, amp=float(rng.uniform(0.4, 1.2))),
            WhiteGaussianNoise(std=float(rng.uniform(0.02, 0.1))),
        )
    )
    regime_b = Add(
        (
            LinearTrend(slope=float(rng.uniform(0.0, 0.03)), intercept=float(rng.uniform(0.0, 1.0))),
            SineSeasonality(freq=12.0, amp=float(rng.uniform(0.1, 0.7))),
            AR1Noise(phi=float(rng.uniform(0.5, 0.95)), sigma=float(rng.uniform(0.05, 0.2))),
        )
    )
    transition = np.array([[0.97, 0.03], [0.05, 0.95]], dtype=float)
    return SeriesSpec(
        latent=RegimeSwitch(regimes=(regime_a, regime_b), transition_matrix=transition),
        structure_id="regime_switch",
        tags=("regime_switch", "change_point"),
        name="regime_switch",
    )


def _robot_step_response(rng: np.random.Generator) -> SeriesSpec:
    ref = StepReference(
        levels=(0.0, float(rng.uniform(0.8, 1.6)), float(rng.uniform(-0.4, 0.4))),
        switch_points=(float(rng.uniform(0.12, 0.25)), float(rng.uniform(0.55, 0.78))),
    )
    return SeriesSpec(
        latent=SecondOrderTracking(
            reference=ref,
            natural_freq=float(rng.uniform(0.10, 0.25)),
            damping=float(rng.uniform(0.45, 1.1)),
            process_noise_std=float(rng.uniform(0.0, 0.01)),
        ),
        structure_id="robot_step_response",
        tags=("robotics", "control", "step_response"),
        name="robot_step_response",
    )


def _robot_waypoint_tracking(rng: np.random.Generator) -> SeriesSpec:
    ref = WaypointTrajectory(
        waypoints=(
            0.0,
            float(rng.uniform(0.6, 1.4)),
            float(rng.uniform(-0.8, 0.2)),
            float(rng.uniform(0.2, 1.0)),
        ),
        anchor_points=(0.0, float(rng.uniform(0.2, 0.35)), float(rng.uniform(0.5, 0.7)), 1.0),
    )
    return SeriesSpec(
        latent=Add(
            (
                SecondOrderTracking(
                    reference=ref,
                    natural_freq=float(rng.uniform(0.08, 0.18)),
                    damping=float(rng.uniform(0.7, 1.2)),
                    process_noise_std=float(rng.uniform(0.0, 0.008)),
                ),
                AR1Noise(phi=float(rng.uniform(0.2, 0.6)), sigma=float(rng.uniform(0.01, 0.04))),
            )
        ),
        structure_id="robot_waypoint_tracking",
        tags=("robotics", "control", "trajectory_tracking"),
        name="robot_waypoint_tracking",
    )


def _event_triggered_control(rng: np.random.Generator) -> SeriesSpec:
    ref = StepReference(
        levels=(0.0, float(rng.uniform(0.6, 1.4)), float(rng.uniform(-0.5, 0.1)), float(rng.uniform(0.3, 1.0))),
        switch_points=(0.18, 0.46, 0.74),
    )
    return SeriesSpec(
        latent=EventTriggeredController(
            reference=ref,
            kp=float(rng.uniform(1.1, 2.0)),
            plant_time_constant=float(rng.uniform(5.0, 12.0)),
            event_threshold=float(rng.uniform(0.10, 0.22)),
            max_hold_steps=int(rng.integers(8, 20)),
            min_inter_event_steps=int(rng.integers(3, 8)),
            actuator_limit=float(rng.uniform(1.8, 3.2)),
            process_noise_std=float(rng.uniform(0.0, 0.01)),
        ),
        structure_id="event_triggered_control",
        tags=("robotics", "control", "event_driven", "hybrid_system"),
        name="event_triggered_control",
    )


def _bursty_event_response(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=BurstyPulseTrain(
            base_rate=float(rng.uniform(0.003, 0.02)),
            burst_probability=float(rng.uniform(0.01, 0.05)),
            burst_size=int(rng.integers(4, 12)),
            decay=float(rng.uniform(2.0, 6.0)),
            amplitude=float(rng.uniform(0.6, 1.5)),
        ),
        structure_id="bursty_event_response",
        tags=("event_based", "event_driven", "bursty"),
        name="bursty_event_response",
    )


def _robot_joint_servo_mimo(rng: np.random.Generator) -> SeriesSpec:
    # Keep output dimensionality stable across samples for dataset stacking.
    n_joints = 6
    refs = []
    for _ in range(n_joints):
        refs.append(
            StepReference(
                levels=(0.0, float(rng.uniform(0.6, 1.5)), float(rng.uniform(-0.6, 0.4))),
                switch_points=(float(rng.uniform(0.15, 0.3)), float(rng.uniform(0.55, 0.8))),
            )
        )
    ref = Stack(tuple(refs))
    return SeriesSpec(
        latent=JointServoMIMO(
            n_joints=n_joints,
            reference=ref,
            wn=float(rng.uniform(0.10, 0.25)),
            zeta=float(rng.uniform(0.6, 1.2)),
            kp=float(rng.uniform(1.2, 3.0)),
            kd=float(rng.uniform(0.2, 1.0)),
            plant_gain=float(rng.uniform(0.8, 1.4)),
            actuator_limit=float(rng.uniform(2.0, 4.0)),
            process_noise_std=float(rng.uniform(0.0, 0.02)),
            measurement_noise_std=float(rng.uniform(0.005, 0.05)),
            # Occasionally event-triggered
            event_threshold=float(rng.choice([0.0, float(rng.uniform(0.05, 0.18))])),
            max_hold_steps=int(rng.integers(6, 14)),
            min_inter_event_steps=int(rng.integers(2, 5)),
            output_channels=("q", "dq", "u", "acc"),
        ),
        structure_id="robot_joint_servo_mimo",
        tags=("robotics", "control", "mimo", "state_space", "robot_sensors"),
        name="robot_joint_servo_mimo",
    )


def _linear_state_space_io(rng: np.random.Generator) -> SeriesSpec:
    # Random but mildly stable-ish continuous-time system (small eigenvalues).
    # Keep dimensionality stable across samples for dataset stacking.
    n_state = 5
    n_in = 2
    n_out = 4
    A = rng.normal(scale=0.15, size=(n_state, n_state))
    # Stabilize by shifting diagonal negative
    A = A - 0.35 * np.eye(n_state)
    B = rng.normal(scale=0.3, size=(n_state, n_in))
    C = rng.normal(scale=0.6, size=(n_out, n_state))
    D = rng.normal(scale=0.05, size=(n_out, n_in))
    u = PiecewiseConstantInput(dim=n_in, hold_min=8, hold_max=30, scale=float(rng.uniform(0.4, 1.2)))
    return SeriesSpec(
        latent=LinearStateSpace(
            A=A,
            B=B,
            C=C,
            D=D,
            input=u,
            form="continuous",
            process_noise_std=float(rng.uniform(0.0, 0.03)),
            measurement_noise_std=float(rng.uniform(0.0, 0.02)),
        ),
        structure_id="linear_state_space_io",
        tags=("state_space", "control", "io_pair", "multivariate"),
        name="linear_state_space_io",
    )


def _causal_varx(rng: np.random.Generator) -> SeriesSpec:
    # Small lag-2 causal VARX with sparse directed lag graph + exogenous drivers.
    n_vars = 6
    n_exog = 2
    lags = 2

    # Sparse adjacency mask (include diagonal self-lags).
    adj = (rng.random((n_vars, n_vars)) < 0.25).astype(float)
    np.fill_diagonal(adj, 1.0)

    A = np.zeros((lags, n_vars, n_vars), dtype=float)
    # Lag-1: dominant self influence for stability.
    A[0] = 0.35 * np.eye(n_vars) + adj * rng.normal(scale=0.10, size=(n_vars, n_vars))
    # Lag-2: weaker influence.
    A[1] = adj * rng.normal(scale=0.05, size=(n_vars, n_vars))

    # Exogenous drivers (control inputs)
    B = rng.normal(scale=0.35, size=(n_vars, n_exog))
    # Encourage sparse input influence
    B *= (rng.random((n_vars, n_exog)) < 0.6)
    u = PiecewiseConstantInput(dim=n_exog, hold_min=6, hold_max=26, scale=float(rng.uniform(0.4, 1.2)))

    return SeriesSpec(
        latent=CausalVARX(
            A=A,
            B=B,
            bias=rng.normal(scale=0.1, size=(n_vars,)),
            exogenous=u,
            noise_std=float(rng.uniform(0.02, 0.08)),
            nonlinearity=str(rng.choice(["linear", "tanh"])),
        ),
        structure_id="causal_varx",
        tags=("causal", "varx", "control", "io_pair", "multivariate"),
        name="causal_varx",
    )


def _causal_treatment_outcome(rng: np.random.Generator) -> SeriesSpec:
    return SeriesSpec(
        latent=CausalTreatmentOutcome(
            confounder_ar=float(rng.uniform(0.65, 0.98)),
            outcome_ar=float(rng.uniform(0.55, 0.95)),
            treatment_confounding=float(rng.uniform(0.8, 2.2)),
            outcome_confounding=float(rng.uniform(0.4, 1.4)),
            treatment_effect=float(rng.uniform(0.6, 1.8)),
            treatment_lag=int(rng.integers(0, 3)),
            treatment_type="binary",
            noise_std=(float(rng.uniform(0.15, 0.45)), float(rng.uniform(0.15, 0.45)), float(rng.uniform(0.15, 0.55))),
            return_potential_outcomes=True,
        ),
        structure_id="causal_treatment_outcome",
        tags=("causal", "treatment_outcome", "confounding", "multivariate"),
        name="causal_treatment_outcome",
    )


def _policy_controlled_state_space(rng: np.random.Generator) -> SeriesSpec:
    n_state = 4
    n_input = 2
    n_output = 3
    A = 0.82 * np.eye(n_state) + rng.normal(scale=0.04, size=(n_state, n_state))
    B = rng.normal(scale=0.25, size=(n_state, n_input))
    C = rng.normal(scale=0.45, size=(n_output, n_state))
    D = rng.normal(scale=0.04, size=(n_output, n_input))
    K = rng.normal(scale=0.35, size=(n_input, n_state))
    base_policy = LinearFeedbackPolicy(K=K, bias=np.zeros(n_input), source="state", name="feedback")
    alt_policy = PiecewiseConstantPolicy(
        actions=((0.0, 0.0), (0.6, -0.4), (-0.3, 0.5)),
        switch_points=(0.25, 0.6),
        name="open_loop_shift",
    )
    interventions = (
        InterventionSpec(target="input", index=int(rng.integers(0, n_input)), start=0.55, end=0.8, value=float(rng.uniform(-0.8, 0.8)), mode="add", name="input_perturb"),
    )
    return SeriesSpec(
        latent=PolicyControlledStateSpace(
            A=A,
            B=B,
            C=C,
            D=D,
            policy=base_policy,
            counterfactual_policies=(alt_policy, ConstantPolicy(action_value=(0.0, 0.0), name="zero_policy")),
            form="discrete",
            process_noise_std=float(rng.uniform(0.0, 0.03)),
            measurement_noise_std=float(rng.uniform(0.0, 0.02)),
            action_limit=float(rng.uniform(1.0, 2.0)),
            interventions=interventions,
            reward_state_weight=float(rng.uniform(0.6, 1.5)),
            reward_action_weight=float(rng.uniform(0.02, 0.2)),
            output_names=("sensor0", "sensor1", "sensor2"),
        ),
        structure_id="policy_controlled_state_space",
        tags=("control", "causal", "policy_driven", "counterfactual", "state_space", "multivariate"),
        name="policy_controlled_state_space",
    )


AVAILABLE_RECIPES: dict[str, RecipeFn] = {
    "white_noise": _white_noise,
    "colored_noise": _colored_noise,
    "random_walk": _random_walk,
    "linear_trend": _linear_trend,
    "piecewise_trend": _piecewise_trend,
    "sine_periodic": _sine_periodic,
    "multi_periodic": _multi_periodic,
    "quasi_periodic": _quasi_periodic,
    "trend_seasonal_noise": _trend_seasonal_noise,
    "regime_switch": _regime_switch,
    "robot_step_response": _robot_step_response,
    "robot_waypoint_tracking": _robot_waypoint_tracking,
    "event_triggered_control": _event_triggered_control,
    "bursty_event_response": _bursty_event_response,
    "robot_joint_servo_mimo": _robot_joint_servo_mimo,
    "linear_state_space_io": _linear_state_space_io,
    "causal_varx": _causal_varx,
    "causal_treatment_outcome": _causal_treatment_outcome,
    "policy_controlled_state_space": _policy_controlled_state_space,
}


def recipe_names() -> tuple[str, ...]:
    return tuple(AVAILABLE_RECIPES)


def build_recipe(name: str, seed: int | np.random.Generator | None = None) -> SeriesSpec:
    rng = as_rng(seed)
    if name not in AVAILABLE_RECIPES:
        raise KeyError(f"Unknown structure recipe: {name!r}")
    return AVAILABLE_RECIPES[name](rng)
