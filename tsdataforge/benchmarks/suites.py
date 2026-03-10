from __future__ import annotations

import numpy as np

from ..datasets.builder import generate_dataset
from ..observation import BlockMissing, IrregularSampling, MeasurementNoise
from ..specs import ObservationSpec

TSDF10_STRUCTURES = (
    "white_noise",
    "colored_noise",
    "random_walk",
    "linear_trend",
    "piecewise_trend",
    "sine_periodic",
    "multi_periodic",
    "quasi_periodic",
    "trend_seasonal_noise",
    "regime_switch",
)

TSDF_CONTROL_STRUCTURES = (
    "robot_step_response",
    "robot_waypoint_tracking",
    "event_triggered_control",
    "bursty_event_response",
)

TSDF_CONTROL_MIMO_STRUCTURES = (
    "robot_joint_servo_mimo",
    "linear_state_space_io",
)

TSDF_CAUSAL_STRUCTURES = (
    "causal_varx",
    "causal_treatment_outcome",
    "policy_controlled_state_space",
)

TSDF_POLICY_STRUCTURES = (
    "policy_controlled_state_space",
    "causal_varx",
)


def tsdf10(
    *,
    task: str = "classification",
    n_series: int = 1000,
    length: int = 256,
    seed: int | None = None,
):
    return generate_dataset(task=task, structures=TSDF10_STRUCTURES, n_series=n_series, length=length, seed=seed)



def _robust_observation(rng: np.random.Generator) -> ObservationSpec:
    return ObservationSpec(
        sampling=IrregularSampling(dt=1.0, jitter=float(rng.uniform(0.05, 0.25))),
        missing=BlockMissing(rate=float(rng.uniform(0.05, 0.15)), block_min=4, block_max=18),
        measurement_noise=MeasurementNoise(std=float(rng.uniform(0.01, 0.08))),
    )



def tsdf_robust(
    *,
    task: str = "classification",
    n_series: int = 1000,
    length: int = 256,
    seed: int | None = None,
):
    return generate_dataset(
        task=task,
        structures=TSDF10_STRUCTURES,
        n_series=n_series,
        length=length,
        seed=seed,
        observation_factory=_robust_observation,
    )



def tsdf_control(
    *,
    task: str = "classification",
    n_series: int = 600,
    length: int = 256,
    seed: int | None = None,
):
    return generate_dataset(
        task=task,
        structures=TSDF_CONTROL_STRUCTURES,
        n_series=n_series,
        length=length,
        seed=seed,
    )



def tsdf_control_mimo(
    *,
    task: str = "system_identification",
    n_series: int = 400,
    length: int = 256,
    horizon: int = 32,
    seed: int | None = None,
):
    return generate_dataset(
        task=task,
        structures=TSDF_CONTROL_MIMO_STRUCTURES,
        n_series=n_series,
        length=length,
        horizon=horizon,
        seed=seed,
    )



def tsdf_causal(
    *,
    task: str = "causal_response",
    n_series: int = 600,
    length: int = 256,
    horizon: int = 32,
    seed: int | None = None,
):
    return generate_dataset(
        task=task,
        structures=TSDF_CAUSAL_STRUCTURES,
        n_series=n_series,
        length=length,
        horizon=horizon,
        seed=seed,
    )



def tsdf_policy(
    *,
    task: str = "counterfactual_response",
    n_series: int = 400,
    length: int = 256,
    horizon: int = 32,
    seed: int | None = None,
):
    """Policy / intervention-oriented suite.

    Task-to-structure routing keeps the suite shape-consistent:
    - counterfactual_response / policy_value_estimation / intervention_detection -> policy-controlled state-space
    - causal_discovery -> causal VARX
    - causal_ite -> treatment/outcome SCM
    """

    task_key = str(task)
    if task_key == "causal_discovery":
        structures = ("causal_varx",)
    elif task_key == "causal_ite":
        structures = ("causal_treatment_outcome",)
    else:
        structures = ("policy_controlled_state_space",)

    return generate_dataset(
        task=task,
        structures=structures,
        n_series=n_series,
        length=length,
        horizon=horizon,
        seed=seed,
    )
