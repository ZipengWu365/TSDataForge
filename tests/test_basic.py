from dataclasses import replace

import numpy as np
import pytest

from tsdataforge import Compiler, ObservationSpec, SeriesSpec, describe_series, generate_dataset, generate_series
from tsdataforge import generate_eda_report, generate_dataset_eda_report
from tsdataforge import generate_counterfactual_pair
from tsdataforge.counterfactual import with_policy
from tsdataforge.dynamics import PolicyControlledStateSpace
from tsdataforge.interventions import InterventionSpec
from tsdataforge.policies import ConstantPolicy, LinearFeedbackPolicy
from tsdataforge import wrap_external_series
from tsdataforge.core.registry import object_from_dict
from tsdataforge.dynamics import (
    CausalTreatmentOutcome,
    CausalVARX,
    EventTriggeredController,
    JointServoMIMO,
    RegimeSwitch,
    SecondOrderTracking,
)
from tsdataforge.observation import BlockMissing, IrregularSampling, MeasurementNoise
from tsdataforge.operators import Add, Stack
from tsdataforge.primitives import (
    LinearTrend,
    MultiSineSeasonality,
    StepReference,
    WaypointTrajectory,
    WhiteGaussianNoise,
)



def test_spec_roundtrip_and_compile():
    spec = SeriesSpec(
        latent=Add((LinearTrend(slope=0.01), MultiSineSeasonality(freqs=(24.0, 96.0), amps=(1.0, 0.3)))),
        observation=ObservationSpec(
            sampling=IrregularSampling(dt=1.0, jitter=0.2),
            missing=BlockMissing(rate=0.1, block_min=3, block_max=8),
            measurement_noise=MeasurementNoise(std=0.05),
        ),
        structure_id="demo",
        tags=("trend", "seasonal", "irregular"),
    )
    restored = object_from_dict(spec.to_dict())
    assert isinstance(restored, SeriesSpec)
    assert restored.structure_id == "demo"

    series = Compiler(seed=123).compile(restored, length=128)
    assert series.values.shape == (128,)
    assert series.trace is not None
    assert series.trace.structure_id == "demo"
    assert "observed_mask" in series.trace.masks



def test_dataset_builder_shapes():
    dataset = generate_dataset(
        task="classification",
        structures=["white_noise", "trend_seasonal_noise"],
        n_series=12,
        length=64,
        seed=123,
    )
    assert dataset.X.shape == (12, 64)
    assert dataset.y.shape == (12,)
    assert dataset.label_names == ["white_noise", "trend_seasonal_noise"]



def test_change_point_detection_targets():
    dataset = generate_dataset(
        task="change_point_detection",
        structures=["regime_switch"],
        n_series=4,
        length=128,
        seed=7,
    )
    assert dataset.y is not None
    assert dataset.y.shape == (4, 128)
    assert np.any(dataset.y.sum(axis=1) > 0)



def test_control_components_and_event_detection():
    spec = SeriesSpec(
        latent=EventTriggeredController(
            reference=StepReference(levels=(0.0, 1.0, -0.3, 0.7), switch_points=(0.2, 0.5, 0.8)),
            kp=1.5,
            plant_time_constant=8.0,
            event_threshold=0.1,
            max_hold_steps=10,
        ),
        structure_id="event_triggered_control",
        tags=("robotics", "control", "event_driven"),
    )
    restored = object_from_dict(spec.to_dict())
    assert isinstance(restored.latent, EventTriggeredController)

    series = Compiler(seed=5).compile(restored, length=160)
    assert "latent/trigger_mask" in series.trace.states
    trigger_keys = [k for k in series.trace.states if k.endswith('/trigger_mask')]
    assert trigger_keys
    trigger_mask = series.trace.states[trigger_keys[0]]
    assert int(np.sum(trigger_mask)) > 0

    dataset = generate_dataset(
        task="event_detection",
        structures=["event_triggered_control", "bursty_event_response"],
        n_series=8,
        length=160,
        seed=9,
    )
    assert dataset.y is not None
    assert dataset.y.shape == (8, 160)
    assert np.any(dataset.y.sum(axis=1) > 0)



def test_second_order_tracking_reference_trace():
    sample = generate_series(
        length=128,
        components=[
            SecondOrderTracking(
                reference=WaypointTrajectory(waypoints=(0.0, 1.0, -0.2, 0.6), anchor_points=(0.0, 0.3, 0.7, 1.0)),
                natural_freq=0.14,
                damping=0.85,
            )
        ],
        seed=12,
    )
    assert sample.trace is not None
    keys = sample.trace.contributions.keys()
    assert any('reference:waypointtrajectory' in key for key in keys)
    assert 'latent' in keys


def test_mimo_joint_servo_trace_shapes():
    ref = Stack(
        (
            StepReference(levels=(0.0, 1.0, 0.2), switch_points=(0.2, 0.7)),
            StepReference(levels=(0.0, -0.8, 0.6), switch_points=(0.25, 0.75)),
            StepReference(levels=(0.0, 0.5, -0.4), switch_points=(0.15, 0.65)),
        )
    )
    sample = generate_series(
        length=200,
        components=[JointServoMIMO(n_joints=3, reference=ref, output_channels=("q", "dq", "u", "acc"), event_threshold=0.1)],
        seed=123,
    )
    assert sample.values.shape == (200, 12)
    assert sample.trace is not None
    state_key = next(k for k in sample.trace.states if k.endswith("/state"))
    input_key = next(k for k in sample.trace.states if k.endswith("/input"))
    X = np.asarray(sample.trace.states[state_key])
    U = np.asarray(sample.trace.states[input_key])
    assert X.shape == (200, 6)
    assert U.shape == (200, 3)


def test_system_identification_dataset_shapes():
    dataset = generate_dataset(
        task="system_identification",
        structures=["linear_state_space_io"],
        n_series=6,
        length=120,
        horizon=16,
        seed=11,
    )
    assert dataset.X.shape == (6, 104, 6)
    assert dataset.y is not None
    assert dataset.y.shape == (6, 16, 4)
    assert dataset.aux is not None
    assert dataset.aux["u"].shape == (6, 120, 2)
    assert dataset.aux["x"].shape == (6, 120, 5)


def test_causal_generators_and_tasks():
    # Causal VARX should expose adjacency for causal discovery.
    ds = generate_dataset(
        task="causal_discovery",
        structures=["causal_varx"],
        n_series=3,
        length=80,
        seed=0,
    )
    assert ds.y is not None
    assert ds.y.shape[0] == 3
    assert ds.y.ndim == 3  # (N, n_vars, n_vars)

    # Treatment/outcome generator should expose ITE.
    ds2 = generate_dataset(
        task="causal_ite",
        structures=["causal_treatment_outcome"],
        n_series=4,
        length=120,
        horizon=16,
        seed=1,
    )
    assert ds2.y is not None
    assert ds2.y.shape == (4, 16)

    # Direct component compile sanity
    spec = SeriesSpec(latent=CausalTreatmentOutcome(return_potential_outcomes=True), structure_id="cto")
    series = Compiler(seed=2).compile(spec, length=100)
    assert series.values.shape == (100, 3)
    assert series.trace is not None
    assert any(k.endswith("/ite") for k in series.trace.states)


def test_describe_series_infers_seasonality():
    t = np.arange(512, dtype=float)
    y = np.sin(2 * np.pi * t / 24.0) + 0.2 * np.random.default_rng(0).normal(size=len(t))
    desc = describe_series(y, t)
    assert "seasonal" in desc.inferred_tags


def test_eda_report_generation(tmp_path):
    pytest.importorskip("matplotlib")

    t = np.arange(256, dtype=float)
    y = np.sin(2 * np.pi * t / 24.0) + 0.1 * np.random.default_rng(0).normal(size=len(t))
    out = tmp_path / "report.html"
    rep = generate_eda_report(y, t, output_path=out)
    assert out.exists()
    assert "TSDataForge" in rep.html
    assert "seasonal" in rep.html


def test_dataset_eda_report_generation(tmp_path):
    pytest.importorskip("matplotlib")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 64))
    out = tmp_path / "ds_report.html"
    rep = generate_dataset_eda_report(X, output_path=out)
    assert out.exists()
    assert "Dataset" in rep.html or "dataset" in rep.html.lower()


def test_wrap_external_series_creates_trace():
    t = np.linspace(0, 1, 50)
    y = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)], axis=1)
    gs = wrap_external_series(y, t, channel_names=["sin", "cos"], tags=("robotics", "external"))
    assert gs.values.shape == (50, 2)
    assert gs.trace is not None
    assert "observed_mask" in gs.trace.masks


def test_policy_controlled_state_space_and_counterfactual_tasks():
    ds = generate_dataset(
        task="counterfactual_response",
        structures=["policy_controlled_state_space"],
        n_series=4,
        length=128,
        horizon=16,
        seed=21,
    )
    assert ds.y is not None
    assert ds.y.shape == (4, 16)

    ds2 = generate_dataset(
        task="policy_value_estimation",
        structures=["policy_controlled_state_space"],
        n_series=5,
        length=96,
        seed=22,
    )
    assert ds2.y is not None
    assert ds2.y.shape == (5,)

    ds3 = generate_dataset(
        task="intervention_detection",
        structures=["policy_controlled_state_space"],
        n_series=3,
        length=100,
        seed=23,
    )
    assert ds3.y is not None
    assert ds3.y.shape == (3, 100)
    assert np.any(ds3.y.sum(axis=1) > 0)


def test_generate_counterfactual_pair_helper():
    spec = SeriesSpec(latent=CausalTreatmentOutcome(return_potential_outcomes=False), structure_id="cto")
    pair = generate_counterfactual_pair(
        spec=spec,
        length=80,
        seed=5,
        intervention=InterventionSpec(target="treatment", index=1, start=0.4, end=0.8, value=1.0, mode="override"),
    )
    assert pair.factual.values.shape == (80, 3)
    assert pair.counterfactual.values.shape == (80, 3)
    assert not np.allclose(pair.factual.values[:, 2], pair.counterfactual.values[:, 2])


def test_trace_aware_eda_report_generation(tmp_path):
    pytest.importorskip("matplotlib")

    A = 0.9 * np.eye(2)
    B = np.ones((2, 1)) * 0.25
    C = np.eye(2)
    policy = LinearFeedbackPolicy(K=np.array([[0.6, -0.2]]), source="state", name="feedback")
    alt = ConstantPolicy(action_value=(0.0,), name="zero")
    sample = generate_series(
        length=80,
        components=[PolicyControlledStateSpace(A=A, B=B, C=C, policy=policy, counterfactual_policies=(alt,), interventions=(InterventionSpec(target="input", index=0, start=0.5, value=0.5, mode="add"),))],
        seed=4,
    )
    out = tmp_path / "trace_report.html"
    rep = generate_eda_report(sample, output_path=out)
    assert out.exists()
    assert "counterfactual" in rep.html.lower()
    assert "Control / causal trace summary" in rep.html
