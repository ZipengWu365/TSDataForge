from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


def _time_steps(t: np.ndarray) -> np.ndarray:
    if len(t) == 0:
        return np.zeros(0, dtype=float)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        fallback = float(np.mean(dt[1:]))
        dt[0] = fallback if np.isfinite(fallback) and fallback > 0 else 1.0
    else:
        dt[0] = 1.0
    return np.clip(dt, 1e-6, None)


@register_type
@dataclass(frozen=True)
class SecondOrderTracking(Component):
    reference: Component
    natural_freq: float = 0.2
    damping: float = 0.8
    gain: float = 1.0
    process_noise_std: float = 0.0
    initial_position: float = 0.0
    initial_velocity: float = 0.0
    canonical_tags = (
        "control.second_order_tracking",
        "control.closed_loop",
        "control",
        "robotics",
    )

    def children(self) -> tuple[Component, ...]:
        return (self.reference,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        ref_result = self.reference.evaluate(t, rng, context, path=f"{path}/reference:{self.reference.label()}")
        ref = np.asarray(ref_result.values, dtype=float)
        n = len(t)
        values = np.zeros(n, dtype=float)
        velocity = np.zeros(n, dtype=float)
        error = np.zeros(n, dtype=float)
        control_effort = np.zeros(n, dtype=float)
        dt = _time_steps(t)

        x = float(self.initial_position)
        v = float(self.initial_velocity)
        wn = max(float(self.natural_freq), 1e-6)
        zeta = float(self.damping)

        for i in range(n):
            r = float(self.gain) * ref[i]
            err = r - x
            u = (wn**2) * err - 2.0 * zeta * wn * v
            a = u + rng.normal(loc=0.0, scale=self.process_noise_std)
            v = v + dt[i] * a
            x = x + dt[i] * v
            values[i] = x
            velocity[i] = v
            error[i] = r - x
            control_effort[i] = u

        contributions = dict(ref_result.contributions)
        contributions[path] = values.copy()
        states = dict(ref_result.states)
        states[f"{path}/velocity"] = velocity
        states[f"{path}/error"] = error
        states[f"{path}/control_effort"] = control_effort
        tags = set(ref_result.tags)
        tags.update(self.tags())
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class EventTriggeredController(Component):
    reference: Component
    kp: float = 1.5
    plant_gain: float = 1.0
    plant_time_constant: float = 8.0
    event_threshold: float = 0.12
    max_hold_steps: int = 12
    min_inter_event_steps: int = 4
    actuator_limit: float = 2.5
    process_noise_std: float = 0.0
    initial_state: float = 0.0
    initial_control: float = 0.0
    canonical_tags = (
        "control.event_triggered",
        "control.closed_loop",
        "event_driven",
        "hybrid_system",
        "robotics",
        "control",
    )

    def children(self) -> tuple[Component, ...]:
        return (self.reference,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        ref_result = self.reference.evaluate(t, rng, context, path=f"{path}/reference:{self.reference.label()}")
        ref = np.asarray(ref_result.values, dtype=float)
        n = len(t)
        values = np.zeros(n, dtype=float)
        control_effort = np.zeros(n, dtype=float)
        error = np.zeros(n, dtype=float)
        trigger_mask = np.zeros(n, dtype=int)
        hold_counter = np.zeros(n, dtype=int)
        dt = _time_steps(t)

        x = float(self.initial_state)
        u = float(self.initial_control)
        hold = int(self.max_hold_steps)
        sampled_error = 0.0
        tau = max(float(self.plant_time_constant), 1e-6)
        limit = abs(float(self.actuator_limit))

        for i in range(n):
            r = ref[i]
            current_error = float(r - x)
            event_due = hold >= int(self.min_inter_event_steps) and abs(current_error - sampled_error) >= float(self.event_threshold)
            should_trigger = i == 0 or event_due or hold >= int(self.max_hold_steps)
            if should_trigger:
                sampled_error = current_error
                u = float(np.clip(float(self.kp) * sampled_error, -limit, limit))
                trigger_mask[i] = 1
                hold = 0
            else:
                hold += 1
            dx = dt[i] * ((-x / tau) + float(self.plant_gain) * u)
            x = x + dx + rng.normal(loc=0.0, scale=self.process_noise_std)
            values[i] = x
            control_effort[i] = u
            error[i] = r - x
            hold_counter[i] = hold

        contributions = dict(ref_result.contributions)
        contributions[path] = values.copy()
        states = dict(ref_result.states)
        states[f"{path}/control_effort"] = control_effort
        states[f"{path}/error"] = error
        states[f"{path}/trigger_mask"] = trigger_mask
        states[f"{path}/event_indices"] = np.flatnonzero(trigger_mask).astype(int)
        states[f"{path}/hold_counter"] = hold_counter
        tags = set(ref_result.tags)
        tags.update(self.tags())
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)
