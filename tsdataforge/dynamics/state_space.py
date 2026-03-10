from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected 1D or 2D array")


@register_type
@dataclass(frozen=True)
class LinearStateSpace(Component):
    """Linear state-space simulator.

    This component generates a multivariate output sequence while keeping the
    full ground-truth state and input in the trace.

    Model
    -----
    Continuous-time (Euler integration):
        x_{k+1} = x_k + dt * (A x_k + B u_k) + w_k
        y_k     = C x_k + D u_k + v_k

    Discrete-time:
        x_{k+1} = A x_k + B u_k + w_k
        y_k     = C x_k + D u_k + v_k

    Notes
    -----
    - `input` may be a scalar (T,) / vector (T,m) component.
    - If `input` is None, the simulator uses zero input by default.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray | None = None
    input: Component | None = None
    form: str = "continuous"  # "continuous" | "discrete"
    process_noise_std: float | Sequence[float] = 0.0
    measurement_noise_std: float | Sequence[float] = 0.0
    x0: Sequence[float] | None = None
    canonical_tags = (
        "state_space",
        "control",
        "multivariate",
    )

    def children(self) -> tuple[Component, ...]:
        return () if self.input is None else (self.input,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, Any],
        path: str,
    ) -> EvalResult:
        del context
        A = np.asarray(self.A, dtype=float)
        B = np.asarray(self.B, dtype=float)
        C = np.asarray(self.C, dtype=float)
        D = np.zeros((C.shape[0], B.shape[1]), dtype=float) if self.D is None else np.asarray(self.D, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        n = int(A.shape[0])
        if B.ndim != 2 or B.shape[0] != n:
            raise ValueError("B must have shape (n_state, n_input)")
        m = int(B.shape[1])
        if C.ndim != 2 or C.shape[1] != n:
            raise ValueError("C must have shape (n_output, n_state)")
        p = int(C.shape[0])
        if D.shape != (p, m):
            raise ValueError("D must have shape (n_output, n_input)")

        dt = _time_steps(t)
        if self.input is None:
            U = np.zeros((len(t), m), dtype=float)
            input_result: EvalResult | None = None
        else:
            input_result = self.input.evaluate(t, rng, context={}, path=f"{path}/input:{self.input.label()}")
            U = _as_2d(input_result.values)
            if U.shape[0] != len(t):
                raise ValueError("Input length mismatch")
            if U.shape[1] == 1 and m > 1:
                U = np.repeat(U, m, axis=1)
            if U.shape[1] != m:
                raise ValueError(f"Input dim mismatch: expected {m}, got {U.shape[1]}")

        if self.x0 is None:
            x = np.zeros(n, dtype=float)
        else:
            x = np.asarray(self.x0, dtype=float).reshape(-1)
            if x.shape[0] != n:
                raise ValueError(f"x0 must have length {n}")

        proc_std = np.asarray(self.process_noise_std, dtype=float)
        if proc_std.ndim == 0:
            proc_std = np.full(n, float(proc_std), dtype=float)
        if proc_std.shape != (n,):
            raise ValueError("process_noise_std must be a scalar or length-n_state sequence")
        meas_std = np.asarray(self.measurement_noise_std, dtype=float)
        if meas_std.ndim == 0:
            meas_std = np.full(p, float(meas_std), dtype=float)
        if meas_std.shape != (p,):
            raise ValueError("measurement_noise_std must be a scalar or length-n_output sequence")

        X = np.zeros((len(t), n), dtype=float)
        Y = np.zeros((len(t), p), dtype=float)

        discrete = str(self.form).lower().startswith("disc")
        for k in range(len(t)):
            u = U[k]
            if discrete:
                x_next = A @ x + B @ u
            else:
                x_next = x + dt[k] * (A @ x + B @ u)
            if float(np.max(proc_std)) > 0:
                # scale by sqrt(dt) in continuous mode
                scale = proc_std * (np.sqrt(dt[k]) if not discrete else 1.0)
                x_next = x_next + rng.normal(loc=0.0, scale=scale, size=n)
            x = x_next
            y = C @ x + D @ u
            if float(np.max(meas_std)) > 0:
                y = y + rng.normal(loc=0.0, scale=meas_std, size=p)
            X[k] = x
            Y[k] = y

        contributions = {} if input_result is None else dict(input_result.contributions)
        contributions[path] = Y.copy()
        states = {} if input_result is None else dict(input_result.states)
        states[f"{path}/A"] = A
        states[f"{path}/B"] = B
        states[f"{path}/C"] = C
        states[f"{path}/D"] = D
        states[f"{path}/state"] = X
        states[f"{path}/input"] = U
        states[f"{path}/output"] = Y
        states[f"{path}/n_state"] = n
        states[f"{path}/n_input"] = m
        states[f"{path}/n_output"] = p
        tags = set(self.tags())
        if input_result is not None:
            tags.update(input_result.tags)
        return EvalResult(values=Y, contributions=contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class JointServoMIMO(Component):
    """A simple MIMO joint servo model producing multi-channel robot sensor traces.

    State
    -----
    x = [q, dq] where q, dq are per-joint position/velocity.

    Control
    -------
    PD control (optionally event-triggered):
        u = Kp (r - q) - Kd dq

    Plant (per joint)
    -----------------
        ddq = -2 zeta wn dq - wn^2 q + gain * u

    Output
    ------
    Concatenated sensor channels (selected by `output_channels`).

    Supported channels: "q", "dq", "u", "error", "gyro", "acc".
    "gyro" is approximated by dq and "acc" by ddq.
    """

    n_joints: int = 6
    reference: Component | None = None
    wn: float = 0.18
    zeta: float = 0.9
    kp: float = 2.0
    kd: float = 0.6
    plant_gain: float = 1.0
    actuator_limit: float = 3.0
    process_noise_std: float = 0.0
    measurement_noise_std: float = 0.02
    initial_position: float = 0.0
    initial_velocity: float = 0.0
    # Optional event-triggered updates (set threshold > 0)
    event_threshold: float = 0.0
    max_hold_steps: int = 10
    min_inter_event_steps: int = 2
    output_channels: tuple[str, ...] = ("q", "dq", "u")
    canonical_tags = (
        "robotics",
        "control",
        "state_space",
        "mimo",
        "multivariate",
        "robot_sensors",
    )

    def children(self) -> tuple[Component, ...]:
        return () if self.reference is None else (self.reference,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, Any],
        path: str,
    ) -> EvalResult:
        del context
        n = int(self.n_joints)
        if n <= 0:
            raise ValueError("n_joints must be positive")
        dt = _time_steps(t)

        if self.reference is None:
            R = np.zeros((len(t), n), dtype=float)
            ref_result: EvalResult | None = None
        else:
            ref_result = self.reference.evaluate(t, rng, context={}, path=f"{path}/reference:{self.reference.label()}")
            R = _as_2d(ref_result.values)
            if R.shape[0] != len(t):
                raise ValueError("Reference length mismatch")
            if R.shape[1] == 1 and n > 1:
                R = np.repeat(R, n, axis=1)
            if R.shape[1] != n:
                raise ValueError(f"Reference dim mismatch: expected {n}, got {R.shape[1]}")

        q = np.full(n, float(self.initial_position), dtype=float)
        dq = np.full(n, float(self.initial_velocity), dtype=float)
        u = np.zeros(n, dtype=float)
        last_err = np.zeros(n, dtype=float)
        hold = int(self.max_hold_steps)
        limit = abs(float(self.actuator_limit))

        Q = np.zeros((len(t), n), dtype=float)
        DQ = np.zeros((len(t), n), dtype=float)
        U = np.zeros((len(t), n), dtype=float)
        ERR = np.zeros((len(t), n), dtype=float)
        DDQ = np.zeros((len(t), n), dtype=float)
        trigger = np.zeros(len(t), dtype=int)

        wn = max(float(self.wn), 1e-6)
        zeta = float(self.zeta)
        kp = float(self.kp)
        kd = float(self.kd)
        gain = float(self.plant_gain)
        thr = float(self.event_threshold)
        min_inter = int(self.min_inter_event_steps)
        max_hold = int(self.max_hold_steps)

        for k in range(len(t)):
            r = R[k]
            err = r - q

            should_trigger = True
            if thr > 0:
                delta = np.max(np.abs(err - last_err))
                event_due = hold >= min_inter and delta >= thr
                should_trigger = k == 0 or event_due or hold >= max_hold
            if should_trigger:
                u = np.clip(kp * err - kd * dq, -limit, limit)
                last_err = err.copy()
                trigger[k] = 1
                hold = 0
            else:
                hold += 1

            ddq = (-2.0 * zeta * wn) * dq - (wn**2) * q + gain * u
            if float(self.process_noise_std) > 0:
                ddq = ddq + rng.normal(loc=0.0, scale=float(self.process_noise_std), size=n)
            dq = dq + dt[k] * ddq
            q = q + dt[k] * dq

            Q[k] = q
            DQ[k] = dq
            U[k] = u
            ERR[k] = err
            DDQ[k] = ddq

        # Build output sensors
        channel_map = {
            "q": Q,
            "dq": DQ,
            "u": U,
            "error": ERR,
            "gyro": DQ,
            "acc": DDQ,
        }
        selected: list[np.ndarray] = []
        names: list[str] = []
        for ch in self.output_channels:
            key = str(ch)
            if key not in channel_map:
                raise ValueError(f"Unsupported output channel: {ch!r}")
            block = channel_map[key]
            selected.append(block)
            for j in range(n):
                names.append(f"{key}_{j}")
        Y = np.concatenate(selected, axis=1) if selected else np.zeros((len(t), 0), dtype=float)
        if float(self.measurement_noise_std) > 0:
            Y = Y + rng.normal(loc=0.0, scale=float(self.measurement_noise_std), size=Y.shape)

        contributions = {} if ref_result is None else dict(ref_result.contributions)
        contributions[path] = Y.copy()
        states: dict[str, Any] = {} if ref_result is None else dict(ref_result.states)
        states[f"{path}/state"] = np.concatenate([Q, DQ], axis=1)
        states[f"{path}/q"] = Q
        states[f"{path}/dq"] = DQ
        states[f"{path}/ddq"] = DDQ
        states[f"{path}/input"] = U
        states[f"{path}/error"] = ERR
        states[f"{path}/reference"] = R
        states[f"{path}/trigger_mask"] = trigger
        states[f"{path}/event_indices"] = np.flatnonzero(trigger).astype(int)
        states[f"{path}/output_names"] = names

        tags = set(self.tags())
        if thr > 0:
            tags.update({"event_driven", "control.event_triggered"})
        if ref_result is not None:
            tags.update(ref_result.tags)
        return EvalResult(values=Y, contributions=contributions, states=states, tags=tags)
