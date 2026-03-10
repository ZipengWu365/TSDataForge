from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult
from ..interventions import InterventionLike, InterventionSpec, intervention_mask_values
from ..policies import ConstantPolicy, Policy



def _as_1d(x: np.ndarray | Sequence[float] | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)



def _std_vector(value: float | Sequence[float], dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(dim, float(abs(arr)), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected std dim {dim}, got {arr.size}")
    return np.abs(arr)



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



def _clip_action(u: np.ndarray, limit: float | Sequence[float] | None) -> np.ndarray:
    out = np.asarray(u, dtype=float).copy()
    if limit is None:
        return out
    lim = np.asarray(limit, dtype=float)
    if lim.ndim == 0:
        m = float(abs(lim))
        return np.clip(out, -m, m)
    lim = np.abs(lim.reshape(-1))
    if lim.size != out.size:
        raise ValueError("action_limit length must match action dimension")
    return np.minimum(np.maximum(out, -lim), lim)


@dataclass(frozen=True)
class _InterventionEntry:
    target: str
    index: int
    mode: str
    mask: np.ndarray
    values: np.ndarray
    label: str



def _prepare_interventions(interventions: Sequence[InterventionLike], n_steps: int) -> tuple[list[_InterventionEntry], dict[str, Any]]:
    entries: list[_InterventionEntry] = []
    states: dict[str, Any] = {}
    for i, item in enumerate(interventions):
        if isinstance(item, InterventionSpec):
            spec = item
        elif isinstance(item, Mapping):
            spec = InterventionSpec(
                target=str(item.get("target", item.get("kind", "input"))),
                index=int(item.get("index", item.get("var", 0))),
                start=item.get("start", item.get("at", 0)),
                end=item.get("end"),
                value=item.get("value", 0.0),
                mode=str(item.get("mode", "override")),
                name=item.get("name"),
            )
        else:
            raise TypeError(f"Unsupported intervention type: {type(item)!r}")
        mask, values = intervention_mask_values(n_steps, spec)
        target = str(spec.target).lower().strip()
        idx = int(spec.index)
        mode = str(spec.mode).lower().strip()
        label = spec.label() if hasattr(spec, "label") else f"{target}_{idx}_{i}"
        states[f"intervention/{label}/target"] = target
        states[f"intervention/{label}/index"] = idx
        states[f"intervention/{label}/mode"] = mode
        states[f"intervention/{label}/mask"] = mask.astype(int)
        states[f"intervention/{label}/values"] = values
        states[f"intervention/{target}_mask{idx}"] = np.maximum(
            np.asarray(states.get(f"intervention/{target}_mask{idx}", np.zeros(n_steps, dtype=int))),
            mask.astype(int),
        )
        entries.append(_InterventionEntry(target=target, index=idx, mode=mode, mask=mask, values=values, label=label))
    return entries, states



def _apply_interventions(vec: np.ndarray, *, entries: Sequence[_InterventionEntry], target: str, step: int) -> np.ndarray:
    out = np.asarray(vec, dtype=float).copy()
    tkey = str(target).lower().strip()
    for entry in entries:
        if entry.target != tkey or not bool(entry.mask[step]):
            continue
        if not (0 <= entry.index < out.size):
            continue
        val = float(entry.values[step])
        if entry.mode == "add":
            out[entry.index] += val
        else:
            out[entry.index] = val
    return out


@register_type
@dataclass(frozen=True)
class PolicyControlledStateSpace(Component):
    """Policy-driven linear state-space generator with optional counterfactual rollouts.

    This component unifies three needs that come up repeatedly in control- and
    causal-style time-series work:

    1. a time-driven dynamical system with explicit state / input / output,
    2. a policy that closes the loop,
    3. alternate policies and interventions for counterfactual analysis.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray | None = None
    policy: Policy | None = None
    counterfactual_policies: tuple[Policy, ...] = ()
    x0: Sequence[float] | None = None
    form: str = "discrete"  # discrete | continuous
    process_noise_std: float | Sequence[float] = 0.0
    measurement_noise_std: float | Sequence[float] = 0.0
    action_limit: float | Sequence[float] | None = None
    interventions: tuple[InterventionLike, ...] = ()
    reward_state_weight: float = 1.0
    reward_action_weight: float = 0.1
    output_names: tuple[str, ...] | None = None
    canonical_tags = (
        "control",
        "causal",
        "state_space",
        "policy_driven",
        "time_driven",
        "multivariate",
    )

    def _simulate_single(
        self,
        *,
        t: np.ndarray,
        policy: Policy,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        entries: Sequence[_InterventionEntry],
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        x_init: np.ndarray,
        dt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_steps = int(len(t))
        n_state = int(A.shape[0])
        n_in = int(B.shape[1])
        n_out = int(C.shape[0])
        X = np.zeros((n_steps, n_state), dtype=float)
        U = np.zeros((n_steps, n_in), dtype=float)
        Y = np.zeros((n_steps, n_out), dtype=float)
        R = np.zeros(n_steps, dtype=float)
        x = x_init.copy()
        prev_u = np.zeros(n_in, dtype=float)

        for k in range(n_steps):
            y_for_policy = C @ x
            if D.size:
                y_for_policy = y_for_policy + D @ prev_u
            u = _as_1d(
                policy.action(
                    state=x.copy(),
                    output=y_for_policy.copy(),
                    t_index=k,
                    time_value=float(t[k]),
                    prev_action=prev_u.copy(),
                    context={"t": t, "dt": dt},
                )
            )
            if u.size != n_in:
                if u.size == 1 and n_in > 1:
                    u = np.full(n_in, float(u[0]), dtype=float)
                else:
                    raise ValueError(f"Policy produced action dim {u.size}, expected {n_in}")
            u = _apply_interventions(u, entries=entries, target="input", step=k)
            u = _apply_interventions(u, entries=entries, target="action", step=k)
            u = _clip_action(u, self.action_limit)

            y = C @ x + D @ u + measurement_noise[k]
            y = _apply_interventions(y, entries=entries, target="output", step=k)

            X[k] = x
            U[k] = u
            Y[k] = y
            R[k] = -float(self.reward_state_weight) * float(np.sum(x**2)) - float(self.reward_action_weight) * float(np.sum(u**2))

            if str(self.form).lower().startswith("cont"):
                x_next = x + dt[k] * (A @ x + B @ u) + process_noise[k]
            else:
                x_next = A @ x + B @ u + process_noise[k]
            x_next = _apply_interventions(x_next, entries=entries, target="state", step=k)
            x = x_next
            prev_u = u
        return X, U, Y, R

    def _evaluate(self, t: np.ndarray, rng: np.random.Generator, context: Mapping[str, Any], path: str) -> EvalResult:
        del context
        n_steps = int(len(t))
        A = np.asarray(self.A, dtype=float)
        B = np.asarray(self.B, dtype=float)
        C = np.asarray(self.C, dtype=float)
        D = np.zeros((C.shape[0], B.shape[1]), dtype=float) if self.D is None else np.asarray(self.D, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError("B must have shape (n_state, n_input)")
        if C.ndim != 2 or C.shape[1] != A.shape[0]:
            raise ValueError("C must have shape (n_output, n_state)")
        if D.ndim != 2 or D.shape != (C.shape[0], B.shape[1]):
            raise ValueError("D must have shape (n_output, n_input)")

        n_state = int(A.shape[0])
        n_in = int(B.shape[1])
        n_out = int(C.shape[0])
        dt = _time_steps(np.asarray(t, dtype=float))
        if self.x0 is None:
            x_init = np.zeros(n_state, dtype=float)
        else:
            x_init = _as_1d(self.x0)
            if x_init.size != n_state:
                raise ValueError("x0 length must match n_state")

        if self.policy is None:
            policy = ConstantPolicy(action_value=np.zeros(n_in, dtype=float), name="zero_policy")
        else:
            policy = self.policy

        process_std = _std_vector(self.process_noise_std, n_state)
        meas_std = _std_vector(self.measurement_noise_std, n_out)
        process_noise = rng.normal(scale=process_std, size=(n_steps, n_state)) if n_steps else np.zeros((0, n_state))
        measurement_noise = rng.normal(scale=meas_std, size=(n_steps, n_out)) if n_steps else np.zeros((0, n_out))

        entries, iv_states = _prepare_interventions(self.interventions, n_steps)
        X, U, Y, R = self._simulate_single(
            t=np.asarray(t, dtype=float),
            policy=policy,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            entries=entries,
            A=A,
            B=B,
            C=C,
            D=D,
            x_init=x_init,
            dt=dt,
        )

        output_names = self.output_names or tuple(f"y{i}" for i in range(n_out))
        states: dict[str, Any] = {
            f"{path}/state": X,
            f"{path}/input": U,
            f"{path}/output": Y,
            f"{path}/reward": R,
            f"{path}/return": float(np.sum(R)),
            f"{path}/A": A,
            f"{path}/B": B,
            f"{path}/C": C,
            f"{path}/D": D,
            f"{path}/n_state": n_state,
            f"{path}/n_input": n_in,
            f"{path}/n_output": n_out,
            f"{path}/policy_name": policy.label(),
            f"{path}/output_names": tuple(output_names),
        }
        for key, value in iv_states.items():
            states[f"{path}/{key}"] = value

        tags = set(self.tags())
        if self.interventions:
            tags.add("intervention")
        if self.counterfactual_policies:
            tags.add("counterfactual")

        for alt in self.counterfactual_policies:
            Xcf, Ucf, Ycf, Rcf = self._simulate_single(
                t=np.asarray(t, dtype=float),
                policy=alt,
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                entries=entries,
                A=A,
                B=B,
                C=C,
                D=D,
                x_init=x_init,
                dt=dt,
            )
            label = alt.label()
            states[f"{path}/counterfactual/{label}/state"] = Xcf
            states[f"{path}/counterfactual/{label}/input"] = Ucf
            states[f"{path}/counterfactual/{label}/output"] = Ycf
            states[f"{path}/counterfactual/{label}/reward"] = Rcf
            states[f"{path}/counterfactual/{label}/return"] = float(np.sum(Rcf))
            if Ycf.shape == Y.shape:
                states[f"{path}/counterfactual/{label}/delta_output"] = Ycf - Y
                states[f"{path}/counterfactual/{label}/delta_return"] = float(np.sum(Rcf) - np.sum(R))

        contributions = {path: Y.copy()}
        return EvalResult(values=Y, contributions=contributions, states=states, tags=tags)
