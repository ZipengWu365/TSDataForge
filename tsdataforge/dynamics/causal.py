from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult
from ..interventions import InterventionLike, InterventionSpec, intervention_mask_values


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected 1D or 2D array")


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


def _parse_interventions(interventions: Sequence[InterventionLike], n: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Convert intervention payloads into per-variable value vectors + masks.

    Supported payloads
    ------------------
    - legacy mappings: {"var": i, "at": [...], "value": ...}
    - interval mappings: {"index": i, "start": ..., "end": ..., "value": ...}
    - :class:`~tsdataforge.interventions.InterventionSpec`
    """

    values_map: dict[int, np.ndarray] = {}
    mask_map: dict[int, np.ndarray] = {}
    for item in interventions:
        if isinstance(item, InterventionSpec):
            var = int(item.index)
            mask, values = intervention_mask_values(n, item)
        elif isinstance(item, Mapping):
            if "var" in item or "index" in item:
                var = int(item.get("var", item.get("index", 0)))
            else:
                continue
            mask, values = intervention_mask_values(n, item)
        else:
            raise TypeError(f"Unsupported intervention type: {type(item)!r}")

        if var not in mask_map:
            mask_map[var] = np.zeros(n, dtype=bool)
            values_map[var] = np.zeros(n, dtype=float)
        mask_map[var] = np.asarray(mask_map[var] | mask.astype(bool), dtype=bool)
        overwrite = np.asarray(mask, dtype=bool)
        values_map[var][overwrite] = np.asarray(values, dtype=float)[overwrite]
    return values_map, mask_map


def _apply_nonlinearity(x: np.ndarray, name: str) -> np.ndarray:
    key = str(name).lower().strip()
    if key in {"linear", "none", "identity"}:
        return x
    if key in {"tanh"}:
        return np.tanh(x)
    if key in {"relu"}:
        return np.maximum(0.0, x)
    if key in {"sigmoid"}:
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unsupported nonlinearity: {name!r}")


@register_type
@dataclass(frozen=True)
class CausalVARX(Component):
    """Causal (time-lag) VARX generator with optional exogenous drivers and do-interventions.

    Model (lag-L)
    -------------
        x_t = b + sum_{l=1..L} A[l-1] x_{t-l} + B u_t + eps_t

    Notes
    -----
    - This component is intended for *causal / control-friendly* synthetic datasets.
    - Causality here is encoded as a directed lag graph (parents at t-1..t-L).
    - `interventions` implement a simple do-operator: selected variables are overridden at
      specified time indices.
    """

    A: np.ndarray  # (L, n, n) or (n,n)
    B: np.ndarray | None = None  # (n, m)
    bias: np.ndarray | None = None  # (n,)
    exogenous: Component | None = None
    noise_std: float | Sequence[float] = 0.1
    x0: Sequence[float] | None = None
    nonlinearity: str = "linear"
    interventions: tuple[InterventionLike, ...] = ()
    canonical_tags = (
        "causal",
        "varx",
        "multivariate",
        "time_driven",
    )

    def children(self) -> tuple[Component, ...]:
        return () if self.exogenous is None else (self.exogenous,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, Any],
        path: str,
    ) -> EvalResult:
        del context
        n_steps = int(len(t))
        if n_steps == 0:
            return EvalResult(values=np.zeros((0, 0), dtype=float))

        A = np.asarray(self.A, dtype=float)
        if A.ndim == 2:
            A = A[None, :, :]
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError("A must have shape (L,n,n) or (n,n)")
        L = int(A.shape[0])
        n = int(A.shape[1])
        if n <= 0:
            raise ValueError("n_vars must be positive")

        bias = np.zeros(n, dtype=float) if self.bias is None else np.asarray(self.bias, dtype=float).reshape(-1)
        if bias.shape != (n,):
            raise ValueError("bias must have shape (n_vars,)")

        # Exogenous input
        if self.exogenous is None:
            if self.B is None:
                U = np.zeros((n_steps, 0), dtype=float)
                B = None
            else:
                B = np.asarray(self.B, dtype=float)
                if B.ndim != 2 or B.shape[0] != n:
                    raise ValueError("B must have shape (n_vars, n_exog)")
                U = np.zeros((n_steps, int(B.shape[1])), dtype=float)
        else:
            if self.B is None:
                raise ValueError("When exogenous is provided, B must also be provided.")
            B = np.asarray(self.B, dtype=float)
            if B.ndim != 2 or B.shape[0] != n:
                raise ValueError("B must have shape (n_vars, n_exog)")
            ex = self.exogenous.evaluate(t, rng, context={}, path=f"{path}/exogenous:{self.exogenous.label()}")
            U = _as_2d(ex.values)
            if U.shape[0] != n_steps:
                raise ValueError("Exogenous length mismatch")
            if U.shape[1] != int(B.shape[1]):
                raise ValueError(f"Exogenous dim mismatch: expected {int(B.shape[1])}, got {U.shape[1]}")

        # Noise std
        std = np.asarray(self.noise_std, dtype=float)
        if std.ndim == 0:
            std = np.full(n, float(std), dtype=float)
        if std.shape != (n,):
            raise ValueError("noise_std must be a scalar or length-n_vars sequence")

        # Initial history for negative indices
        if self.x0 is None:
            x_init = np.zeros(n, dtype=float)
        else:
            x_init = np.asarray(self.x0, dtype=float).reshape(-1)
            if x_init.shape != (n,):
                raise ValueError("x0 must have length n_vars")
        history = np.tile(x_init[None, :], (L, 1))

        # Interventions
        iv_values, iv_masks = _parse_interventions(self.interventions, n_steps)

        X = np.zeros((n_steps, n), dtype=float)
        for k in range(n_steps):
            x = bias.copy()
            for lag in range(1, L + 1):
                prev = X[k - lag] if k - lag >= 0 else history[lag - 1]
                x = x + A[lag - 1] @ prev
            if B is not None and U.size:
                x = x + B @ U[k]
            x = _apply_nonlinearity(x, self.nonlinearity)
            if float(np.max(std)) > 0:
                x = x + rng.normal(loc=0.0, scale=std, size=n)
            # Apply do-interventions (override)
            for var, m in iv_masks.items():
                if 0 <= var < n and bool(m[k]):
                    x[var] = float(iv_values[var][k])
            X[k] = x

        adjacency = (np.sum(np.abs(A), axis=0) > 0).astype(int)
        states: dict[str, Any] = {
            f"{path}/A": A,
            f"{path}/lags": L,
            f"{path}/n_vars": n,
            f"{path}/bias": bias,
            f"{path}/adjacency": adjacency,
        }
        if B is not None:
            states[f"{path}/B"] = B
            states[f"{path}/input"] = U
            states[f"{path}/n_exog"] = int(B.shape[1])
        if self.interventions:
            # Store as compact masks per variable.
            for var, m in iv_masks.items():
                states[f"{path}/do_mask_var{var}"] = m.astype(int)
                states[f"{path}/do_value_var{var}"] = iv_values[var]

        contributions: dict[str, np.ndarray] = {path: X.copy()}
        tags = set(self.tags())
        if B is not None and U.size:
            tags.add("control")
            tags.add("exogenous")
        if self.interventions:
            tags.add("intervention")
        return EvalResult(values=X, contributions=contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class CausalTreatmentOutcome(Component):
    """A small dynamic SCM for treatment/outcome time series with confounding.

    Generates three channels:
        [confounder C_t, treatment A_t, outcome Y_t]

    and (optionally) stores potential outcomes and ITE in the trace.

    This component is intentionally simple and explainable; it is meant as a
    *benchmark surface* for time-series causal inference.
    """

    confounder_ar: float = 0.9
    outcome_ar: float = 0.7
    treatment_confounding: float = 1.0
    outcome_confounding: float = 1.0
    treatment_effect: float = 1.0
    treatment_lag: int = 1
    treatment_type: str = "binary"  # "binary" | "continuous"
    noise_std: tuple[float, float, float] = (0.3, 0.3, 0.3)
    intervention: InterventionLike | None = None
    return_potential_outcomes: bool = True
    canonical_tags = (
        "causal",
        "treatment_outcome",
        "time_driven",
        "confounding",
        "multivariate",
    )

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, Any],
        path: str,
    ) -> EvalResult:
        del context
        n = int(len(t))
        if n == 0:
            return EvalResult(values=np.zeros((0, 3), dtype=float))

        # Pre-draw noise so potential outcomes share the same exogenous randomness.
        std_c, std_a, std_y = (abs(float(s)) for s in self.noise_std)
        eps_c = rng.normal(scale=std_c, size=n)
        eps_a = rng.normal(scale=std_a, size=n)
        eps_y = rng.normal(scale=std_y, size=n)

        # Intervention schedule on treatment
        interventions: tuple[InterventionLike, ...] = ()
        if self.intervention is not None:
            # Normalize to the same format as CausalVARX
            interventions = (self.intervention,)
        iv_values, iv_masks = _parse_interventions(interventions, n)
        treat_mask = iv_masks.get(1, np.zeros(n, dtype=bool))

        def _simulate(force_treatment: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            C = np.zeros(n, dtype=float)
            A = np.zeros(n, dtype=float)
            Y = np.zeros(n, dtype=float)
            conf_ar = float(self.confounder_ar)
            out_ar = float(self.outcome_ar)
            lag = int(max(0, self.treatment_lag))

            for k in range(n):
                c_prev = C[k - 1] if k > 0 else 0.0
                y_prev = Y[k - 1] if k > 0 else 0.0
                C[k] = conf_ar * c_prev + eps_c[k]

                if force_treatment is None:
                    a_lat = float(self.treatment_confounding) * C[k] + 0.15 * y_prev + eps_a[k]
                    if str(self.treatment_type).lower().startswith("bin"):
                        A[k] = 1.0 if a_lat > 0.0 else 0.0
                    else:
                        A[k] = a_lat
                else:
                    A[k] = float(force_treatment)

                # Apply scheduled intervention overrides (do operator)
                if force_treatment is None and 1 in iv_masks and bool(iv_masks[1][k]):
                    A[k] = float(iv_values[1][k])

                a_lag = A[k - lag] if k - lag >= 0 else 0.0
                Y[k] = out_ar * y_prev + float(self.outcome_confounding) * C[k] + float(self.treatment_effect) * a_lag + eps_y[k]
            return C, A, Y

        C, A, Y = _simulate(force_treatment=None)

        values = np.stack([C, A, Y], axis=1)
        states: dict[str, Any] = {
            f"{path}/confounder": C,
            f"{path}/treatment": A,
            f"{path}/outcome": Y,
            f"{path}/treatment_index": 1,
            f"{path}/outcome_index": 2,
            f"{path}/treatment_type": str(self.treatment_type),
            f"{path}/treatment_lag": int(self.treatment_lag),
        }
        if self.intervention is not None:
            states[f"{path}/intervention_mask"] = treat_mask.astype(int)
            states[f"{path}/intervention"] = self.intervention.to_dict() if hasattr(self.intervention, "to_dict") else dict(self.intervention)

        if self.return_potential_outcomes and str(self.treatment_type).lower().startswith("bin"):
            _, _, y0 = _simulate(force_treatment=0.0)
            _, _, y1 = _simulate(force_treatment=1.0)
            ite = y1 - y0
            states[f"{path}/potential_outcome_do0"] = y0
            states[f"{path}/potential_outcome_do1"] = y1
            states[f"{path}/ite"] = ite

        contributions = {path: values.copy()}
        tags = set(self.tags())
        if self.intervention is not None:
            tags.add("intervention")
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)
