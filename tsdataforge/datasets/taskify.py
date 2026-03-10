from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..self_supervised.contrastive import make_contrastive_pair
from ..self_supervised.masking import apply_mask, block_mask
from ..self_supervised.order import segment_shuffle
from .builder import TaskDataset
from .series_dataset import SeriesDataset, _stack_or_object



def _extract_changepoint_target(trace_states: dict[str, Any], length: int) -> np.ndarray:
    target = np.zeros(length, dtype=int)
    for key, value in trace_states.items():
        if key.endswith("/changepoints"):
            idx = np.asarray(value, dtype=int)
            idx = idx[(idx >= 0) & (idx < length)]
            target[idx] = 1
    return target



def _extract_event_target(trace_states: dict[str, Any], length: int) -> np.ndarray:
    target = np.zeros(length, dtype=int)
    vector_suffixes = ("/trigger_mask", "/event_mask", "/impulses")
    index_suffixes = ("/event_indices", "/events")
    for key, value in trace_states.items():
        if key.endswith(vector_suffixes):
            arr = np.asarray(value)
            if arr.ndim == 1 and len(arr) == length:
                target = np.maximum(target, (arr > 0).astype(int))
        elif key.endswith(index_suffixes):
            idx = np.asarray(value, dtype=int)
            idx = idx[(idx >= 0) & (idx < length)]
            target[idx] = 1
    return target



def _extract_intervention_target(trace_states: dict[str, Any], length: int) -> np.ndarray:
    target = np.zeros(length, dtype=int)
    suffixes = (
        "/intervention_mask",
        "/action_mask",
        "/state_mask",
        "/output_mask",
        "/do_mask",
    )
    for key, value in trace_states.items():
        if "/intervention/" in key and key.endswith("/mask"):
            arr = np.asarray(value)
        elif key.endswith(suffixes) or "_mask" in key and ("intervention" in key or "/do_mask_var" in key):
            arr = np.asarray(value)
        else:
            continue
        if arr.ndim == 1 and len(arr) == length:
            target = np.maximum(target, (arr > 0).astype(int))
    return target



def _inject_anomalies(values: np.ndarray, rng: np.random.Generator, anomaly_rate: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(values, dtype=float).copy()
    n = len(out)
    mask = np.zeros(n, dtype=int)
    if n == 0:
        return out, mask
    k = max(1, int(round(n * anomaly_rate)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    scale = np.nanstd(out)
    scale = 1.0 if not np.isfinite(scale) or scale == 0 else scale

    if out.ndim == 1:
        out[idx] += rng.normal(scale=5.0 * scale, size=len(idx))
    elif out.ndim == 2:
        d = out.shape[1]
        ch = rng.integers(0, d, size=len(idx))
        out[idx, ch] += rng.normal(scale=5.0 * scale, size=len(idx))
    else:
        raise ValueError("Anomaly injection supports 1D or 2D arrays")
    mask[idx] = 1

    if n >= 10:
        start = int(rng.integers(0, max(1, n - max(2, n // 10))))
        width = int(rng.integers(2, max(3, n // 10)))
        end = min(n, start + width)
        if out.ndim == 1:
            out[start:end] += 3.0 * scale
        else:
            out[start:end, :] += rng.normal(scale=3.0 * scale, size=out[start:end, :].shape)
        mask[start:end] = 1
    return out, mask



def _first_suffix(states: dict[str, Any], suffix: str) -> Any:
    for k, v in states.items():
        if k.endswith(suffix):
            return v
    return None



def _first_counterfactual_output(states: dict[str, Any]) -> tuple[str | None, np.ndarray | None]:
    keys = sorted(k for k in states if "/counterfactual/" in k and k.endswith("/output"))
    if not keys:
        return None, None
    key = keys[0]
    return key, np.asarray(states[key], dtype=float)



def _labels_from_meta(meta: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    names = [m.get("structure_id", "unknown") for m in meta]
    uniq: list[str] = []
    for n in names:
        if n not in uniq:
            uniq.append(n)
    return uniq, {name: i for i, name in enumerate(uniq)}


@dataclass(frozen=True)
class TaskSpec:
    """A minimal, serializable task definition for agent-friendly pipelines."""

    task: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "params": dict(self.params)}



def taskify_dataset(
    dataset: SeriesDataset,
    *,
    task: str,
    horizon: int = 32,
    mask_ratio: float = 0.15,
    order_segments: int = 4,
    anomaly_rate: float = 0.02,
    seed: int | None = None,
    # Forecasting-like windowing
    window: int | None = None,
    stride: int = 1,
    # Causal/task semantics
    outcome_channel: int | None = None,
    include_aux: bool = True,
    gamma: float = 1.0,
) -> TaskDataset:
    """Convert a base `SeriesDataset` into a task-specific `TaskDataset`."""

    rng = np.random.default_rng(int(seed) if seed is not None else 0)
    task = str(task)

    X_list: list[np.ndarray] = []
    y_list: list[Any] = []
    time_list: list[np.ndarray] = []
    mask_map: dict[str, list[np.ndarray]] = {}
    aux_map: dict[str, list[np.ndarray]] = {}
    meta_out: list[dict[str, Any]] = []

    label_names, label_to_index = _labels_from_meta(dataset.meta)

    def _emit_sample(x: np.ndarray, y: Any | None, t: np.ndarray | None, m: dict[str, np.ndarray] | None, aux: dict[str, np.ndarray] | None, meta: dict[str, Any]):
        X_list.append(np.asarray(x))
        if y is not None:
            y_list.append(y)
        if t is not None:
            time_list.append(np.asarray(t))
        if m:
            for k, v in m.items():
                mask_map.setdefault(k, []).append(np.asarray(v))
        if aux:
            for k, v in aux.items():
                aux_map.setdefault(k, []).append(np.asarray(v))
        meta_out.append(meta)

    for s, m in zip(dataset.series, dataset.meta or [{}] * len(dataset.series)):
        values = np.asarray(s.values, dtype=float)
        time = np.asarray(s.time, dtype=float)
        trace_states = {} if s.trace is None else dict(s.trace.states)
        structure_id = str(m.get("structure_id") or (s.spec.structure_id if s.spec else "unknown") or "unknown")

        label = int(label_to_index.get(structure_id, -1))

        if window is None:
            if task == "classification":
                _emit_sample(values, label, time, None, None, dict(m))
            elif task in {"forecasting", "future_prediction"}:
                if horizon <= 0 or horizon >= len(values):
                    raise ValueError("`horizon` must be between 1 and len(series)-1")
                _emit_sample(values[:-horizon], values[-horizon:], time[:-horizon], None, None, dict(m))
            elif task == "masked_reconstruction":
                mask = block_mask(len(values), rng, mask_ratio=mask_ratio)
                _emit_sample(apply_mask(values, mask, fill_value=0.0), values, time, {"pretext_mask": mask}, None, dict(m))
            elif task == "temporal_order":
                shuffled, order = segment_shuffle(values, rng, n_segments=order_segments)
                _emit_sample(shuffled, order, time, None, None, dict(m))
            elif task == "contrastive":
                v1, v2 = make_contrastive_pair(values, rng)
                _emit_sample(np.stack([v1, v2], axis=0), label, time, None, None, dict(m))
            elif task == "anomaly_detection":
                corrupted, anomaly_mask = _inject_anomalies(values, rng, anomaly_rate=anomaly_rate)
                _emit_sample(corrupted, anomaly_mask, time, None, None, dict(m))
            elif task == "change_point_detection":
                cp_target = _extract_changepoint_target(trace_states, len(values))
                _emit_sample(values, cp_target, time, None, None, dict(m))
            elif task == "event_detection":
                ev_target = _extract_event_target(trace_states, len(values))
                _emit_sample(values, ev_target, time, None, None, dict(m))
            elif task == "intervention_detection":
                iv_target = _extract_intervention_target(trace_states, len(values))
                _emit_sample(values, iv_target, time, None, None, dict(m))
            elif task == "system_identification":
                u = _first_suffix(trace_states, "/input")
                x = _first_suffix(trace_states, "/state")
                if u is None:
                    raise ValueError("system_identification requires '/input' in trace.states")
                u = np.asarray(u, dtype=float)
                if u.ndim == 1:
                    u = u[:, None]
                if horizon <= 0 or horizon >= len(values):
                    raise ValueError("`horizon` must be between 1 and len(series)-1")
                X = np.concatenate([u[:-horizon], values[:-horizon]], axis=-1)
                y = values[-horizon:]
                aux = {"u": u}
                if x is not None:
                    aux["x"] = np.asarray(x, dtype=float)
                _emit_sample(X, y, time[:-horizon], None, aux, dict(m))
            elif task == "causal_response":
                if horizon <= 0 or horizon >= len(values):
                    raise ValueError("`horizon` must be between 1 and len(series)-1")
                out_idx = outcome_channel
                if out_idx is None:
                    out_idx = _first_suffix(trace_states, "/outcome_index")
                    out_idx = int(out_idx) if out_idx is not None else (values.shape[1] - 1 if values.ndim == 2 else 0)
                if values.ndim == 1:
                    raise ValueError("causal_response expects a multivariate series")
                if not (0 <= int(out_idx) < int(values.shape[1])):
                    raise ValueError("Invalid outcome_channel")
                X = values[:-horizon]
                y = values[-horizon:, int(out_idx)]
                _emit_sample(X, y, time[:-horizon], None, None, dict(m))
            elif task == "counterfactual_response":
                if horizon <= 0 or horizon >= len(values):
                    raise ValueError("`horizon` must be between 1 and len(series)-1")
                aux: dict[str, np.ndarray] = {}
                cf_key, cf_output = _first_counterfactual_output(trace_states)
                if cf_output is not None:
                    out_idx = int(outcome_channel) if outcome_channel is not None else (cf_output.shape[1] - 1 if cf_output.ndim == 2 else 0)
                    y = cf_output[-horizon:, out_idx] if cf_output.ndim == 2 else cf_output[-horizon:]
                    if include_aux:
                        aux["counterfactual_output"] = cf_output
                        aux["counterfactual_key"] = np.asarray([0])  # placeholder scalar for manifest compatibility
                else:
                    y0 = _first_suffix(trace_states, "/potential_outcome_do0")
                    y1 = _first_suffix(trace_states, "/potential_outcome_do1")
                    ite = _first_suffix(trace_states, "/ite")
                    if y0 is None or y1 is None:
                        raise ValueError("counterfactual_response requires counterfactual outputs or potential outcomes in trace.states")
                    y0 = np.asarray(y0, dtype=float)
                    y1 = np.asarray(y1, dtype=float)
                    y = np.stack([y0[-horizon:], y1[-horizon:]], axis=-1)
                    if include_aux and ite is not None:
                        aux["ite"] = np.asarray(ite, dtype=float)
                _emit_sample(values[:-horizon], y, time[:-horizon], None, aux if (include_aux and aux) else None, dict(m))
            elif task == "policy_value_estimation":
                reward = _first_suffix(trace_states, "/reward")
                if reward is None:
                    raise ValueError("policy_value_estimation requires '/reward' in trace.states")
                reward = np.asarray(reward, dtype=float).reshape(-1)
                disc = np.power(float(gamma), np.arange(len(reward), dtype=float))
                ret = float(np.sum(disc * reward))
                aux = {"reward": reward} if include_aux else None
                _emit_sample(values, np.asarray(ret, dtype=float), time, None, aux, dict(m))
            elif task == "causal_ite":
                ite = _first_suffix(trace_states, "/ite")
                if ite is None:
                    raise ValueError("causal_ite requires '/ite' in trace.states (use CausalTreatmentOutcome)")
                ite = np.asarray(ite, dtype=float)
                if horizon <= 0 or horizon >= len(values):
                    raise ValueError("`horizon` must be between 1 and len(series)-1")
                X = values[:-horizon]
                y = ite[-horizon:]
                _emit_sample(X, y, time[:-horizon], None, None, dict(m))
            elif task == "causal_discovery":
                adj = _first_suffix(trace_states, "/adjacency")
                if adj is None:
                    raise ValueError("causal_discovery requires '/adjacency' in trace.states")
                _emit_sample(values, np.asarray(adj, dtype=int), time, None, None, dict(m))
            else:
                raise ValueError(f"Unsupported task={task!r}")
        else:
            w = int(window)
            st = int(max(1, stride))
            if w <= 0:
                raise ValueError("window must be positive")
            if horizon <= 0:
                raise ValueError("horizon must be positive")
            if len(values) < w + horizon:
                continue
            reward = None
            if task == "policy_value_estimation":
                reward = _first_suffix(trace_states, "/reward")
                if reward is None:
                    raise ValueError("policy_value_estimation requires '/reward' in trace.states")
                reward = np.asarray(reward, dtype=float).reshape(-1)
            cf_key, cf_output = _first_counterfactual_output(trace_states) if task == "counterfactual_response" else (None, None)
            for start in range(0, len(values) - (w + horizon) + 1, st):
                end = start + w
                fut = end + horizon
                x_win = values[start:end]
                t_win = time[start:end]
                y_future = values[end:fut]
                if task in {"forecasting", "future_prediction"}:
                    _emit_sample(x_win, y_future, t_win, None, None, dict(m))
                elif task == "causal_response":
                    if values.ndim == 1:
                        raise ValueError("causal_response expects a multivariate series")
                    out_idx = outcome_channel
                    if out_idx is None:
                        out_idx = _first_suffix(trace_states, "/outcome_index")
                        out_idx = int(out_idx) if out_idx is not None else values.shape[1] - 1
                    _emit_sample(x_win, y_future[:, int(out_idx)], t_win, None, None, dict(m))
                elif task == "counterfactual_response":
                    if cf_output is not None:
                        out_idx = int(outcome_channel) if outcome_channel is not None else (cf_output.shape[1] - 1 if cf_output.ndim == 2 else 0)
                        y_cf = cf_output[end:fut, out_idx] if cf_output.ndim == 2 else cf_output[end:fut]
                        aux = {"counterfactual_key": np.asarray([0])} if include_aux else None
                        _emit_sample(x_win, y_cf, t_win, None, aux, dict(m))
                    else:
                        y0 = _first_suffix(trace_states, "/potential_outcome_do0")
                        y1 = _first_suffix(trace_states, "/potential_outcome_do1")
                        if y0 is None or y1 is None:
                            raise ValueError("counterfactual_response requires counterfactual outputs or potential outcomes in trace.states")
                        y = np.stack([np.asarray(y0)[end:fut], np.asarray(y1)[end:fut]], axis=-1)
                        _emit_sample(x_win, y, t_win, None, None, dict(m))
                elif task == "policy_value_estimation":
                    future_reward = reward[end:fut]
                    disc = np.power(float(gamma), np.arange(len(future_reward), dtype=float))
                    ret = float(np.sum(disc * future_reward))
                    _emit_sample(x_win, np.asarray(ret, dtype=float), t_win, None, None, dict(m))
                else:
                    raise ValueError(f"Sliding window not implemented for task={task!r}")

    X = _stack_or_object([np.asarray(x) for x in X_list])
    y = None
    if y_list:
        if task in {"classification", "contrastive"}:
            y = np.asarray(y_list, dtype=int)
        elif task == "policy_value_estimation":
            y = np.asarray(y_list, dtype=float)
        else:
            y = _stack_or_object([np.asarray(v) for v in y_list])

    masks = {k: _stack_or_object(v) for k, v in mask_map.items()} if mask_map else None
    aux = {k: _stack_or_object(v) for k, v in aux_map.items()} if aux_map else None
    time_arr = _stack_or_object(time_list) if time_list else None

    spec = TaskSpec(task=task, params={
        "horizon": int(horizon),
        "mask_ratio": float(mask_ratio),
        "order_segments": int(order_segments),
        "anomaly_rate": float(anomaly_rate),
        "window": None if window is None else int(window),
        "stride": int(stride),
        "outcome_channel": None if outcome_channel is None else int(outcome_channel),
        "gamma": float(gamma),
    })

    return TaskDataset(
        task=task,
        X=X,
        y=y,
        time=time_arr,
        masks=masks,
        aux=aux,
        meta=meta_out,
        label_names=label_names if task in {"classification", "contrastive"} else None,
        schema=spec.to_dict(),
    )
