from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .datasets.builder import TaskDataset
from .datasets.series_dataset import SeriesDataset


def _unwrap_maybe_object(value: Any) -> Any:
    arr = np.asarray(value, dtype=object)
    if arr.dtype == object:
        return [np.asarray(item) for item in arr.tolist()]
    return np.asarray(value)


def load_asset_file(path: str | Path) -> tuple[str, Any, Any | None]:
    """Load a time-series asset from disk.

    Returns `(kind, values, time)` where `kind` is one of:
    - ``series`` for a single sequence-like asset
    - ``dataset`` for a base ``SeriesDataset``-compatible asset
    - ``task_dataset`` for a saved ``TaskDataset`` archive
    - ``auto`` when the arrays should be inferred by downstream coercion
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        return 'auto', arr, None
    if suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        files = set(data.files)
        if {'values', 'time'}.issubset(files):
            return 'dataset', _unwrap_maybe_object(data['values']), _unwrap_maybe_object(data['time'])
        if 'X' in files:
            payload: dict[str, Any] = {'X': _unwrap_maybe_object(data['X'])}
            if 'y' in files:
                payload['y'] = _unwrap_maybe_object(data['y'])
            if 'time' in files:
                payload['time'] = _unwrap_maybe_object(data['time'])
            masks = {k.split('mask__', 1)[1]: _unwrap_maybe_object(data[k]) for k in files if k.startswith('mask__')}
            aux = {k.split('aux__', 1)[1]: _unwrap_maybe_object(data[k]) for k in files if k.startswith('aux__')}
            payload['masks'] = masks or None
            payload['aux'] = aux or None
            return 'task_dataset', payload, None
        raise ValueError(f'Unsupported .npz keys: {sorted(files)}')
    if suffix in {'.csv', '.txt'}:
        arr = np.genfromtxt(path, delimiter=',' if suffix == '.csv' else None)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            first = np.asarray(arr[:, 0], dtype=float)
            if np.all(np.diff(first) >= 0):
                return 'series', np.asarray(arr[:, 1:], dtype=float), first
        return 'auto', np.asarray(arr, dtype=float), None
    if suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and 'values' in payload:
            return 'auto', payload['values'], payload.get('time')
        if isinstance(payload, dict) and 'X' in payload:
            return 'task_dataset', payload, None
        raise ValueError('JSON input must contain at least a `values` field or task arrays.')
    raise ValueError(f'Unsupported input format: {suffix}')


def coerce_asset(source: Any, time: Any | None = None, *, dataset_id: str | None = None, channel_names: list[str] | None = None) -> SeriesDataset | TaskDataset:
    """Coerce a path, raw arrays, or an existing asset into a TSDataForge asset object."""

    if isinstance(source, SeriesDataset | TaskDataset):
        return source
    if isinstance(source, (str, Path)):
        kind, values, loaded_time = load_asset_file(source)
        dataset_id = dataset_id or Path(source).stem.replace(' ', '_')
        if kind == 'task_dataset':
            payload = dict(values)
            return TaskDataset(
                task=str(payload.get('task', 'unknown')),
                X=np.asarray(payload['X'], dtype=object),
                y=None if payload.get('y') is None else np.asarray(payload['y'], dtype=object),
                time=None if payload.get('time') is None else np.asarray(payload['time'], dtype=object),
                masks=payload.get('masks'),
                aux=payload.get('aux'),
                meta=payload.get('meta'),
                label_names=payload.get('label_names'),
                schema=payload.get('schema'),
            )
        return SeriesDataset.from_arrays(values, loaded_time if loaded_time is not None else time, dataset_id=dataset_id, channel_names=channel_names)
    return SeriesDataset.from_arrays(source, time, dataset_id=dataset_id or 'external', channel_names=channel_names)
