from __future__ import annotations

from typing import Any

import numpy as np

_TYPE_REGISTRY: dict[str, type] = {}


def type_key(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def register_type(cls: type) -> type:
    _TYPE_REGISTRY[type_key(cls)] = cls
    return cls


def serialize_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, tuple):
        return {"__tuple__": [serialize_value(v) for v in value]}
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def deserialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "type" in value and "params" in value and value["type"] in _TYPE_REGISTRY:
            return object_from_dict(value)
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=float)
        if "__tuple__" in value:
            return tuple(deserialize_value(v) for v in value["__tuple__"])
        return {k: deserialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deserialize_value(v) for v in value]
    return value


def object_from_dict(payload: dict[str, Any]) -> Any:
    cls = _TYPE_REGISTRY[payload["type"]]
    kwargs = {key: deserialize_value(value) for key, value in payload.get("params", {}).items()}
    return cls(**kwargs)
