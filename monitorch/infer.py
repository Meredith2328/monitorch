from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping, Tuple

import torch


def infer_call_arguments(
    fn: Callable[..., Any],
    *,
    overrides: Mapping[str, Any],
    hints: Mapping[str, Any],
    local_scope: Mapping[str, Any],
    global_scope: Mapping[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    signature = inspect.signature(fn)
    positional = []
    keyword: Dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        value = _select_value(
            name=name,
            parameter=parameter,
            overrides=overrides,
            hints=hints,
            local_scope=local_scope,
            global_scope=global_scope,
        )

        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keyword[name] = value

    return tuple(positional), keyword


def _select_value(
    *,
    name: str,
    parameter: inspect.Parameter,
    overrides: Mapping[str, Any],
    hints: Mapping[str, Any],
    local_scope: Mapping[str, Any],
    global_scope: Mapping[str, Any],
) -> Any:
    if name in overrides:
        return overrides[name]
    if name in hints:
        return hints[name]
    if name in local_scope:
        return local_scope[name]
    if name in global_scope:
        return global_scope[name]
    if parameter.default is not inspect.Parameter.empty:
        return parameter.default
    return _infer_value(name=name, annotation=parameter.annotation)


def _infer_value(*, name: str, annotation: Any) -> Any:
    if annotation is int:
        return 4
    if annotation is float:
        return 0.1
    if annotation is bool:
        return False
    if annotation is str:
        return name

    lower = name.lower()
    if "shape" in lower or "size" in lower:
        return (4, 4)
    if "dim" in lower or lower.endswith("n"):
        return 4
    if "lr" in lower or "rate" in lower:
        return 0.01
    if "label" in lower or "target" in lower:
        return torch.randint(0, 2, (8,))
    return torch.randn(8, 8)
