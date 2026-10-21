"""Tiny plug-in registry: add a method with @register("name"), nothing else to edit."""
from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Attribution method {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown attribution method {name!r}. Available: {available()}")
    return _REGISTRY[name](**kwargs)
