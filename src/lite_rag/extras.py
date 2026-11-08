"""Optional dependencies and secrets — the checks every adapter needs, written once."""

import importlib
import os
from types import ModuleType

from lite_rag.errors import ConfigError


def require_extra(module: str, extra: str) -> ModuleType:
    """Import `module`, or say which `pip install 'lite-rag[extra]'` provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ConfigError(f"install the {extra} extra: pip install 'lite-rag[{extra}]'") from exc


def api_key(section: str, configured_env: str | None, default_env: str | None) -> str | None:
    """The key from `configured_env`, else `default_env`; an error if the variable it names is
    unset. None when neither names one (a provider that needs no key)."""
    env = configured_env or default_env
    if env is None:
        return None
    key = os.environ.get(env)
    if not key:
        raise ConfigError(f"{section} needs an API key: set {env}")
    return key
