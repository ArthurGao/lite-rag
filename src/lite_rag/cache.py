"""lite-cache's L1 for lite-rag's model calls (S18; lite-cache design §9). The only module that
imports `lite_cache` (an optional extra, `lite-rag[cache]`).

The runtime opens a lite-cache store on lite-rag's PostgreSQL and registers its `CacheHook` on
the one lite-model every role shares; each role's calls then run inside `cache_context`, which
tells the hook the scope (`literag/<namespace>`) and the policy, and turns a cache failure into
lite-rag's `StoreError` (a read failure sends nothing: no silent fallback, rule 5)."""

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lite_model import call_context

from lite_rag.config import Role, Settings
from lite_rag.errors import ConfigError, StoreError
from lite_rag.extras import require_extra
from lite_rag.prompts import keyed_nonces

MIN_SECRET = 32  # characters of the prompt secret
_LOG_LOCK = threading.Lock()  # one process's asks write the question log one line at a time
MIN_DISTINCT = 12  # distinct characters: `openssl rand -hex 32` has about 16, "aaaa…" has 1


@dataclass
class CacheBinding:
    """The cache store and the hook registered on the runtime's lite-model."""

    store: Any  # lite_cache.PgStore
    hook: Any  # lite_cache.CacheHook

    def close(self) -> None:
        self.store.close()


def prompt_key(settings: Settings) -> bytes:
    """The secret for the source-block nonces of cached calls (design §6), from the variable
    `cache.prompt_secret_env` names; ConfigError when it is unset or short."""
    env = settings.cache.prompt_secret_env
    secret = os.environ.get(env, "")
    # Length cannot prove a secret unpredictable; refusing the obviously repetitive ones catches
    # the common mistakes (Codex S18). Generate it: `openssl rand -hex 32`.
    if len(secret) < MIN_SECRET or len(set(secret)) < MIN_DISTINCT:
        raise ConfigError(
            f"cache needs a prompt secret of at least {MIN_SECRET} random characters "
            f"(e.g. `openssl rand -hex 32`): set {env}"
        )
    return secret.encode("utf-8")


def check_cache(settings: Settings) -> Any:
    """lite_cache, after checking what an enabled cache needs from this process (the extra and the
    prompt secret): at startup, like the reranker, not on the first request."""
    prompt_key(settings)
    return require_extra("lite_cache", "cache")


@contextmanager
def _as_store_error(lite_cache: Any) -> Iterator[None]:
    try:
        yield
    except lite_cache.LiteCacheError as exc:
        raise StoreError(f"lite-cache: {exc}") from exc


def open_cache(settings: Settings) -> CacheBinding:
    """A lite-cache store in `cache.schema` of lite-rag's database (migrated when behind)."""
    lite_cache = check_cache(settings)
    with _as_store_error(lite_cache):
        store = lite_cache.open_store(
            lite_cache.Settings(
                database_url=settings.database_url, schema_name=settings.cache.schema_name
            )
        )
    return CacheBinding(store, lite_cache.CacheHook(store))


def scope(namespace: str) -> str:
    """Every cached call of a namespace shares one scope; namespaces never share (lite-cache §0)."""
    return f"literag/{namespace}"


@contextmanager
def cache_context(
    settings: Settings, namespace: str, role: Role, question: str | None = None
) -> Iterator[None]:
    """Run `role`'s calls cached when the cache is on and lists the role; otherwise as before."""
    cfg = settings.cache
    if not cfg.enabled or role not in cfg.roles:
        yield
        return
    lite_cache = require_extra("lite_cache", "cache")
    values: dict[str, Any] = {"scope": scope(namespace), "cache": lite_cache.Policy(cfg.ttl_s)}
    if question is not None:
        values["question"] = question
    with (
        _as_store_error(lite_cache),
        call_context(**values),
        keyed_nonces(prompt_key(settings)),
    ):
        yield


def log_question(path: Path, namespace: str, question: str) -> None:
    """Append one JSON line {ts, scope, question} to `path` (directory created), for a later
    replay evaluation (lite-cache §8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "scope": scope(namespace),
        "question": question,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOG_LOCK, path.open("a", encoding="utf-8") as f:  # concurrent asks: whole lines only
        f.write(line)
