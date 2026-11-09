import pytest

from lite_rag.config import ApiKeyConfig
from lite_rag.server.auth import Authorizer, Denied, key_hash, new_key


def _authorizer(**overrides: object) -> tuple[Authorizer, str]:
    key, digest = new_key()
    cfg = {
        "name": "steward",
        "sha256": digest,
        "namespaces": ["steward/*"],
        "kbs": ["c-*"],
        "mode": "write",
        **overrides,
    }
    return Authorizer([ApiKeyConfig(**cfg)]), key  # type: ignore[arg-type]


def test_new_keys_are_unique_and_only_their_hash_is_kept() -> None:
    key, digest = new_key()
    assert key.startswith("lrk_") and digest == key_hash(key) and key not in digest
    assert new_key()[0] != key


@pytest.mark.parametrize("header", [None, "", "Bearer", "Basic abc", "Bearer lrk_x_unknown"])
def test_no_valid_key_is_401(header: str | None) -> None:
    auth, _ = _authorizer()
    with pytest.raises(Denied) as denied:
        auth.check(header, "steward/1", ["c-1"], write=False)
    assert denied.value.status == 401


@pytest.mark.parametrize(
    ("namespace", "kb", "write", "why"),
    [
        ("tenderted/1", "c-1", False, "namespace"),  # outside the namespace globs
        ("steward", "c-1", False, "namespace"),  # "steward/*" needs a level below
        ("steward/1", "faq", False, "knowledge base"),  # outside the kb globs
    ],
)
def test_a_valid_key_outside_its_scope_is_403(
    namespace: str, kb: str, write: bool, why: str
) -> None:
    auth, key = _authorizer()
    with pytest.raises(Denied, match=why) as denied:
        auth.check(f"Bearer {key}", namespace, [kb], write)
    assert denied.value.status == 403


def test_a_read_key_cannot_write_but_can_read() -> None:
    auth, key = _authorizer(mode="read")
    auth.check(f"Bearer {key}", "steward/1", ["c-1"], write=False)  # allowed: no exception
    with pytest.raises(Denied, match="read-only") as denied:
        auth.check(f"bearer {key}", "steward/1", ["c-1"], write=True)
    assert denied.value.status == 403


def test_a_namespace_level_request_checks_the_namespace_only() -> None:
    auth, key = _authorizer()
    auth.check(f"Bearer {key}", "steward/7/x", [], write=False)  # no kb to check


def test_without_keys_the_server_is_open() -> None:
    assert Authorizer([]).open
    Authorizer([]).check(None, "any/ns", "kb", write=True)  # no key needed


def test_a_key_needs_at_least_one_namespace() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ApiKeyConfig(name="x", sha256="0" * 64, namespaces=[])


def test_a_valid_key_under_another_scheme_is_401() -> None:
    auth, key = _authorizer()
    with pytest.raises(Denied) as denied:
        auth.check(f"Basic {key}", "steward/1", ["c-1"], write=False)
    assert denied.value.status == 401


def test_logged_errors_have_their_credentials_removed() -> None:
    from lite_rag.errors import redact

    msg = "connection to postgresql://lite_rag:s3cret@db:5432/x failed; password=hunter2 host=db"
    out = redact(msg)
    assert "s3cret" not in out and "hunter2" not in out
    assert "postgresql://***@db:5432/x" in out and "password=***" in out
