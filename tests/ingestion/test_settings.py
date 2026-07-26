import pytest
from pydantic import ValidationError

from vollab.ingestion.tradier_client import Settings

_ENV_VARS = ["VOLLAB_TRADIER_TOKEN", "VOLLAB_TRADIER_SANDBOX"]


@pytest.fixture(autouse=True)
def _clear_tradier_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_missing_token_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLLAB_TRADIER_TOKEN", "secret-token")
    settings = Settings(_env_file=None)
    assert settings.token.get_secret_value() == "secret-token"


def test_sandbox_toggles_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLLAB_TRADIER_TOKEN", "secret-token")
    prod = Settings(_env_file=None)
    assert prod.base_url == "https://api.tradier.com/v1/"

    monkeypatch.setenv("VOLLAB_TRADIER_SANDBOX", "true")
    sandbox = Settings(_env_file=None)
    assert sandbox.base_url == "https://sandbox.tradier.com/v1/"


def test_sandbox_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLLAB_TRADIER_TOKEN", "secret-token")
    settings = Settings(_env_file=None)
    assert settings.sandbox is False
