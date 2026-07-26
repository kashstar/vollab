from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-derived configuration for Tradier ingestion.

    Reads VOLLAB_TRADIER_* environment variables (and an optional .env
    file). Instantiating Settings() with no VOLLAB_TRADIER_TOKEN set raises
    a pydantic ValidationError immediately.
    """

    model_config = SettingsConfigDict(
        env_prefix="VOLLAB_TRADIER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    token: SecretStr
    sandbox: bool = False
    timeout_seconds: float = 30.0
    min_request_interval_seconds: float = 0.5
    max_retries: int = 3

    @property
    def base_url(self) -> str:
        """Tradier API base URL. Trailing slash required for httpx path joins."""
        host = "sandbox.tradier.com" if self.sandbox else "api.tradier.com"
        return f"https://{host}/v1/"
