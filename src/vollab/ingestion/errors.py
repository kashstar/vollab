class TradierError(Exception):
    """Raised for non-retryable Tradier API failures, or once retries are
    exhausted for a transient failure.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.endpoint = endpoint
