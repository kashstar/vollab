# VolLab — options research, pricing & deep-hedging engine

## What this is
Ingests live Deribit BTC/ETH options chains, calibrates vol surfaces (SVI/SSVI),
prices under Heston (COS + Monte Carlo), and benchmarks neural hedging vs Greeks.

## Stack & commands
- Python 3.11+, uv for env/deps. Install: `uv pip install -e ".[dev]"`
- Lint: `ruff check .` · Types: `mypy src/ scripts/`
- No automated test suite. Verify by running the relevant script in
  `scripts/` (e.g. `python scripts/run_deribit.py`) and reading its output —
  Deribit's public market data API needs no account/API key, so this is
  always runnable.
- Layout: src/vollab/{ingestion,surface,pricing,hedging}/, scripts/, notebooks/

## Non-negotiable rules
- Every numerical/pricing result gets an independent check (two methods
  agreeing, a round-trip identity) verified by running a script and reading
  the output — not an automated test suite.
- Never silently drop data: rejected quotes go to quarantine with a reason code.
- All timestamps UTC. All ingestion writes keyed on snapshot_ts (idempotent).
- Type hints everywhere. No bare dicts crossing module boundaries — pydantic models.
- Small single-purpose commits, imperative messages.
- Deribit option prices are quoted in crypto terms — convert to USD via index price.
