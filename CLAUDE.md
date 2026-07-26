# VolLab — options research, pricing & deep-hedging engine

## What this is
Ingests live Deribit BTC/ETH options chains, calibrates vol surfaces (SVI/SSVI),
prices under Heston (COS + Monte Carlo), and benchmarks neural hedging vs Greeks.

## Stack & commands
- Python 3.11+, uv for env/deps. Install: `uv pip install -e ".[dev]"`
- Test: `pytest` · Lint: `ruff check .` · Types: `mypy src/`
- Layout: src/vollab/{ingestion,surface,pricing,hedging}/, tests/, notebooks/

## Non-negotiable rules
- Every module gets tests. Pricing code gets cross-check tests
  (two independent methods must agree) and property tests (round-trip identities).
- Never silently drop data: rejected quotes go to quarantine with a reason code.
- All timestamps UTC. All ingestion writes keyed on snapshot_ts (idempotent).
- Type hints everywhere. No bare dicts crossing module boundaries — pydantic models.
- Small single-purpose commits, imperative messages.
- Deribit option prices are quoted in crypto terms — convert to USD via index price.
