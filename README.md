# VolLab — options research, pricing & deep-hedging engine

## What this is
Ingests live Deribit BTC/ETH options chains, calibrates vol surfaces (SVI/SSVI),
prices under Heston (COS + Monte Carlo), and benchmarks neural hedging vs Greeks.

## Stack & commands
- Python 3.11+, uv for env/deps. Install: `uv pip install -e ".[dev]"`
- Test: `pytest` · Lint: `ruff check .` · Types: `mypy src/`
- Layout: src/vollab/{ingestion,surface,pricing,hedging}/, tests/, notebooks/
