# VolLab — options research, pricing & deep-hedging engine

## What this is
Ingests live Deribit BTC/ETH options chains, calibrates vol surfaces (SVI/SSVI),
prices under Heston (COS + Monte Carlo), and benchmarks neural hedging vs Greeks.

## Stack & commands
- Python 3.11+, uv for env/deps. Install: `uv pip install -e ".[dev]"`
- Lint: `ruff check .` · Types: `mypy src/ scripts/`
- Check things work by running the real scripts and reading the output —
  e.g. `python scripts/run_deribit.py`. Deribit's market data API is public
  (no account/API key needed), so this always runs. No automated test
  suite; this project is learn-by-running.
- Layout: src/vollab/{ingestion,surface,pricing,hedging}/, scripts/, notebooks/
