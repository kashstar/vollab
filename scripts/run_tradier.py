"""Fetch a live SPY option chain from Tradier and print it.

Requires a .env with VOLLAB_TRADIER_TOKEN set (see .env.example).
Run with: python scripts/run_tradier.py
"""

from vollab.ingestion import Settings, TradierClient

client = TradierClient(Settings())

expirations = client.get_expirations("SPY")
print(f"Found {len(expirations)} expirations. Next few: {expirations[:5]}")

nearest = expirations[0]
quotes = client.get_chain("SPY", nearest)
print(f"\n{len(quotes)} contracts for expiration {nearest}:\n")
for quote in quotes[:10]:
    print(quote)

client.close()
