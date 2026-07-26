"""Fetch a live BTC option chain from Deribit and print it.

No account or API key needed — Deribit's market data is public.
Run with: python scripts/run_deribit.py
"""

from vollab.ingestion import DeribitClient

client = DeribitClient()

expirations = client.get_expirations("BTC")
print(f"Found {len(expirations)} expirations. Next few: {expirations[:5]}")

nearest = expirations[0]
quotes = client.get_chain("BTC", nearest)
print(f"\n{len(quotes)} contracts for expiration {nearest}:\n")
for quote in quotes[:10]:
    print(quote)

client.close()
