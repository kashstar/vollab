"""Check SurfaceStore against real, live Deribit BTC data.

Run with: python scripts/check_surface_store.py

Runs the full Tier 2 pipeline (ingestion -> forward -> implied vol -> SVI
calibration) across every usable expiry, saves the resulting slices, then
saves the exact same snapshot a second time to prove save() is actually
idempotent -- the second call should write 0 rows, not duplicates.
"""

from pathlib import Path

from vollab.ingestion import DeribitClient
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SurfaceStore, SVICalibrator

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
calibrator = SVICalibrator()
store = SurfaceStore(Path("data/surface_params.csv"))

snapshot_ts = None
forward_estimates = []
slices = []

for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)
    if snapshot_ts is None:
        snapshot_ts = quotes[0].snapshot_ts

    try:
        forward_estimate = estimator.estimate(quotes, expiry, max_moneyness=0.15)
    except ValueError as exc:
        print(f"{expiry}: skipped forward estimate ({exc})")
        continue

    time_to_expiry = (expiry - quotes[0].snapshot_ts.date()).days / 365.0

    usable_quotes = []
    vols = []
    for quote in quotes:
        if quote.bid <= 0 or quote.ask <= 0:
            continue
        try:
            vol = solver.solve(quote, forward_estimate)
        except ValueError:
            continue
        usable_quotes.append(quote)
        vols.append(vol)

    try:
        slice_ = calibrator.calibrate(usable_quotes, vols, forward_estimate, time_to_expiry)
    except ValueError as exc:
        print(f"{expiry}: skipped SVI calibration ({exc})")
        continue

    forward_estimates.append(forward_estimate)
    slices.append(slice_)
    print(f"{expiry}: ready to store")

assert snapshot_ts is not None

print(f"\nSaving snapshot {snapshot_ts.isoformat()}...")
written = store.save(snapshot_ts, "BTC", forward_estimates, slices)
print(f"First save: {written} rows written")

written_again = store.save(snapshot_ts, "BTC", forward_estimates, slices)
print(f"Second save (same snapshot, same data): {written_again} rows written (should be 0)")

client.close()
