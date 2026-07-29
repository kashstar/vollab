"""Check ArbitrageChecker against real, live Deribit BTC SVI slices.

Run with: python scripts/check_arbitrage_checker.py

Calibrates an SVI slice for every expiry ForwardEstimator accepts, then
checks the whole set for butterfly (within one expiry) and calendar
(across expiries) no-arbitrage violations. Given the known SVICalibrator
limitation (sigma occasionally landing near its lower bound, producing a
kinked curve), finding real butterfly violations here isn't surprising --
it's the checker doing its job.
"""

from vollab.ingestion import DeribitClient
from vollab.surface import ArbitrageChecker, ForwardEstimator, ImpliedVolSolver, SVICalibrator

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
calibrator = SVICalibrator()
checker = ArbitrageChecker()

slices = []
for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)

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

    slices.append(slice_)
    print(
        f"{expiry}: calibrated (a={slice_.a:.5f} b={slice_.b:.5f} "
        f"rho={slice_.rho:+.4f} m={slice_.m:+.4f} sigma={slice_.sigma:.5f})"
    )

print(f"\n{len(slices)} slices calibrated. Checking for arbitrage violations...\n")

violations = checker.check(slices)

if not violations:
    print("No violations found.")
else:
    butterfly = [v for v in violations if v.kind == "butterfly"]
    calendar = [v for v in violations if v.kind == "calendar"]
    print(
        f"{len(violations)} violations found: "
        f"{len(butterfly)} butterfly, {len(calendar)} calendar\n"
    )

    for v in violations[:15]:
        print(f"  [{v.kind}] {v.expiry} {v.detail}")
    if len(violations) > 15:
        print(f"  ... and {len(violations) - 15} more")

client.close()
