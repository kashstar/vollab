"""Check HestonCalibrator against a real, live Deribit BTC surface.

Run with: python scripts/check_heston_calibrator.py

Runs the full pipeline (ingestion -> forward -> implied vol -> SVI) across
every usable expiry, fits one set of Heston parameters to the whole
surface at once, then compares Heston's own implied vol back against
each SVI slice's implied vol at a few strikes -- since SVI fits each
expiry independently with 5 numbers each, while Heston fits the entire
surface at once with only 5 numbers total, some gap between them is
expected; the question is whether it's a reasonable approximation or a
bad one.
"""

from math import exp

from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionQuote, OptionType
from vollab.pricing import COSPricer, HestonCalibrator
from vollab.pricing.models import OptionContract
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SVICalibrator

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
svi_calibrator = SVICalibrator()
heston_calibrator = HestonCalibrator(COSPricer(num_terms=64))

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
        slice_ = svi_calibrator.calibrate(usable_quotes, vols, forward_estimate, time_to_expiry)
    except ValueError as exc:
        print(f"{expiry}: skipped SVI calibration ({exc})")
        continue

    forward_estimates.append(forward_estimate)
    slices.append(slice_)
    print(f"{expiry}: SVI fitted")

assert snapshot_ts is not None

print("\nCalibrating Heston to the whole surface at once...")
heston_params = heston_calibrator.calibrate(slices, forward_estimates, snapshot_ts)
print(
    f"kappa={heston_params.kappa:.4f}  theta={heston_params.theta:.4f}  "
    f"xi={heston_params.xi:.4f}  rho={heston_params.rho:+.4f}  v0={heston_params.v0:.4f}\n"
)

cos_pricer = COSPricer()
print(f"{'expiry':>12} {'moneyness':>10} {'SVI vol':>9} {'Heston vol':>11} {'diff':>8}")
for slice_, forward_estimate in zip(slices, forward_estimates, strict=True):
    time_to_expiry = (slice_.expiry - snapshot_ts.date()).days / 365.0
    if time_to_expiry <= 0:
        continue

    for k in [-0.1, 0.0, 0.1]:
        svi_vol = slice_.implied_vol(k, time_to_expiry)

        strike = forward_estimate.forward * exp(k)
        option_type = OptionType.CALL if k >= 0 else OptionType.PUT
        contract = OptionContract(
            strike=strike,
            option_type=option_type,
            forward=forward_estimate.forward,
            discount_factor=forward_estimate.discount_factor,
            time_to_expiry=time_to_expiry,
        )
        heston_price = cos_pricer.price(contract, heston_params).price

        # Back out Heston's own implied vol at this point by re-solving
        # from its COS price, reusing ImpliedVolSolver's Black-76
        # inversion the same way real market vols were extracted earlier.
        fake_quote = OptionQuote(
            source="heston-check",
            underlying="BTC",
            snapshot_ts=snapshot_ts,
            expiry=slice_.expiry,
            strike=strike,
            option_type=option_type,
            bid=heston_price,
            ask=heston_price,
            last=heston_price,
            volume=0.0,
            open_interest=0.0,
            underlying_price=forward_estimate.forward,
        )
        heston_vol = solver.solve(fake_quote, forward_estimate)

        print(
            f"{str(slice_.expiry):>12} {k:>+10.2f} {svi_vol:>9.1%} "
            f"{heston_vol:>11.1%} {heston_vol - svi_vol:>+8.1%}"
        )

client.close()
