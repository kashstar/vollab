# VolLab

Crypto options don't behave the way Black-Scholes says they should. Pull up
any BTC or ETH chain on Deribit and the volatility smile is far more
extreme than anything you'd see in equities. Deep out-of-the-money puts get
priced tens of vol points above at-the-money, skew flips around macro
events, and the term structure never really settles down because these
things trade 24/7 with no weekend for realized vol to cool off.

Black-Scholes assumes one constant volatility per underlying. The market
clearly disagrees: every strike and expiry gets priced with its own implied
vol. This project is my attempt to model that properly. Fit the actual
surface, price options consistently across it, and then get to the more
interesting question: once you have a realistic model of how the surface
moves, can a neural network hedge an option book better than classic delta
hedging, once real trading costs are in the picture?

## The plan

**Ingestion.** Pull live option chains from Deribit. Already built, see
below.

**Surface.** Fit each expiry's smile with SVI (Gatheral's parameterization,
five numbers that describe an entire smile shape), then stitch expiries
together into one arbitrage-free surface with SSVI. Deribit quotes option
premiums in BTC or ETH, not USD, which is its own headache. Everything has
to be converted through the index price before any of this math is usable.

**Pricing.** SVI gives you a smile, but it's just curve-fitting. It can't
price something that doesn't already trade, like a barrier option. That
needs an actual stochastic process. Heston is the standard choice:
volatility itself follows a random, mean-reverting process correlated with
the spot. The catch is Heston has no closed form except through a
characteristic-function integral, and it's easy to implement subtly wrong
in a way that still spits out plausible-looking numbers. So every price
gets computed two independent ways, the Fourier/COS method and Monte
Carlo, and they have to agree within a few standard errors. If they don't,
something's broken.

**Hedging.** The actual point of all this. Once you can simulate realistic
surface dynamics (or replay historical ones), does a neural network trained
to minimize hedging risk, "deep hedging" per Buehler et al. (2019), actually
beat plain delta or delta-gamma hedging once transaction costs are in the
picture? Textbook theory says perfect hedging is free and continuous. Real
markets charge you every time you trade.

## What's actually built

Right now: ingestion, and nothing past it. `DeribitClient` pulls live
option chains straight from Deribit's public market data API. No account,
no API key needed, it's just open. Every quote gets validated into a strict
`OptionQuote` model (strikes have to be positive, bids can't be negative,
expiries have to be in the future), and prices get converted from
crypto-denominated to USD using each contract's index price at snapshot
time, so nothing downstream has to think about "0.012 BTC" as an option
premium.

The surface fitting, the two pricers, and the hedging backtest are all
still ahead. `SPEC.md` has the full design if you want the details.

## Running it

```bash
uv pip install -e ".[dev]"
python scripts/run_deribit.py
```

No credentials needed, it only touches Deribit's public endpoints. It'll
print a real, live BTC option chain.

For code quality: `ruff check .` and `mypy src/ scripts/`. There's no
automated test suite by design (see `CLAUDE.md` for why). The short version
is that for a project like this, actually running the thing and reading
real output catches more than a green checkmark does.
