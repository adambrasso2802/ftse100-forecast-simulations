# FTSE 100 Forecast Simulation (L36 best vs L60 complexity trade-off)

Python project that simulates FTSE 100 price paths using historical data up to a cutoff date.
Includes an optional **test-period verification** (sometimes labelled “backtest”) by comparing the
**median forecast** to a realised FTSE 100 level on a target date.

> Note: This is **not** a trading-strategy backtest (no entry/exit rules, no transaction-cost P&L).
> “Backtest” here means: fit/calibrate on past data → forecast forward → compare to a later realised value.

## What’s inside
- **L36**: best-performing specification (lowest test error in my runs).
- **L60**: higher-complexity variant used to illustrate a complexity / generalisation trade-off.

## Repo structure
- `src/ftse100_forecast_l36.py`
- `src/ftse100_forecast_l60_tradeoff.py`
- `results/`

## How to run

### 1) Install dependencies

```bash
pip install -r requirements.txt
