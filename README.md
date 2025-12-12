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
- `results/RESULTS.md`
- `results/level36_forecast`
- `results/level60_ensemble`

## How to run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Test-period verification
Uses historical data up to `--end` as the training cutoff, simulates `--days` forward, then compares the **median forecast** to `--actual-price` and prints the percent error.
```bash
python src/ftse100_forecast_l36.py --end 2024-12-09 --days 252 --sims 20000 --actual-price 9642.01
python src/ftse100_forecast_l60_tradeoff.py --end 2024-12-09 --days 252 --sims 20000 --seed 42 --actual-price 9642.01
```

### 3) Forecast mode
These scripts default to test-period verification because `--actual-price` has a numeric default.
To run forecast mode (no error printed), set `--actual-price` default to `None` in the script, then run:

```bash
# Set --end to today's date (YYYY-MM-DD)

python src/ftse100_forecast_l36.py --end YYYY-MM-DD --days 252 --sims 20000
python src/ftse100_forecast_l60_tradeoff.py --end YYYY-MM-DD --days 252 --sims 25200 --seed 42
