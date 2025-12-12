## Results (test-period verification)

| Model | Sims | Days ahead | Actual | Median forecast | Error |
|------:|-----:|----------:|------:|---------------:|------:|
| L36   | 20000 | 252 | 9642.01 | **9635.15** | **0.07%** |
| L60   | 20000 | 252 | 9642.01 | **9680.80** | **0.40%** |

**Error** = `abs(median_forecast - actual) / actual * 100`

### Outputs
- L36 saves: `level36_forecast.png`
- L60 saves: `level60_ensemble.png`
