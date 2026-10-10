# Data audit

Sample 2026-03-02 to 2026-10-09; 155 NYSE trading days. Market data source: **cache**.

## 1. Equity, VIX and Treasury data

### Alignment to the NYSE calendar

| index | missing_days | forward_filled |
|---|---|---|
| ICLN | 0 | 0 |
| ITA | 0 | 0 |
| SPY | 0 | 0 |
| XAR | 0 | 0 |
| XLE | 0 | 0 |
| XLF | 0 | 0 |
| XLK | 0 | 0 |
| XLV | 0 | 0 |
| ^VIX | 0 | 0 |
| DGS2 | 1 | 1 |
| DGS3MO | 1 | 1 |

### OHLC consistency violations

_None._

### Return outliers (> 5 robust SD)

Each is checked against the second vendor below before being trusted.

| index | series | date | return | robust_z |
|---|---|---|---|---|
| 0 | XLK_ex | .4f | -0.0428 | -5.1223 |

### Consecutive unchanged closes

| index | series | unchanged_days | dates |
|---|---|---|---|
| 0 | XAR | 1 | 2026-09-25 |
| 1 | XLF | 1 | 2026-04-07 |
| 2 | XLK | 2 | 2026-03-10, 2026-09-09 |
| 3 | XLV | 1 | 2026-07-15 |

### Yahoo (adjusted) vs Stooq (unadjusted) daily returns

Differences on ex-dividend days are expected; anything else is a feed error.

| index | error |
|---|---|
| 0 | ConnectionError(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))) |

### FRED yields

| series | obs | min | max | days_jump_gt_25bp | largest_jump_bp |
|---|---|---|---|---|---|
| DGS2 | 155 | 3.4700 | 4.9200 | 0 | 15.0000 |
| DGS3MO | 155 | 3.6500 | 4.2800 | 0 | 9.0000 |

## 2. Prediction markets

### Coverage (trading days with a quote at or before 16:00 ET)

| venue | House | Senate |
|---|---|---|
| kalshi | 155 | 155 |
| polymarket | 155 | 155 |
| predictit | 64 | 64 |

### Quote staleness at the 16:00 ET close

| venue | chamber | days | median_age_h | p95_age_h | max_age_h | days_older_than_24h |
|---|---|---|---|---|---|---|
| kalshi | House | 155 | 0.0003 | 0.0003 | 0.0003 | 0 |
| kalshi | Senate | 155 | 0.0003 | 0.0003 | 0.0003 | 0 |
| polymarket | House | 155 | 0.9953 | 0.9986 | 3.9844 | 0 |
| polymarket | Senate | 155 | 0.9953 | 0.9989 | 6.9972 | 0 |
| predictit | House | 64 | 16.0167 | 16.0167 | 16.0167 | 0 |
| predictit | Senate | 64 | 16.0167 | 16.0167 | 16.0167 | 0 |

### Polymarket vs Kalshi agreement

| chamber | days | level_corr | change_corr | mean_abs_gap_pp | max_gap_pp | days_gap_gt_5pp |
|---|---|---|---|---|---|---|
| House | 155 | 0.9350 | 0.4356 | 2.1460 | 6.0000 | 6 |
| Senate | 155 | 0.9598 | 0.3805 | 1.8893 | 6.0000 | 6 |

### Days the two venues differ by more than 5pp

| index | chamber | date | polymarket | kalshi |
|---|---|---|---|---|
| 0 | House | .4f | 0.7850 | 0.7350 |
| 1 | House | .4f | 0.7850 | 0.7350 |
| 2 | House | .4f | 0.7950 | 0.7450 |
| 3 | House | .4f | 0.8050 | 0.7550 |
| 4 | House | .4f | 0.8250 | 0.7650 |
| 5 | House | .4f | 0.8232 | 0.7650 |
| 6 | Senate | .4f | 0.5350 | 0.4850 |
| 7 | Senate | .4f | 0.5000 | 0.4461 |
| 8 | Senate | .4f | 0.4950 | 0.4350 |
| 9 | Senate | .4f | 0.4850 | 0.4337 |
| 10 | Senate | .4f | 0.4650 | 0.4150 |
| 11 | Senate | .4f | 0.4650 | 0.4150 |

### Polymarket vs PredictIt agreement (PredictIt is daily, cross-check only)

| chamber | days | level_corr | change_corr | mean_abs_gap_pp | max_gap_pp | days_gap_gt_5pp |
|---|---|---|---|---|---|---|
| House | 64 | 0.9050 | 0.0926 | 6.6239 | 10.9902 | 47 |
| Senate | 64 | 0.9777 | 0.4519 | 3.4573 | 8.5000 | 10 |

### Consensus moves larger than 4pp, by venue

| index | chamber | date | dProb_pp | d_kalshi_pp | d_polymarket_pp |
|---|---|---|---|---|---|
| 0 | House | .4f | -5.1931 | -4.2129 | -6.1733 |
| 1 | Senate | .4f | 4.0000 | 2.0000 | 6.0000 |
| 2 | Senate | .4f | -4.0931 | -5.1863 | -3.0000 |

### Consensus P(Dem) summary

| index | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| House | 155.0000 | 0.8405 | 0.0385 | 0.7550 | 0.8200 | 0.8450 | 0.8607 | 0.9222 |
| Senate | 155.0000 | 0.4886 | 0.0579 | 0.4101 | 0.4475 | 0.4774 | 0.5038 | 0.6500 |

### Errors on this run

_None._

