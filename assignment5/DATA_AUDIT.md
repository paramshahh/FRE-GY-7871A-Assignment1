# Data audit

Sample 2026-03-02 to 2026-10-09; 155 NYSE trading days. Market data source: **cache**.

## 1. Equity, VIX and Treasury data

### Alignment to the NYSE calendar

| index | missing_days | forward_filled |
|---|---|---|
| AAPL | 0 | 0 |
| ENPH | 0 | 0 |
| GOOGL | 0 | 0 |
| ICLN | 0 | 0 |
| ITA | 0 | 0 |
| ITB | 0 | 0 |
| JPM | 0 | 0 |
| LMT | 0 | 0 |
| META | 0 | 0 |
| MSFT | 0 | 0 |
| NVDA | 0 | 0 |
| SPY | 0 | 0 |
| UNH | 0 | 0 |
| XAR | 0 | 0 |
| XHB | 0 | 0 |
| XLE | 0 | 0 |
| XLF | 0 | 0 |
| XLK | 0 | 0 |
| XLV | 0 | 0 |
| XOM | 0 | 0 |
| ^VIX | 0 | 0 |
| DGS2 | 1 | 1 |
| DGS3MO | 1 | 1 |

### OHLC consistency violations

_None._

### Return outliers (> 5 robust SD)

Each is checked against the second vendor below before being trusted.

| index | series | date | return | robust_z |
|---|---|---|---|---|
| 0 | UNH | 2026-04-07 | 0.0896 | 5.5724 |
| 1 | LMT | 2026-07-23 | 0.1002 | 6.1422 |
| 2 | MSFT | 2026-07-30 | 0.1442 | 8.0052 |
| 3 | GOOGL | 2026-04-30 | 0.0950 | 5.8420 |
| 4 | META | 2026-09-21 | 0.1082 | 5.8870 |
| 5 | AAPL | 2026-07-31 | -0.0764 | -5.7618 |
| 6 | XLK_ex | 2026-06-05 | -0.0428 | -5.1223 |
| 7 | UNH_ex | 2026-04-07 | 0.0891 | 5.8910 |
| 8 | LMT_ex | 2026-07-23 | 0.1127 | 6.6615 |
| 9 | MSFT_ex | 2026-07-30 | 0.1275 | 9.3814 |
| 10 | GOOGL_ex | 2026-04-30 | 0.0851 | 7.0939 |
| 11 | GOOGL_ex | 2026-07-23 | -0.0616 | -5.1402 |
| 12 | META_ex | 2026-04-30 | -0.0993 | -5.8961 |
| 13 | META_ex | 2026-07-01 | 0.0858 | 5.1269 |
| 14 | META_ex | 2026-07-30 | -0.0995 | -5.9098 |
| 15 | META_ex | 2026-09-21 | 0.0928 | 5.5469 |
| 16 | AAPL_ex | 2026-06-25 | -0.0646 | -5.6804 |
| 17 | AAPL_ex | 2026-07-31 | -0.0836 | -7.3363 |

### Consecutive unchanged closes

| index | series | unchanged_days | dates |
|---|---|---|---|
| 0 | JPM | 1 | 2026-10-05 |
| 1 | XAR | 1 | 2026-09-25 |
| 2 | XLF | 1 | 2026-04-07 |
| 3 | XLK | 2 | 2026-03-10, 2026-09-09 |
| 4 | XLV | 1 | 2026-07-15 |

### Same-exposure funds from other issuers (second-source check)

Stooq blocked repeated downloads, so each fund is triangulated against funds with the same exposure from other issuers instead.

| index | fund | sibling | days | corr | median_abs_diff_bp | days_over_6sd |
|---|---|---|---|---|---|---|
| 0 | SPY | IVV | 154 | 0.9989 | 1.4779 | 5 |
| 1 | SPY | VOO | 154 | 0.9994 | 1.2575 | 2 |
| 2 | XLE | VDE | 154 | 0.9964 | 8.7803 | 0 |
| 3 | XLV | VHT | 154 | 0.9882 | 11.1639 | 0 |
| 4 | XLF | VFH | 154 | 0.9944 | 5.8136 | 0 |
| 5 | XLK | VGT | 154 | 0.9927 | 13.6761 | 0 |
| 6 | XAR | ITA | 154 | 0.9257 | 52.4824 | 0 |
| 7 | ITB | XHB | 154 | 0.9661 | 33.9982 | 0 |

### Days a fund diverges from its sibling by > 6 robust SD

| index | fund | sibling | date | fund_bp | sibling_bp | z |
|---|---|---|---|---|---|---|
| 0 | SPY | IVV | 2026-06-25 | 14.4459 | -2.1715 | 7.5894 |
| 1 | SPY | IVV | 2026-06-26 | -72.5760 | -86.3186 | 6.2774 |
| 2 | SPY | IVV | 2026-06-29 | 163.4058 | 190.1902 | -12.2186 |
| 3 | SPY | IVV | 2026-06-30 | 77.5665 | 62.9574 | 6.6728 |
| 4 | SPY | IVV | 2026-07-01 | -13.5342 | 2.8032 | -7.4507 |
| 5 | SPY | VOO | 2026-06-25 | 14.4459 | 0.2961 | 7.6881 |
| 6 | SPY | VOO | 2026-06-26 | -72.5760 | -51.9047 | -11.3404 |

### SPY vs the S&P 500 index (FRED SP500)

25 Jun - 1 Jul 2026: the index (FRED and Yahoo ^GSPC, one upstream source) and the three S&P 500 trackers SPY, IVV, VOO disagree by 15-30bp a day, while agreeing on the week's total (+155 to +170bp). Daily closes that week are mistimed in at least one feed. Sensitivity: replacing SPY with the median of SPY/IVV/VOO (mean change 0.9bp) or dropping 24 Jun - 2 Jul leaves every Stage 2 result's sign and significance unchanged (see the report).

| index | fund | sibling | days | corr | median_abs_diff_bp | days_over_6sd |
|---|---|---|---|---|---|---|
| 0 | SPY | SP500 | 149 | 0.9958 | 2.7901 | 2 |

### SPY vs index divergence days

| index | fund | sibling | date | fund_bp | sibling_bp | z |
|---|---|---|---|---|---|---|
| 0 | SPY | SP500 | 2026-06-26 | -72.5760 | -4.7174 | -16.2340 |
| 1 | SPY | SP500 | 2026-06-29 | 163.4058 | 116.8154 | 10.9529 |

### FRED yields

| series | obs | min | max | days_jump_gt_25bp | largest_jump_bp |
|---|---|---|---|---|---|
| DGS2 | 155 | 3.4700 | 4.9200 | 0 | 15.0000 |
| DGS3MO | 155 | 3.6500 | 4.2800 | 0 | 9.0000 |

## 2. Prediction markets

### Coverage (trading days with a quote at or before 16:00 ET)

| venue | House | Senate | Sweep |
|---|---|---|---|
| kalshi | 155 | 155 | 155 |
| polymarket | 155 | 155 | 155 |
| predictit | 64 | 64 | 0 |

### Quote staleness at the 16:00 ET close

| venue | chamber | days | median_age_h | p95_age_h | max_age_h | days_older_than_24h |
|---|---|---|---|---|---|---|
| kalshi | House | 155 | 0.0003 | 0.0003 | 0.0003 | 0 |
| kalshi | Senate | 155 | 0.0003 | 0.0003 | 0.0003 | 0 |
| kalshi | Sweep | 155 | 0.0003 | 0.0003 | 0.0003 | 0 |
| polymarket | House | 155 | 0.9953 | 0.9986 | 3.9844 | 0 |
| polymarket | Senate | 155 | 0.9953 | 0.9989 | 6.9972 | 0 |
| polymarket | Sweep | 155 | 0.9950 | 0.9986 | 1.9850 | 0 |
| predictit | House | 64 | 16.0167 | 16.0167 | 16.0167 | 0 |
| predictit | Senate | 64 | 16.0167 | 16.0167 | 16.0167 | 0 |

### Polymarket vs Kalshi agreement

| chamber | days | level_corr | change_corr | mean_abs_gap_pp | max_gap_pp | days_gap_gt_5pp |
|---|---|---|---|---|---|---|
| House | 155 | 0.9350 | 0.4356 | 2.1460 | 6.0000 | 6 |
| Senate | 155 | 0.9598 | 0.3805 | 1.8893 | 6.0000 | 6 |
| Sweep | 155 | 0.9589 | 0.3944 | 2.1205 | 6.0599 | 7 |

### Days the two venues differ by more than 5pp

| index | chamber | date | polymarket | kalshi |
|---|---|---|---|---|
| 0 | House | 2026-05-14 | 0.7850 | 0.7350 |
| 1 | House | 2026-05-15 | 0.7850 | 0.7350 |
| 2 | House | 2026-05-18 | 0.7950 | 0.7450 |
| 3 | House | 2026-05-22 | 0.8050 | 0.7550 |
| 4 | House | 2026-06-16 | 0.8250 | 0.7650 |
| 5 | House | 2026-06-17 | 0.8232 | 0.7650 |
| 6 | Senate | 2026-04-10 | 0.5350 | 0.4850 |
| 7 | Senate | 2026-05-05 | 0.5000 | 0.4461 |
| 8 | Senate | 2026-05-06 | 0.4950 | 0.4350 |
| 9 | Senate | 2026-05-07 | 0.4850 | 0.4337 |
| 10 | Senate | 2026-06-02 | 0.4650 | 0.4150 |
| 11 | Senate | 2026-06-03 | 0.4650 | 0.4150 |
| 12 | Sweep | 2026-08-27 | 0.5172 | 0.4570 |
| 13 | Sweep | 2026-09-01 | 0.5122 | 0.4568 |
| 14 | Sweep | 2026-09-02 | 0.5172 | 0.4566 |
| 15 | Sweep | 2026-09-03 | 0.5172 | 0.4574 |
| 16 | Sweep | 2026-09-04 | 0.5116 | 0.4616 |
| 17 | Sweep | 2026-09-08 | 0.5213 | 0.4694 |
| 18 | Sweep | 2026-09-15 | 0.5611 | 0.5053 |

### Polymarket vs PredictIt agreement (PredictIt is daily, cross-check only)

| chamber | days | level_corr | change_corr | mean_abs_gap_pp | max_gap_pp | days_gap_gt_5pp |
|---|---|---|---|---|---|---|
| House | 64 | 0.9050 | 0.0926 | 6.6239 | 10.9902 | 47 |
| Senate | 64 | 0.9777 | 0.4519 | 3.4573 | 8.5000 | 10 |

### Consensus moves larger than 4pp, by venue

| index | chamber | date | dProb_pp | d_kalshi_pp | d_polymarket_pp |
|---|---|---|---|---|---|
| 0 | House | 2026-05-08 | -5.1931 | -4.2129 | -6.1733 |
| 1 | Senate | 2026-09-15 | 4.0000 | 2.0000 | 6.0000 |
| 2 | Senate | 2026-09-24 | -4.0931 | -5.1863 | -3.0000 |

### Consensus P(Dem) summary

| index | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| House | 155.0000 | 0.8405 | 0.0385 | 0.7550 | 0.8200 | 0.8450 | 0.8607 | 0.9222 |
| Senate | 155.0000 | 0.4886 | 0.0579 | 0.4101 | 0.4475 | 0.4774 | 0.5038 | 0.6500 |
| Sweep | 155.0000 | 0.4737 | 0.0609 | 0.3933 | 0.4305 | 0.4606 | 0.4899 | 0.6416 |

## 3. Text

### Tweets by stratum (twitterapi.io)

Paid calls in the ledger: 318 for 93,450 credits.

| stratum | tweets | sessions_with_tweets | sessions | min_per_session | median_per_session | authors | top10_author_share | followers_p25 | followers_median | followers_p75 | share_under_10k_followers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| affordability | 2,782 | 155 | 155 | 9 | 18.0000 | 1,725 | 0.0791 | 6104.0000 | 24132.0000 | 75665.0000 | 0.3379 |
| general | 3,100 | 155 | 155 | 20 | 20.0000 | 1,740 | 0.0600 | 13764.0000 | 42405.0000 | 142341.0000 | 0.1874 |

### Tweet quality-filter waterfall

| index | step | remaining | removed |
|---|---|---|---|
| 0 | raw | 5,882 | 0 |
| 1 | duplicate ids | 5,882 | 0 |
| 2 | non-English | 5,882 | 0 |
| 3 | retweets | 5,882 | 0 |
| 4 | followers < 10 | 5,882 | 0 |
| 5 | views < 100 | 5,816 | 66 |
| 6 | astroturf (same text, >= 5 accounts) | 5,816 | 0 |
| 7 | fewer than 4 tokens | 5,816 | 0 |

### GDELT DOC timelines

| index | calendar_days | days_returned | outage_days | median_norm | queries_cached |
|---|---|---|---|---|---|
| 0.0000 | 222.0000 | 206.0000 | 19.0000 | 146819.5000 | 18.0000 |

### GDELT outage days (masked, never read as quiet news days)

| index | date | articles_monitored |
|---|---|---|
| 0 | 2026-07-01 | – |
| 1 | 2026-07-02 | – |
| 2 | 2026-07-03 | – |
| 3 | 2026-07-04 | – |
| 4 | 2026-09-13 | 29203.0000 |
| 5 | 2026-09-14 | – |
| 6 | 2026-09-15 | – |
| 7 | 2026-09-16 | – |
| 8 | 2026-09-17 | – |
| 9 | 2026-09-18 | – |
| 10 | 2026-09-19 | – |
| 11 | 2026-09-26 | – |
| 12 | 2026-10-03 | – |
| 13 | 2026-10-04 | – |
| 14 | 2026-10-05 | – |
| 15 | 2026-10-06 | 3958.0000 |
| 16 | 2026-10-07 | – |
| 17 | 2026-10-08 | 678.0000 |
| 18 | 2026-10-09 | – |

### Errors on this run

_None._

