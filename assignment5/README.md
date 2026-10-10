# Assignment 5: Topic Modeling + Sentiment Analysis Around the US Midterms

FRE-GY 7871 · NLP and the Investment Process · 2026 US midterms (election day 3 Nov 2026)

Does voter discourse on X move election prediction markets, and do those markets move the
"midterm basket" of sector ETFs and single stocks? Uncentred LSA (truncated SVD, varimax)
extracts voter-priority topics from 5,816 filtered tweets. Topic-conditioned sentiment is
then tested against Kalshi/Polymarket House, Senate and Democratic-sweep probabilities
(Stage 1), and those probabilities against 19 assets (Stage 2), with FDR control, a
robustness battery, an SVD of the signal–return cross-covariance, and a Rigobon–Sack
heteroskedasticity check.

**Deliverables:** `assignment5_report.pdf` (executive briefing, Tables 1–3),
`assignment5_capstone.ipynb` (executed), `app.py` (Streamlit), `BRAINSTORMING.md` (Step 0),
`DATA_AUDIT.md` (data checks), `AI_USE.md`.

## Headline results (sample 2 Mar – 9 Oct 2026, 155 sessions)

* **Stage 1:** 0 of 21 sentiment → ΔProb tests survive the FDR correction. The covariance-SVD
  factors do not beat a permutation null, and the leading factor flips sign out of sample.
* **Stage 2:** 7 of 57 ΔProb → excess-return tests survive FDR, and 5 of those pass every
  robustness check. A rise in P(Democratic sweep) is followed the next session by
  underperformance of XLF and UNH (both hold in each half of the sample) and outperformance of
  XLK and NVDA (mainly the second half); P(Dem Senate) → META (second half). No volatility
  effects.
* **Rigobon–Sack:** the variance of ΔProb does rise on high-news days (Senate, Sweep), but not
  enough to identify contemporaneous effects. Every CI includes zero.

## Layout

| Path | What it does |
|---|---|
| `src/config.py` | Sample, topics and seed words, the asset basket, contract IDs, budgets |
| `src/data_loader.py` | Yahoo/FRED (Stooq backup), NYSE calendar, 16:00 ET session mapping, gap handling |
| `src/prediction_markets.py` | Polymarket CLOB + Kalshi hourly quotes at 16:00 ET; PredictIt cross-check; sweep contract |
| `src/nlp_topics.py` | twitterapi.io client with credit ledger, GDELT client, quality filters, issue-conditioned VADER |
| `src/nlp_svd_topics.py` | TF-IDF → truncated SVD (LSA) → varimax, pole labels, stability, daily projection, S_k,t, covariance SVD |
| `src/granger_econometrics.py` | ADF, lag selection, Granger F (HC1/HAC), BH-FDR, robustness battery, factor OOS, Rigobon–Sack (+ IV form), IRFs, event studies |
| `src/table_generator.py` | Tables 1A/1B/2/3 as Markdown and LaTeX (`outputs/tables/`) |
| `src/data_quality.py`, `audit_data.py` | Data audit → `DATA_AUDIT.md` |
| `src/pipeline.py` | End-to-end run shared by the report, notebook and app |
| `collect_tweets.py` | Paid tweet collection (budgeted, cached, ledgered) |
| `fit_topics.py` | Step-1 topic diagnostics |
| `build_report.py` | PDF + executed notebook |
| `app.py` | Streamlit dashboard |
| `tests/` | 52 unit tests |

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q                                   # unit tests

# 1. data (free sources; resumable, cached under data/raw/)
python audit_data.py                                  # prices, FRED, prediction markets + audit
python -c "from src import nlp_topics as nt, config; nt.collect_gdelt_timelines(config.START, config.END)"

# 2. tweets (paid: 15 credits per tweet; stops at TWITTER_BUDGET_CREDITS)
export TWITTERAPI_IO_KEY=...
python collect_tweets.py --stratum p                  # general issue query, 1 page per session
python collect_tweets.py --stratum a                  # affordability oversample

# 3. analysis and deliverables
python build_report.py                                # PDF + executed notebook (cached data)
streamlit run app.py
```

Network hosts needed: `query1/query2.finance.yahoo.com`, `fc.yahoo.com`,
`fred.stlouisfed.org`, `stooq.com`, `gamma-api.polymarket.com`, `clob.polymarket.com`,
`api.elections.kalshi.com`, `www.predictit.org`, `api.twitterapi.io`,
`api.gdeltproject.org`.

There is no simulated-data fallback: if a source cannot be reached and nothing is cached,
the run stops and says which source failed. `data/` is not committed: tweet text cannot be
redistributed, and everything else is reproducible from the scripts.
