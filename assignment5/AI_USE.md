# AI use disclosure: Assignment 5

**Tools used:** Claude (via Claude Code, in a cloud session), for design, code, data collection
runs, analysis and drafting. No other AI tools.

---

## What the model did

Essentially all of the implementation:

* the data layer: `src/data_loader.py`, `src/prediction_markets.py`, `src/data_quality.py`,
  `audit_data.py`
* the text layer: `src/nlp_topics.py` (twitterapi.io client with a credit ledger, GDELT client,
  quality filters, issue-conditioned tone), and the LSA engine in `src/nlp_svd_topics.py`
* the econometrics in `src/granger_econometrics.py` and the orchestration in `src/pipeline.py`
* the tables, the dashboard, the report builder, the unit tests, `BRAINSTORMING.md`,
  `DATA_AUDIT.md` and the report text

It also ran every download and every analysis step, and wrote the first interpretation of the
results.

## Decisions I made

Recorded from the session, in order:

1. **Real data only.** The model started building a synthetic data generator when the cloud
   environment's network blocked every data host. I stopped it and asked for real data
   sources instead. I then widened the environment's network allowlist so the APIs could be
   reached.
2. **Data before analysis.** I asked for the data to be collected, cleaned and audited first,
   and for the implementation to be discussed only after that.
3. **Topics.** I asked for homebuyer/housing affordability and healthcare affordability to be
   tracked as their own topics.
4. **Paid API budget.** I supplied the twitterapi.io key and capped spending at 100,000 of the
   account's 115,000 credits, keeping 15,000 in reserve. The final spend was 93,450, every call
   recorded in a ledger.
5. **Failure review.** I asked for every failed task to be listed with how it was recovered.
   That led to resumable GDELT caching, retrying empty tweet pages instead of caching them,
   a query-length guard, and vendor triangulation once Stooq blocked downloads.
6. **Specification.** I revised the brief to an LSA/SVD design, with a Democratic-sweep contract
   and a single-stock basket.
7. **News source.** I chose GDELT over RSS, because RSS has no history for a backtest.
8. **Topic review.** I reviewed the LSA topics (k = 30, varimax) before the daily series and
   tests were built on them.

## Errors that were caught and corrected during the build

* **Synthetic data started without asking.** When the data hosts were blocked, the model began
  writing a synthetic data generator rather than stopping to say so. It was removed. The
  pipeline now fails loudly when real data cannot be reached.
* **Prediction-market timing.** Daily bars from Polymarket and Kalshi close after the 16:00 ET
  equity close, so a daily ΔProb overlapped two return days. This was replaced with hourly
  quotes taken at the close.
* **Polymarket coverage.** The first version matched only 1–12 days of Polymarket data, because
  the D and R quotes were joined on exact timestamps.
* **GDELT outages.** GDELT outage days were first missed when the day was absent entirely, or
  sat at the end of the sample. Both cases are now flagged and masked.
* **LSA labelling.**
  * Component 0 was force-labelled "General". A test showed this can delete a real issue, so
    the label is now applied only after a one-signed, no-dominant-issue check.
  * Pole labels from seed-word counts mislabelled mixed poles, so they were replaced with
    loading-energy shares.
  * Linear topic weights let one issue's tone leak into another's; the weights are now squared.
* **Sparse topics.** Daily sentiment for thin topics was read off a sliver of one tweet's
  loading, giving ±1 noise. It is now pooled over 5 sessions, with a minimum-evidence rule.
* **Lagged predictability.** Seven Stage 2 results survived the FDR correction. A robustness
  battery was added before any of them was reported, and two of the seven fail it.

## What I did myself (to be completed by me)

> _Describe here what you reviewed, re-derived or re-ran yourself, which parts of the code
> and output you checked line by line, and how you satisfied yourself that you understand
> each method and number in the report._
