# Assignment 5: Voter Sentiment, Topic Salience and Election Prediction Markets

Work in progress. Code and unit tests are in place; no real data has been downloaded yet.

```bash
pip install -r requirements.txt
export TWITTERAPI_IO_KEY=...        # optional; without it only GDELT headlines are used
python -m pytest -q                 # unit tests
python build_report.py              # downloads data, writes assignment5_report.pdf + notebook
streamlit run app.py                # dashboard (after build_report.py)
```

The pipeline uses real data only. It needs network access to:
query1/query2.finance.yahoo.com, fc.yahoo.com, stooq.com, fred.stlouisfed.org,
gamma-api.polymarket.com, clob.polymarket.com, api.elections.kalshi.com, www.predictit.org,
api.twitterapi.io, api.gdeltproject.org. If none can be reached, it stops with an error.

Modules: `src/data_loader.py`, `src/prediction_markets.py`, `src/nlp_topics.py`,
`src/granger_econometrics.py`, `src/table_generator.py`, `src/pipeline.py`.
