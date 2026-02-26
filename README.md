# Earnings-Call Based Stock Recommendation

This repo provides a structured pipeline to convert earnings call transcripts into buy/sell/hold recommendations.

Quick overview:
- `data_ingestion.py`: load transcripts and prices, align transcripts to future returns, create labels.
- `data_cleaning.py`: simple transcript cleaning (timestamps, speaker labels, lemmatization).
- `data_model.py`: TF-IDF + LogisticRegression pipeline with temporal split to avoid lookahead bias.
- `main.py`: training orchestration — run to train and save a model.
- `dashboard.py`: Streamlit app to upload a transcript and get a recommendation.

Example training command:
```bash
python main.py --transcripts data/transcripts.csv --prices data/prices.csv --train-until 2021-12-31 --out-dir models
```

Run dashboard after training:
```bash
streamlit run dashboard.py
```

## Fetching Transcripts from a Website

Earnings–call transcripts are not distributed as CSVs; they live on web
pages.  The repo includes a helper that scrapes Seeking Alpha and returns a
`DataFrame` you can feed directly to `align_and_label` (or save to CSV for
later reuse).

```python
from data_ingestion import DataIngestion

di = DataIngestion()

# Option 1 – scrape SeekingAlpha (note: the site may block automated
# requests; you might need to adjust headers, maintain a session, or fetch
# pages manually if you run into HTTP 403).
transcripts_df = di.fetch_transcripts_seekingalpha("AAPL")

# Option 2 – use the `earningscall` library, which provides an API wrapper
# around a commercial transcript database.  Basic transcripts are free to
# query; you can also filter by year/quarter.
# di = DataIngestion()
# transcripts_df = di.fetch_transcripts_earningscall("AAPL", years=[2024])

# save if you want a CSV for the pipeline
transcripts_df.to_csv("data/aapl_transcripts.csv", index=False)
```

Then run the normal training command with
`--transcripts data/aapl_transcripts.csv` (or pass `transcripts_df`
directly to other methods).

Notes on avoiding lookahead bias:
- Labels are created based on future price movements after the transcript date.
- The `TextModel.temporal_split` splits train/test by date so that training data only contains transcripts that occurred on or before `train_until`.
