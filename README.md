# Earnings-Call Based Stock Recommendation

This repo provides an institutional-grade pipeline to convert earnings call transcripts into buy/sell/hold recommendations based on NLP analysis. The system predicts **alpha** (outperformance vs. S&P 500) rather than absolute returns.

## Recent Updates (Institutional Grade Refinements)

**Latest improvements** to stabilize bootstrap variance and reduce linguistic artifacts:

1. **Strict Token Filtering** - Removes vocal tics (`"thank you"`, `"operator next"`) and short tokens (< 4 characters) to eliminate noise
2. **Sentence-Level Sentiment Aggregation** - Scores individual sentences and takes median (not mean) to reduce outlier impact
3. **Volatility-Adjusted Residual Labeling** - BUY signals only trigger for residual returns > **1.0σ** (previously 2.0σ) for more stringent thresholds
4. **Feature Importance Pruning** - Zeros out bottom 50% of meta-model coefficients by magnitude to reduce overfitting

## Architecture Overview

- `data_ingestion.py`: Load transcripts/prices, compute **residual returns** (stock - SPY), winsorize outliers, Z-score labeling
- `data_cleaning.py`: Advanced text preprocessing (vocal tic filtering, lemmatization, entity masking, linguistic complexity)
- `data_model.py`: Two-stage ensemble (TF-IDF + LogisticRegression → Ridge meta-model with 7 features)
- `main.py`: Training orchestration with walk-forward validation and bootstrap IC evaluation
- `dashboard.py`: Streamlit app to upload a transcript and get a recommendation

## Quick Start

Example training command:
```bash
python main.py --transcripts data/sample_aapl_transcripts.csv \
    --prices data/sample_aapl_prices.csv \
    --pct-threshold 1.0 \
    --out-dir models
```

The pipeline automatically:
1. Fetches SPY (S&P 500) prices for residual return calculation
2. Computes Z-scores for statistical labeling
3. Trains two-stage model with walk-forward validation (train ≤ 2023-12-31, test > 2023-12-31)
4. Outputs bootstrap IC evaluation with 100 iterations

Run dashboard after training:
```bash
streamlit run dashboard.py
```

## Data Pipeline Steps

The script performs the pipeline in explicit steps, emitting CSV files at each stage for inspection:

1. **step1_transcripts.csv** – Normalized input transcript data
2. **step2_window.csv** – Transcripts joined with residual returns, Z-scores, and volatility metrics
3. **step3_features.csv** – Cleaned text with linguistic features (hedging density, complexity, numeric counts)
4. **train.csv/test.csv** – Walk-forward split for model training (temporal cutoff: 2023-12-31)
5. **aligned.csv** – Full aligned dataset with all computed features
6. **evaluation.txt** – Bootstrap IC statistics and model performance metrics
7. **results.csv** – Predictions on test set with learned scores and labels

### Text Cleaning Pipeline

Applied transformations (in order):
1. **Remove timestamps**: `[00:01:23]` → (removed)
2. **Remove speaker labels**: `Operator:` → (removed)
3. **Lemmatization**: "earnings" → "earning", "improving" → "improve"
4. **Vocal tic filtering**: Remove bigrams (`"have have"`, `"thank you"`) and short tokens (< 4 chars)
5. **Entity masking**: `"iPhone 15"` → `[PRODUCT_GEN]`, `"M3 chip"` → `[CHIP_GEN]`
6. **Section extraction**: Split into management remarks vs. Q&A
7. **Feature extraction**: Compute hedging density, linguistic complexity, numeric mentions

### Optional Transcript Filtering

Build a more robust training set using convenience filters:

* `--require-outlook` - Drops transcripts without forward-looking language (*outlook*, *guidance*, *forecast*)
* `--min-numeric N` - Removes transcripts with fewer than N numeric mentions
* `--split-frac 0.8` - Use 80/20 temporal split instead of fixed cutoff date

Example with filters:
```bash
python main.py --transcripts data/sample_aapl_transcripts.csv \
    --prices data/sample_aapl_prices.csv \
    --require-outlook --min-numeric 5 \
    --pct-threshold 1.0 --out-dir models
```

### Key Configuration Parameters

- `--pct-threshold` (default: 1.0) - Z-score threshold for BUY/SELL labels (in std deviations)
- `--days-forward` (default: 7) - Trading days to look ahead for return computation
- `--train-until` (default: 2023-12-31) - Temporal cutoff for walk-forward validation
- `--out-dir` (default: models) - Output directory for trained models and CSVs

The pipeline predicts **alpha** (outperformance vs. S&P 500) using residual returns:

```python
residual_return = (stock_return - spy_return)
```

Returns are:
1. **Winsorized** at 5th/95th percentiles (removes outliers)
2. **Z-score normalized**: `(residual - mean) / std_dev`
3. **Labeled** based on statistical significance:
   - `Z-score ≥ 1.0σ` → **BUY** (beats market by 1 std dev)
   - `Z-score ≤ -1.0σ` → **SELL** (underperforms market)
   - Otherwise → **HOLD**

The `--pct-threshold` argument (default: **1.0**) controls the Z-score threshold in units of standard deviations. This institutional-grade threshold ensures BUY/SELL signals only trigger for statistically significant outperformance.

You'll see new columns `residual_return`, `residual_zscore`, and `residual_return_winsorized` in the output CSVs.

### Two-Stage Model Architecture

**Stage 1: TF-IDF + Logistic Regression**
- N-grams: 1-3 (captures phrases like "strong demand", "supply chain issues")
- Stop-words: 180+ financial boilerplate terms and vocal tics
- Max features: 30,000
- Classifier: LogisticRegression with class balancing

**Stage 2: Ridge Meta-Model**
Combines 7 engineered features:
1. **full_sentiment** - Sentence-level median sentiment (robust to outliers)
2. **mgmt_sentiment** - Management section sentiment
3. **qa_sentiment** - Q&A sentiment (2× penalty if negative = stress signal)
4. **obfuscation_penalty** - `1.0 - (hedging_density × 0.5)` (penalizes ambiguous language)
5. **numeric_count** - Number of figures/metrics mentioned
6. **outlook_flag** - Binary presence of forward-looking keywords
7. **qa_complexity** - Average sentence length in Q&A (defensiveness proxy)

Features are Z-score normalized, then fed to Ridge regression (α=1.0) to predict residual returns. The bottom 50% of features by coefficient magnitude are pruned to reduce overfitting.

### Signal Evaluation & Bootstrap IC

We no longer rely solely on accuracy. After training, the model reports the **Information Coefficient** (Spearman rank correlation) between the learned score and actual returns, along with a bootstrapped mean/std to gauge statistical significance. These metrics are written to `evaluation.txt` and printed to the console.

The bootstrap procedure runs 100 iterations with random train/test splits to measure stability. A tight IC distribution (low std) indicates consistent predictive power across different market conditions.

## Fetching Transcripts from Web Sources

Earnings call transcripts live on web pages. The repo includes helpers to fetch them:

```python
from data_ingestion import DataIngestion

di = DataIngestion()

# Option 1: earningscall library (recommended)
# Fetches from commercial API; basic transcripts are free
transcripts_df = di.fetch_transcripts_earningscall(
    "AAPL", 
    years=[2023, 2024], 
    quarters=[1, 2, 3, 4],
    timeout=30
)

# Option 2: Seeking Alpha scraper (may require session/headers)
# transcripts_df = di.fetch_transcripts_seekingalpha("AAPL")

# Save to CSV for reuse
transcripts_df.to_csv("data/aapl_transcripts.csv", index=False)
```

Then train with: `--transcripts data/aapl_transcripts.csv`

## Avoiding Lookahead Bias

The pipeline enforces temporal discipline:

1. **Labels** - Computed from future returns (7 days forward by default)
2. **Walk-forward validation** - Splits data by date (train ≤ cutoff, test > cutoff)
3. **No leakage** - Training only uses information available at transcript date
4. **Temporal decay** - Older product mentions decay exponentially
5. **Sentiment velocity** - Compares current sentiment to historical baseline

The model strictly separates training and test sets by time to simulate real-world deployment.
