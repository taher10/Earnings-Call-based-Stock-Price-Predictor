# Earnings-Call Based Stock Price Predictor

A complete NLP pipeline that reads earnings call transcripts and produces **BUY / SELL / HOLD** recommendations by learning which language patterns historically preceded significant stock outperformance vs. the S&P 500.

---

## Table of Contents

1. [What the model does](#what-the-model-does)
2. [Quick Start](#quick-start)
3. [Data Sources](#data-sources)
4. [Step-by-Step Pipeline Walkthrough](#step-by-step-pipeline-walkthrough)
5. [Output Files](#output-files)
6. [CLI Reference](#cli-reference)
7. [Architecture Overview](#architecture-overview)
8. [Avoiding Lookahead Bias](#avoiding-lookahead-bias)

---

## What the model does

Every quarter, a company holds an earnings call. Executives speak for ~60 minutes presenting results, giving forward guidance, and answering analyst questions. The tone and language of that call contains real signal about the company's trajectory.

This pipeline:
1. Reads earnings call transcripts (local Kaggle dataset, ~50 companies, 2017-2025)
2. Computes how the stock moved in the **7 trading days after** each call
3. **Trains** a model to learn which language patterns preceded large stock moves
4. **Tests** that model on unseen future transcripts (2024+) to see if those patterns still hold

The target variable is not raw price return — it is **alpha**: return normalized by the stock's own volatility, so a calm stock and a volatile stock are measured on the same scale.

---

## Quick Start

```bash
# Default run: AAPL only, output to models/
python main.py

# Mag 7 stocks with a 1-sigma label threshold
python main.py --fetch-mag7 --out-dir models_mag7 --pct-threshold 1.0

# Specific tickers
python main.py --fetch-tickers AAPL,MSFT,AMZN --pct-threshold 1.0 --out-dir models_custom

# Launch the dashboard after training
streamlit run dashboard.py
```

---

## Data Sources

### Transcripts
Loaded from a local Kaggle archive (`Kaggle Dataset/archive/`) containing `.txt` files named `{YEAR}_Q{Q}_{ticker}_processed.txt`. Coverage: ~50 companies, 2017-2025, ~1,185 files. Each transcript is cached to `data/transcripts/cache/{TICKER}_{YEAR}_Q{N}.csv` after first load.

### Prices
Fetched live from Yahoo Finance via `yfinance`. SPY (S&P 500 ETF) prices are also fetched for market-relative return computation.

---

## Step-by-Step Pipeline Walkthrough

### Step 1 - Load Transcripts

For each requested ticker, the pipeline scans the Kaggle archive for matching `.txt` files, parses the year and quarter from the filename, and loads the raw text.

**Example:** For AAPL, `2024_Q2_aapl_processed.txt` is loaded and tagged:
```
ticker: AAPL  |  year: 2024  |  quarter: 2  |  date: 2024-06-30
```
Each transcript date is set to the end-of-quarter date. The full text covers prepared remarks and analyst Q&A.

**Output file:** `step1_transcripts.csv` - columns: `ticker, year, quarter, date, content`

---

### Step 2 - Fetch Prices & Compute Returns

For each transcript, the pipeline fetches daily closing prices around the call date and computes a **window return** - how much the stock moved in the 7 trading days after the call.

```
window_return = (price on day+7) / (price on day of call) - 1
```

A **volatility** estimate is computed from the 60 trading days *before* the call (std dev of daily returns), then:

```
vol_adj_return = window_return / historical_volatility
```

This puts returns on a comparable scale. An NVDA +5% move on a high-volatility stock is worth less than an MSFT +5% move on a stable stock.

**Example:**

| Ticker | Date       | Window Return | Volatility  | Vol-Adj Return |
|--------|------------|--------------|-------------|----------------|
| AAPL   | 2023-03-31 | +3.2%        | 1.8% daily  | +1.78          |
| MSFT   | 2023-03-31 | +2.1%        | 1.6% daily  | +1.31          |

**Output file:** `step2_window.csv` - adds: `before_close, after_close, window_return, future_return, volatility, vol_adj_return`

---

### Step 3 - Label Each Transcript

`vol_adj_return` values are cross-sectionally **Z-score normalized** across all transcripts:

```
vol_adj_zscore = (vol_adj_return - mean) / std_dev

label = BUY  (1)  if vol_adj_zscore >= +pct_threshold
label = SELL (0)  if vol_adj_zscore <= -pct_threshold
label = HOLD (2)  otherwise
```

Default `--pct-threshold` is **1.0** (1 standard deviation). Roughly 15-20% of transcripts get BUY or SELL labels.

**Why Z-score, not raw threshold?** Raw `vol_adj_return` values are tiny numbers like `0.003` or `-0.012`. Comparing them to a fixed threshold (e.g., `0.1`) puts nearly every transcript in HOLD. Z-scoring converts them to standard deviation units where the threshold is meaningful regardless of dataset size or volatility regime.

**Example (threshold = 1.0):**

| Transcript   | Vol-Adj Return | Z-Score | Label    |
|-------------|----------------|---------|----------|
| AAPL Q2 2022 | +0.094        | +2.1    | **BUY**  |
| MSFT Q3 2022 | -0.068        | -1.5    | **SELL** |
| AAPL Q1 2023 | +0.003        | +0.1    | HOLD     |

**Output file:** `aligned.csv` - adds: `vol_adj_zscore, label`

---

### Step 4 - Clean & Preprocess Text

The cleaning pipeline strips everything that does not carry predictive meaning:

1. **Timestamps & speaker labels** - `[00:01:23]` and `Operator:` removed
2. **Vocal tics** - `"thank you"`, `"good morning"`, `"next question"` stripped
3. **Short tokens** - words under 4 characters dropped
4. **Lemmatization** - "earnings" -> "earning", "improving" -> "improve"
5. **Entity masking** - "iPhone 15" -> [PRODUCT_GEN], "M3 chip" -> [CHIP_GEN]
6. **Section extraction** - split into management remarks vs. analyst Q&A

**Example:**
```
Before:
  "Good afternoon everyone. Thank you. The iPhone 15 revenue grew 17%
   driven by strong M3 chip demand. We expect continued improvement."

After:
  "afternoon revenue grew driven strong [CHIP_GEN] demand expect continued improvement"
```

**Output file:** `step3_features.csv` / `cleaned.csv` - adds: `has_future_outlook, numeric_count`

---

### Step 5 - Walk-Forward Train/Test Split

The dataset is split by **date**. All transcripts on/before `2023-12-31` go into train. All transcripts after go into test.

This prevents **lookahead bias** - random splitting would let the model see future language patterns during training. Temporal splitting simulates real-world deployment.

Any training rows with `NaN` labels (transcripts too recent to have a complete 7-day forward price window) are dropped before fitting. This prevents sklearn from crashing on NaN in `y`.

**Example (Mag 7 run):**
```
Total: 91 transcripts

Train: 66 transcripts (2021 Q1 -> 2023 Q4)
  Labels: HOLD=59, SELL=5, BUY=2

Test:  25 transcripts (2024 Q1 -> 2025 Q3)
```

**Output files:** `train.csv`, `test.csv`

---

### Step 6 - Stage 1 Model: TF-IDF + Logistic Regression

Cleaned text is converted to TF-IDF vectors (30k features, 1-3 ngrams, 180+ financial stop-words removed). A Logistic Regression classifier learns which word patterns predict BUY / SELL / HOLD.

**Example learned patterns:**
```
Top BUY signals:   "beat", "record revenue", "strong demand", "accelerating"
Top SELL signals:  "headwind", "uncertain", "lower guidance", "decelerate"
```

The classifier scores individual **sentences** separately (not the whole transcript at once). For each sentence it computes `P(BUY) - P(SELL)`, then takes the **median** across all sentences. Using the median (not mean) prevents one extremely bullish sentence in an otherwise cautious call from inflating the score.

---

### Step 7 - Stage 2 Model: Meta-Model (Ridge Regression)

A Ridge Regression model is trained to predict the actual `vol_adj_return` from 7 engineered features, giving a continuous score useful for ranking stocks.

| Feature              | How computed                                  | What it captures                                      |
|---------------------|-----------------------------------------------|-------------------------------------------------------|
| `full_sentiment`    | Median per-sentence P(BUY)-P(SELL), all text  | Overall tone of the call                              |
| `mgmt_sentiment`    | Same, management remarks only                 | What executives actually said                         |
| `qa_sentiment`      | Same, Q&A only. **Doubled if negative**       | Analyst stress signal                                 |
| `obfuscation_penalty`| 1 - (hedging density x 0.5)                  | How evasive management was                            |
| `numeric_count`     | Count of numbers in transcript                | Concreteness of the call                              |
| `outlook_flag`      | 1 if forward-looking words present            | Whether guidance was given                            |
| `qa_complexity`     | Avg sentence length in Q&A                   | Defensive/long answers = bearish                      |

Features are Z-score scaled on training stats. Ridge (alpha=1.0) predicts return. Bottom 50% of coefficients by magnitude are pruned.

**Example feature vector:**
```
full_sentiment:       +0.18   (moderately positive)
mgmt_sentiment:       +0.22   (confident)
qa_sentiment:         -0.09   (skeptical analysts) x2 = -0.18
obfuscation_penalty:   0.80   (some hedging)
numeric_count:          142   (concrete call)
outlook_flag:             1   (guidance given)
qa_complexity:         18.4   (moderate answer length)

Meta-model output:    +0.31   (predicted to outperform)
```

---

### Step 8 - Scoring Future Transcripts

The meta-model output passes through three multipliers:

**1. Obfuscation penalty**
```
obfuscated_score = meta_pred x (1 - hedging_density x 0.5)
```
Vague language ("we may consider", "possibly") penalizes the score by up to 50%.

**2. Temporal decay**
```
decayed_score = obfuscated_score x decay_multiplier
```
Older calls are down-weighted.

**3. Velocity (momentum)**
```
final_score = decayed_score x velocity
```
Tracks each ticker's sentiment trend over recent quarters:
- Declining trend -> x0.85 (15% penalty)
- Rising trend -> x1.05 (5% boost)
- Stable -> x1.00

The asymmetry (-15% vs +5%) is deliberate: deteriorating trends are penalized more aggressively than improving ones are rewarded.

**Label assignment:** `threshold_buy` = 85th percentile of training scores, `threshold_sell` = 15th percentile. Final score is compared to these thresholds. If test scores cluster too tightly (std < 0.05), thresholds adapt to the test set's own 75th/25th percentiles.

**Output file:** `results.csv` - key columns: `learned_score, learned_label, velocity_zscore, guidance_divergence`

---

### Step 9 - Evaluation: IC & Bootstrap

**Information Coefficient (IC)**

IC is the Spearman rank correlation between `learned_score` and actual `vol_adj_return` on the test set. It measures ranking ability rather than absolute accuracy.

```
IC = +1.0  -> model's top-ranked calls had the best actual returns
IC =  0.0  -> random, no predictive signal
IC = -1.0  -> model consistently got it backwards
```

IC > 0.05 is considered meaningful in practice; IC > 0.15 is strong for earnings NLP signals.

**How it works:**
1. Rank all test transcripts by `learned_score` (rank 1 = most bullish prediction)
2. Rank the same transcripts by actual `vol_adj_return` (rank 1 = best actual return)
3. Spearman correlation measures how well those two rank orderings agree

**Bootstrap IC**

Runs 100 random 80/20 train/test splits, re-fitting the full model each time:

```
For each of 100 iterations:
  1. Randomly split all data 80% train / 20% test
  2. Re-fit the full two-stage model on the 80%
  3. Score the held-out 20%, compute Spearman IC

Bootstrap IC mean = average IC across 100 runs
Bootstrap IC std  = variation across runs (lower = more consistent signal)
```

A high std (e.g. mean=0.08, std=0.25) means the model is sensitive to which transcripts land in the test set - common with small datasets.

**Output file:** `evaluation.txt`

---

## Output Files

| File                               | Description                                       |
|------------------------------------|---------------------------------------------------|
| `step1_transcripts.csv`            | Raw loaded transcripts, one row per quarter       |
| `step2_window.csv`                 | Transcripts + window returns + volatility         |
| `aligned.csv`                      | Full dataset with Z-score labels                  |
| `step3_features.csv` / `cleaned.csv` | Cleaned text after preprocessing               |
| `train.csv`                        | Training set (on/before 2023-12-31) with labels  |
| `test.csv`                         | Test set (after 2023-12-31) with learned scores  |
| `results.csv`                      | Final predictions on test set                    |
| `evaluation.txt`                   | IC, bootstrap IC, classification report          |

---

## CLI Reference

```bash
python main.py [options]
```

| Argument               | Default    | Description                                        |
|------------------------|------------|----------------------------------------------------|
| `--fetch-mag7`         | off        | Fetch all Mag 7 tickers                            |
| `--fetch-tickers A,B,C`| -          | Custom comma-separated ticker list                 |
| `--fetch-ticker X`     | AAPL       | Single ticker                                      |
| `--pct-threshold`      | 1.0        | Z-score sigma threshold for BUY/SELL               |
| `--days-forward`       | 7          | Trading days ahead for return computation          |
| `--days-before`        | 5          | Days before call in the alignment window           |
| `--days-after`         | 5          | Days after call in the alignment window            |
| `--train-until`        | 2023-12-31 | Walk-forward cutoff date                           |
| `--out-dir`            | models     | Output directory for all CSVs                     |
| `--require-outlook`    | off        | Drop transcripts without guidance language         |
| `--min-numeric N`      | 0          | Drop transcripts with fewer than N numbers         |
| `--split-frac`         | -          | Fractional split instead of fixed date             |
| `--start-year`         | (auto)     | Earliest year to include                           |
| `--end-year`           | (auto)     | Latest year to include                             |

---

## Architecture Overview

```
Raw .txt files (Kaggle archive)
        |
        v
[Step 1] Load + tag by ticker / year / quarter
        |
        v
[Step 2] Fetch prices  ->  window_return, vol_adj_return
        |
        v
[Step 3] Z-score normalize  ->  vol_adj_zscore  ->  BUY/SELL/HOLD label
        |
        v
[Step 4] Clean text: vocal tics, lemmatize, mask entities, split sections
        |
        v
[Step 5] Walk-forward split: train <= 2023-12-31 | test > 2023-12-31
        |                     drop NaN labels before fit
        v
[Step 6] TF-IDF (30k features, 1-3 ngrams)
        + Logistic Regression  ->  P(BUY), P(SELL), P(HOLD) per sentence
        |
        v
[Step 7] 7-feature extraction per transcript
        + Ridge meta-model (fit to predict vol_adj_return)  ->  continuous score
        |
        v
[Step 8] x obfuscation  x temporal decay  x velocity
        |
        v
        final_score  ->  threshold comparison  ->  BUY / SELL / HOLD
        |
        v
[Step 9] IC = Spearman rank corr(learned_score, vol_adj_return)
         Bootstrap IC = mean +/- std over 100 random 80/20 re-fits
```

---

## Avoiding Lookahead Bias

1. **Labels** computed from future returns - but the model only learns from past calls
2. **Walk-forward split** hardcoded at 2023-12-31 - never randomized
3. **Volatility** estimated from the 60 days *before* the call only
4. **Thresholds** computed on training scores only, applied unchanged to test set
5. **Sentiment velocity** uses only past calls per ticker - never the call being scored
6. **Bootstrap IC** re-fits from scratch each fold - no cross-fold information leakage
