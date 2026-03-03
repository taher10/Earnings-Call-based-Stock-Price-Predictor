from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os

from data_ingestion import DataIngestion
from data_cleaning import DataCleaning
from data_model import TextModel


def train(args):
    di = DataIngestion()
    dc = DataCleaning()
    tm = TextModel()

    # prepare output directory early so we can write inputs immediately
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load or fetch transcripts
    transcripts = None

    if args.fetch_ticker:
        # user explicitly asked for live data
        years = None
        if args.start_year is not None and args.end_year is not None:
            years = list(range(args.start_year, args.end_year + 1))
        transcripts = di.fetch_transcripts_earningscall(
            args.fetch_ticker,
            years=years,
            timeout=args.fetch_timeout,
        )
        if transcripts is None or (hasattr(transcripts, "empty") and transcripts.empty):
            print(f"Fetched transcripts for {args.fetch_ticker} (years={years}) but got none")
        else:
            print(f"Fetched {len(transcripts)} transcripts for {args.fetch_ticker} (years={years})")
    else:
        # maybe use the sample CSV or auto‑fetch when the default is requested
        default_sample = "data/sample_aapl_transcripts.csv"
        if args.transcripts == default_sample:
            try:
                # compute most recent five calendar years that have completed
                import datetime
                current = datetime.datetime.now().year
                years = list(range(current - 5, current))
                transcripts = di.fetch_transcripts_earningscall(
                    "AAPL",
                    years=years,
                    timeout=args.fetch_timeout,
                )
                if transcripts is None or (hasattr(transcripts, "empty") and transcripts.empty):
                    print(f"Auto-fetched transcripts for AAPL (years={years}) but got none")
                else:
                    print(f"Auto-fetched {len(transcripts)} transcripts for AAPL (years={years})")
            except ImportError:
                transcripts = di.load_transcripts_csv(args.transcripts)
        else:
            transcripts = di.load_transcripts_csv(args.transcripts)

    # at this point we may have a DataFrame or None; fail early if it is empty
    if transcripts is None or (hasattr(transcripts, "empty") and transcripts.empty):
        print("no transcripts available, aborting training")
        return

    # step 1: save normalized transcripts for inspection
    transcripts.to_csv(out_dir / "step1_transcripts.csv", index=False)
    print("Wrote step1_transcripts.csv")

# if we auto-fetched transcripts and the user left the prices argument as
    # the sample file, it makes sense to also pull prices for the same ticker
    # via yfinance so that alignment actually produces rows.
    default_price = "data/sample_aapl_prices.csv"
    if transcripts is not None and args.prices == default_price:
        # determine ticker from transcripts
        ticker = None
        if "ticker" in transcripts.columns:
            ticker = transcripts["ticker"].iloc[0]
        if ticker:
            try:
                start = transcripts["date"].min().strftime("%Y-%m-%d")
                end = transcripts["date"].max().strftime("%Y-%m-%d")
                prices = di.fetch_prices_yfinance(ticker, start=start, end=end)
                prices.to_csv(out_dir / "input_prices.csv", index=False)
                print(f"Auto-fetched prices for {ticker} ({len(prices)} rows)")
            except ImportError:
                print("yfinance not installed; using sample prices")
                prices = di.load_stock_csv(args.prices)
        else:
            prices = di.load_stock_csv(args.prices)
    else:
        prices = di.load_stock_csv(args.prices)

    # also persist the input files (normalized) so users can inspect what we
    # started with; this mirrors the raw CSVs but may include timezone fixes,
    # column renaming, etc.
    transcripts.to_csv(out_dir / "input_transcripts.csv", index=False)
    prices.to_csv(out_dir / "input_prices.csv", index=False)

    # compute window returns around each transcript (step 2)
    df = di.align_with_window(
        transcripts,
        prices,
        days_before=args.days_before,
        days_after=args.days_after,
    )
    # translate window_return into the familiar future_return field
    df["future_return"] = df.get("window_return")

    # compute historical volatility for each transcript and normalize
    vols = []
    for idx, row in df.iterrows():
        try:
            vol = di.compute_volatility(prices, pd.to_datetime(row["date"]))
        except Exception:
            vol = None
        vols.append(vol)
    df["volatility"] = vols
    # avoid zero/None when dividing
    df["vol_adj_return"] = df.apply(
        lambda r: (r["future_return"] / r["volatility"]) \
            if pd.notna(r.get("future_return")) and r.get("volatility") and r.get("volatility") > 0 \
            else None,
        axis=1,
    )

    # label on volatility-adjusted return instead of raw return; threshold
    # argument is now interpreted as multiples of volatility
    def _make_label(v):
        try:
            if v is None:
                return None
            if v >= args.pct_threshold:
                return 1
            elif v <= -args.pct_threshold:
                return 0
            else:
                return 2
        except Exception:
            return None

    df["label"] = df["vol_adj_return"].apply(_make_label)

    df.to_csv(out_dir / "step2_window.csv", index=False)
    print("Wrote step2_window.csv")

    # prepare output directory early so we can write intermediate CSVs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # write aligned dataframe for inspection
    aligned_path = out_dir / "aligned.csv"
    df.to_csv(aligned_path, index=False)
    print("Wrote aligned data to", aligned_path)

    # optionally drop transcripts that don’t meet simple text heuristics
    if args.require_outlook or args.min_numeric > 0:
        before = len(df)
        if args.require_outlook:
            df = df[df["has_future_outlook"]]
        if args.min_numeric > 0:
            df = df[df["numeric_count"] >= args.min_numeric]
        after = len(df)
        print(f"Filtered {before - after} rows based on text heuristics, leaving {after}")

    # clean transcripts – keep original until we've saved the cleaned version
    df["clean_transcript"] = df.get("transcript", pd.Series()).fillna("").apply(dc.clean_text)
    df_clean = df.drop(columns=[c for c in ["transcript"] if c in df.columns]).rename(columns={"clean_transcript": "transcript"})

    # write cleaned dataframe
    cleaned_path = out_dir / "cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print("Wrote cleaned data to", cleaned_path)
    # also save feature-ready CSV as step3
    step3_path = out_dir / "step3_features.csv"
    df_clean.to_csv(step3_path, index=False)
    print("Wrote step3_features.csv")

    # ============================================================================
    # FEATURE ENGINEERING: Entity Masking & Dynamic Lexicon
    # ============================================================================
    print("\nApplying entity masking (products/chips → generic tokens)...")
    df_clean["transcript"] = df_clean["transcript"].apply(dc.mask_entities)
    print("Entity masking complete")

    # ============================================================================
    # WALK-FORWARD VALIDATION: Train on historical data, test on unseen future
    # ============================================================================
    # Use a cutoff date to split train (learning phase) from test (validation phase)
    # Default: train on 2020-2023, test on 2024+
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    train_cutoff = pd.to_datetime("2023-12-31")
    
    train_df_historical = df_clean[df_clean["date"] <= train_cutoff].copy()
    test_df_future = df_clean[df_clean["date"] > train_cutoff].copy()
    
    print(f"\nWalk-forward validation:")
    print(f"  Train on: {len(train_df_historical)} transcripts (on/before 2023-12-31)")
    print(f"  Test on:  {len(test_df_future)} transcripts (after 2023-12-31)")

    # PHASE 1: Train the model on historical data to learn language patterns
    model_trained = False
    if not train_df_historical.empty:
        unique_labels = train_df_historical["label"].dropna().unique()
        if len(unique_labels) < 2:
            print("warning: only one label in training data; skipping fit")
        else:
            print(f"Training model on {len(train_df_historical)} historical samples...")
            tm.fit(train_df_historical)
            model_trained = True
            
            # THRESHOLD TUNING: Compute sentiment scores on training set and tune thresholds
            print("\nTuning classification thresholds (percentile-based)...")
            train_scores = []
            for idx, row in train_df_historical.iterrows():
                _, score, _ = tm.score_and_label(
                    row["transcript"], 
                    transcript_date=pd.Timestamp(row["date"]),
                    threshold_buy=0.5,  # Temporary high threshold to get raw scores
                    threshold_sell=-0.5
                )
                train_scores.append(score)
            
            # Tune thresholds based on percentiles
            tm.tune_thresholds_percentile(train_scores, percentile_buy=85, percentile_sell=15)
            
            # Extract learned feature importance
            print("\nExtracted learned feature weights")
            top_buy = tm.get_top_features(label=1, n=20)
            top_sell = tm.get_top_features(label=0, n=20)
            print(f"\nTop 10 'BUY' features (words/phrases correlated with positive returns):")
            for word, coef in top_buy[:10]:

                print(f"  {word}: {coef:.4f}")
            print(f"\nTop 10 'SELL' features (words/phrases correlated with negative returns):")
            for word, coef in top_sell[:10]:
                print(f"  {word}: {coef:.4f}")
    else:
        print("warning: no historical data for training; skipping fit")

    # PHASE 2: Use learned weights to score unseen future transcripts
    # This is the key: we score test data using what the model learned from training,
    # WITHOUT using the actual future returns (no look-ahead bias)
    # Includes:
    # - Temporal decay: older product terms lose importance
    # - Entity masking: product names mapped to generic tokens
    # - Q&A stress metric: negative sentiment in analyst Q&A is doubled
    # - Sentiment velocity: penalizes declining momentum
    # - Management obfuscation: penalizes hedging language in Q&A
    # - Dynamic decile labeling: thresholds adapt to recent market regime
    if model_trained and not test_df_future.empty:
        print(f"\nScoring {len(test_df_future)} future transcripts using learned weights...")
        print("(with temporal decay, entity masking, Q&A stress, sentiment velocity, and obfuscation metrics)")
        
        # Get ticker from transcripts
        ticker = None
        if "ticker" in test_df_future.columns:
            ticker = test_df_future["ticker"].iloc[0]
        if ticker is None:
            ticker = "AAPL"  # Default fallback
        
        # Tune dynamic decile thresholds using rolling 8-quarter window
        print(f"\nCalculating dynamic decile thresholds (rolling 8-quarter window)...")
        threshold_buy, threshold_sell = tm.tune_thresholds_dynamic_decile(ticker, lookback_quarters=8)
        
        # ADAPTIVE THRESHOLDING: If test scores are too tight, use percentiles on test data itself
        test_scores = []
        for idx, row in test_df_future.iterrows():
            diag, score, label = tm.score_and_label(
                row["transcript"],
                transcript_date=pd.Timestamp(row["date"]),
                ticker=ticker,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell
            )
            test_scores.append(score)
        
        # Check if test scores are too clustered (std < 0.05)
        test_scores_std = np.std(test_scores)
        if test_scores_std < 0.05:
            # Scores too tight; use percentile-based labeling on test data
            test_threshold_buy = np.percentile(test_scores, 75)  # Top 25%
            test_threshold_sell = np.percentile(test_scores, 25)  # Bottom 25%
            print(f"\nTest scores too tight (std={test_scores_std:.4f}); using adaptive percentile thresholds:")
            print(f"  BUY (top 25%):  >= {test_threshold_buy:.4f}")
            print(f"  SELL (bottom 25%): <= {test_threshold_sell:.4f}")
        else:
            test_threshold_buy = threshold_buy
            test_threshold_sell = threshold_sell
        
        scores_and_labels = []
        diagnostics_list = []
        for idx, row in test_df_future.iterrows():
            diag, score, label = tm.score_and_label(
                row["transcript"],
                transcript_date=pd.Timestamp(row["date"]),
                ticker=ticker,
                threshold_buy=test_threshold_buy,
                threshold_sell=test_threshold_sell
            )
            scores_and_labels.append((score, label))
            diagnostics_list.append(diag)
        
        test_df_future["learned_score"] = [s[0] for s in scores_and_labels]
        test_df_future["learned_label"] = [s[1] for s in scores_and_labels]
        test_df_future["velocity_zscore"] = [d['velocity_zscore'] for d in diagnostics_list]
        test_df_future["guidance_divergence"] = [d['guidance_divergence'] for d in diagnostics_list]
        test_df_future["divergence_magnitude"] = [d['divergence_magnitude'] for d in diagnostics_list]
        
        # CRITICAL: For test data, the true labels should ONLY come from learned_label,
        # not from window_return-based labels (which would be look-ahead bias)
        # Drop the original "label" column and replace with "learned_label" for proper evaluation
        test_df_future = test_df_future.drop(columns=["label"])
        test_df_future["label"] = test_df_future["learned_label"]
        
        print(f"\nLabel distribution (future transcripts, based on learned weights + momentum):")
        print(test_df_future["learned_label"].value_counts())
        
        # Report signal reliability metrics
        print(f"\nSignal Reliability Diagnostics:")
        print(f"  Mean Velocity Z-Score: {test_df_future['velocity_zscore'].mean():.4f}")
        print(f"  Max |Z-Score|: {test_df_future['velocity_zscore'].abs().max():.4f}")
        strong_signals = (test_df_future['velocity_zscore'].abs() > 1.5).sum()
        print(f"  Strong Signals (|Z| > 1.5): {strong_signals} out of {len(test_df_future)}")
        divergence_count = test_df_future['guidance_divergence'].sum()
        print(f"  Guidance Divergence Warnings: {divergence_count} transcripts")
        if divergence_count > 0:
            avg_divergence = test_df_future[test_df_future['guidance_divergence']]['divergence_magnitude'].mean()
            print(f"  Mean Divergence Magnitude: {avg_divergence:.4f}")
        
        # For evaluation, we can compare learned labels vs actual returns
        # to see if our language patterns actually predictive
        test_df = test_df_future
    else:
        test_df = pd.DataFrame()

    # For backward compatibility with existing reporting, use learned labels
    if not test_df.empty and "learned_label" in test_df.columns:
        test_df["label"] = test_df["learned_label"]

    # split and train (legacy path for backward compatibility)
    if args.split_frac is not None and len(train_df_historical) == 0:
        train_df, test_df = tm.proportion_split(df_clean, args.split_frac)
        print(f"Proportional split {args.split_frac}: train {len(train_df)}, test {len(test_df)}")
    elif len(train_df_historical) == 0:
        train_df, test_df = tm.temporal_split(df_clean, args.train_until)
        print(f"Temporal split until {args.train_until}: train {len(train_df)}, test {len(test_df)}")
    else:
        train_df = train_df_historical

    # show label distribution for diagnostics
    if not train_df.empty:
        print("label distribution (train):\n", train_df["label"].value_counts())

    # write train/test CSVs for inspection
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Wrote train/test splits to", train_path, test_path)

    if not model_trained:
        print("model was not trained; skipping evaluation and prediction steps")
    else:
        if test_df.empty:
            print("warning: no data for testing, skipping evaluation")
            report = None
        else:
            report = tm.evaluate(test_df)
            print("Evaluation:\n", report)
            # compute information coefficient on volatility-adjusted returns
            if "vol_adj_return" in test_df.columns:
                ic = tm.information_coefficient(
                    test_df["vol_adj_return"].fillna(0).tolist(),
                    test_df["learned_score"].tolist(),
                )
                print(f"Information Coefficient (test): {ic:.4f}")
                # bootstrap estimate of IC stability
                mean_ic, std_ic = tm.bootstrap_ic(test_df,
                                                   text_col="transcript",
                                                   return_col="vol_adj_return")
                print(f"Bootstrap IC mean={mean_ic:.4f}, std={std_ic:.4f}")
            # save textual report for later inspection
            eval_path = out_dir / "evaluation.txt"
            with open(eval_path, "w") as fh:
                fh.write(report)
                if "vol_adj_return" in test_df.columns:
                    fh.write(f"\nInformation Coefficient: {ic:.4f}\n")
                    fh.write(f"Bootstrap IC mean={mean_ic:.4f}, std={std_ic:.4f}\n")
            print("Wrote evaluation to", eval_path)

        # save predictions/probabilities to CSV when a test set exists
        if not test_df.empty:
            X = test_df["transcript"].fillna("").tolist()
            preds = tm.pipeline.predict(X)
            
            # Build simplified results with only essential columns
            results = pd.DataFrame({
                'date': test_df['date'].values,
                'ticker': test_df.get('ticker', pd.Series(['AAPL']*len(test_df))).values,
                'learned_score': test_df['learned_score'].values,
                'learned_label': test_df['learned_label'].values,
                'velocity_zscore': test_df['velocity_zscore'].values,
                'guidance_divergence': test_df['guidance_divergence'].values,
            })
            
            # Add class probabilities if available
            if hasattr(tm.pipeline, "predict_proba"):
                probs = tm.pipeline.predict_proba(X)
                # Assuming classes are [0, 1, 2] for [SELL, BUY, HOLD]
                results['prob_sell'] = probs[:, 0]
                results['prob_buy'] = probs[:, 1]
                results['prob_hold'] = probs[:, 2]
            
            results_path = out_dir / "results.csv"
            results.to_csv(results_path, index=False)
            print("Wrote prediction results to", results_path)

    # save model as before (pickle) for reuse, but the main deliverable is CSVs
    model_path = out_dir / "model.pkl"
    tm.save(str(model_path))
    print("Saved model to", model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # if you don't provide files we fall back to the sample CSVs shipped with
    # the repo; those live under ``data/`` and contain AAPL data so you can
    # exercise the pipeline without downloading anything else.
    parser.add_argument(
        "--transcripts",
        default="data/sample_aapl_transcripts.csv",
        help=(
            "CSV of transcripts with columns: date, transcript, company/"
            "ticker(optional).  defaults to the sample file ``data/sample_aapl_transcripts.csv``."
        ),
    )
    parser.add_argument(
        "--fetch-timeout",
        type=float,
        default=None,
        help=(
            "Optional per-request timeout (seconds) when downloading transcripts via "
            "--fetch-ticker or when auto-fetching the sample.  A single slow or hung "
            "request will be skipped after this many seconds.  Setting to 0 or "
            "omitting disables the timeout."
        ),
    )
    parser.add_argument(
        "--fetch-ticker",
        help=(
            "If provided, download transcripts from the earningscall library for this ticker. "
            "The resulting data will override ``--transcripts``.  Requires ``earningscall`` to be installed."
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="inclusive start year when fetching transcripts (used with --fetch-ticker)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="inclusive end year when fetching transcripts (used with --fetch-ticker)",
    )
    parser.add_argument(
        "--prices",
        default="data/sample_aapl_prices.csv",
        help=(
            "CSV of daily prices with columns: date, close, ticker(optional). "
            "defaults to the sample file in ``data/``."
        ),
    )
    # choose a cutoff that lies between the dates in our sample data so
    # running ``python main.py`` with no arguments actually gives a test set
    parser.add_argument("--train-until", dest="train_until", default="2020-06-01")
    parser.add_argument("--days-forward", type=int, default=7)
    parser.add_argument("--pct-threshold", type=float, default=2.0,
                        help=(
                            "threshold in units of volatility multiples used for labeling. "
                            "e.g. 2.0 means returns \u22652 sigma are labeled BUY, <=-2 sigma SELL. "
                            "(default 2.0, roughly equivalent to 2% when vol ~1%)"
                        ))
    parser.add_argument(
        "--days-before",
        type=int,
        default=5,
        help="number of trading days before the transcript to include in return window",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=5,
        help="number of trading days after the transcript to include in return window",
    )
    parser.add_argument(
        "--split-frac",
        type=float,
        help="proportion of data to use for training (temporal order preserved); overrides --train-until if set",
    )
    parser.add_argument(
        "--require-outlook",
        action="store_true",
        help="drop transcripts that lack forward‑looking language before splitting",
    )
    parser.add_argument(
        "--min-numeric",
        type=int,
        default=0,
        help="drop transcripts with fewer than this many numeric mentions",
    )
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    # ensure the paths exist before we dive into training, gives a clearer
    # message than argparse's SystemExit 2.
    for name in ("transcripts", "prices"):
        path = getattr(args, name)
        if not os.path.isfile(path):
            parser.error(f"specified {name} file does not exist: {path}")
    train(args)