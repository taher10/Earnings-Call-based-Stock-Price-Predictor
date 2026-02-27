from pathlib import Path
import argparse
import pandas as pd
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
        years = None
        if args.start_year is not None and args.end_year is not None:
            years = list(range(args.start_year, args.end_year + 1))
        transcripts = di.fetch_transcripts_earningscall(args.fetch_ticker, years=years)
        print(f"Fetched {len(transcripts)} transcripts for {args.fetch_ticker} (years={years})")
    else:
        # if the user is using the bundled sample dataset and the earningscall
        # library is installed, attempt to download a richer set of transcripts
        # (last five years for AAPL).  this makes the script behave more
        # usefully out of the box while still allowing explicit CSV paths.
        default_sample = "data/sample_aapl_transcripts.csv"
        if args.transcripts == default_sample:
            try:
                        # compute most recent five *completed* calendar years.  using the
                # current year would often include future dates (as illustrated by
                # the odd 2026 row earlier) which can't be aligned with prices and
                # confusing to users.  restricting to ``current - 5`` through
                # ``current-1`` keeps us safely in the past regardless of when the
                # script is run.
                import datetime
                current = datetime.datetime.now().year
                years = list(range(current - 5, current))
                transcripts = di.fetch_transcripts_earningscall("AAPL", years=years)
                print(f"Auto-fetched {len(transcripts)} transcripts for AAPL (years={years})")
            except ImportError:
                # earningscall not available; fall back to sample file
                transcripts = di.load_transcripts_csv(args.transcripts)
        else:
            transcripts = di.load_transcripts_csv(args.transcripts)

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

    df = di.align_and_label(transcripts, prices, days_forward=args.days_forward, pct_threshold=args.pct_threshold)

    # prepare output directory early so we can write intermediate CSVs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # write aligned dataframe for inspection
    aligned_path = out_dir / "aligned.csv"
    df.to_csv(aligned_path, index=False)
    print("Wrote aligned data to", aligned_path)

    # clean transcripts – keep original until we've saved the cleaned version
    df["clean_transcript"] = df.get("transcript", pd.Series()).fillna("").apply(dc.clean_text)
    df_clean = df.drop(columns=[c for c in ["transcript"] if c in df.columns]).rename(columns={"clean_transcript": "transcript"})

    # write cleaned dataframe
    cleaned_path = out_dir / "cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print("Wrote cleaned data to", cleaned_path)

    # temporal split and train
    train_df, test_df = tm.temporal_split(df_clean, args.train_until)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    tm.fit(train_df)

    # write train/test CSVs for inspection
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Wrote train/test splits to", train_path, test_path)

    if test_df.empty:
        print("warning: no data after cutoff, skipping evaluation")
        report = None
    else:
        report = tm.evaluate(test_df)
        print("Evaluation:\n", report)
        # save textual report for later inspection
        eval_path = out_dir / "evaluation.txt"
        with open(eval_path, "w") as fh:
            fh.write(report)
        print("Wrote evaluation to", eval_path)

    # save predictions/probabilities to CSV when a test set exists
    if not test_df.empty:
        X = test_df["transcript"].fillna("").tolist()
        preds = tm.pipeline.predict(X)
        results = test_df.copy()
        results["pred"] = preds
        # include class probabilities if available
        if hasattr(tm.pipeline, "predict_proba"):
            probs = tm.pipeline.predict_proba(X)
            for i, cls in enumerate(tm.pipeline.classes_):
                results[f"prob_{cls}"] = probs[:, i]
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
    parser.add_argument("--pct-threshold", type=float, default=0.02)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    # ensure the paths exist before we dive into training, gives a clearer
    # message than argparse's SystemExit 2.
    for name in ("transcripts", "prices"):
        path = getattr(args, name)
        if not os.path.isfile(path):
            parser.error(f"specified {name} file does not exist: {path}")
    train(args)

