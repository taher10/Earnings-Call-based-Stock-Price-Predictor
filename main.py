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

	transcripts = di.load_transcripts_csv(args.transcripts)
	prices = di.load_stock_csv(args.prices)
	df = di.align_and_label(transcripts, prices, days_forward=args.days_forward, pct_threshold=args.pct_threshold)

	# clean transcripts – store result in a temporary column so we can keep the
	# original text around until we've finished aligning/labeling.  once the
	# clean version exists we drop the raw column to avoid having two
	# ``transcript`` columns after the rename step.
	df["clean_transcript"] = df["transcript"].fillna("").apply(dc.clean_text)
	df = df.drop(columns=["transcript"]).rename(columns={"clean_transcript": "transcript"})

	# temporal split and train
	train_df, test_df = tm.temporal_split(df, args.train_until)
	print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
	tm.fit(train_df)
	if test_df.empty:
		print("warning: no data after cutoff, skipping evaluation")
		report = None
	else:
		report = tm.evaluate(test_df)
		print("Evaluation:\n", report)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
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

