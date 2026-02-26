from pathlib import Path
import argparse
import pandas as pd

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

	# clean transcripts
	df["clean_transcript"] = df["transcript"].fillna("").apply(dc.clean_text)

	# temporal split and train
	train_df, test_df = tm.temporal_split(df.rename(columns={"clean_transcript": "transcript"}), args.train_until)
	print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
	tm.fit(train_df)
	report = tm.evaluate(test_df)
	print("Evaluation:\n", report)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	model_path = out_dir / "model.pkl"
	tm.save(str(model_path))
	print("Saved model to", model_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--transcripts", required=True, help="CSV of transcripts with columns: date, transcript, company/ticker(optional)")
	parser.add_argument("--prices", required=True, help="CSV of daily prices with columns: date, close, ticker(optional)")
	parser.add_argument("--train-until", dest="train_until", default="2021-12-31")
	parser.add_argument("--days-forward", type=int, default=7)
	parser.add_argument("--pct-threshold", type=float, default=0.02)
	parser.add_argument("--out-dir", default="models")
	args = parser.parse_args()
	train(args)

