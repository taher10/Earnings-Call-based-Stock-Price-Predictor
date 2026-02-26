import sys, os

# ensure project root is on PYTHONPATH so we can import modules stored there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_ingestion import DataIngestion
import pandas as pd

def test_earningscall_fetch():
    di = DataIngestion()
    # fetch transcripts for the last 5 years (2021-2025 inclusive)
    years = list(range(2021, 2026))
    df = di.fetch_transcripts_earningscall("AAPL", years=years)
    print(f"fetched {len(df)} rows")
    print(df.head())
    df.to_csv("test_transcripts.csv", index=False)  # save to CSV for manual inspection if needed
    # basic sanity checks
    assert not df.empty, "no transcripts returned"
    assert "transcript" in df.columns



def test_load_sample_csv():
    """Ensure the sample CSV can be loaded and dates normalized."""
    di = DataIngestion()
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_transcripts.csv"))
    df = di.load_transcripts_csv(path)
    assert not df.empty
    # dates should be datetime and tz-naive
    assert df["date"].dtype == "datetime64[ns]"
    assert "ticker" in df.columns


def test_alignment_and_model_pipeline():
    """Verify that the sample data can be aligned and a model trained.

    This exercises the logic added to avoid duplicate ``transcript`` columns and
    ensures that ``TextModel.fit`` no longer fails with shape mismatches.
    """
    di = DataIngestion()
    import data_model
    tm = data_model.TextModel()

    # load provided samples from the repo
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    trans_path = os.path.join(base, "data", "sample_aapl_transcripts.csv")
    price_path = os.path.join(base, "data", "sample_aapl_prices.csv")
    transcripts = di.load_transcripts_csv(trans_path)
    prices = di.load_stock_csv(price_path)

    df = di.align_and_label(transcripts, prices)
    # there should only be one column named transcript after cleaning
    cols = list(df.columns)
    assert cols.count("transcript") == 1, f"duplicate transcript columns: {cols}"

    # clean, split, and train on a tiny dataset - mostly checking that it runs
    import data_cleaning
    df["clean_transcript"] = df["transcript"].fillna("").apply(
        data_cleaning.DataCleaning().clean_text
    )
    df = df.drop(columns=["transcript"]).rename(columns={"clean_transcript": "transcript"})
    train_df, test_df = tm.temporal_split(df, train_until="2020-06-01")
    # fit should not raise an error even with only a few rows
    tm.fit(train_df)
    # evaluation should also run, returns a string report
    report = tm.evaluate(test_df)
    assert isinstance(report, str)


def test_train_cli_with_defaults(capsys):
    """Ensure main.train can be invoked with default paths and handles missing
    test data gracefully.
    """
    import argparse
    from main import train

    # create a namespace mimicking argparse with defaults adjusted to point at
    # the sample files.  we rely on main.py having the expected defaults for
    # transcripts/prices; this simply exercises the early path and the empty
    # test set logic added above.
    ns = argparse.Namespace(
        transcripts=os.path.join(os.path.dirname(__file__), "..", "data", "sample_aapl_transcripts.csv"),
        prices=os.path.join(os.path.dirname(__file__), "..", "data", "sample_aapl_prices.csv"),
        train_until="2099-01-01",  # far future ensures all rows land in train
        days_forward=7,
        pct_threshold=0.02,
        out_dir=os.path.join(os.path.dirname(__file__), "..", "models_test"),
    )
    # run train and capture stdout; should not raise
    train(ns)
    out, err = capsys.readouterr()
    assert "warning: no data after cutoff" in out


if __name__ == "__main__":
    test_earningscall_fetch()
    test_load_sample_csv()
    test_alignment_and_model_pipeline()
