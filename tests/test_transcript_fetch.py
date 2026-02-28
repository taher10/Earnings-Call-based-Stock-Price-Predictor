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

    # --- new functionality tests ------------------------------------------------
    # helper methods for text indicators
    assert di._has_future_outlook("we expect growth next quarter")
    assert not di._has_future_outlook("we reported last quarter results")
    assert di._count_numbers("revenue was $5.3 billion and margin 15%") >= 2

    # align_with_window should compute a centred return on the tiny sample
    window_df = di.align_with_window(transcripts, prices, days_before=1, days_after=1)
    # because our sample only has two transcripts 3 months apart, returns may be NaN
    assert "window_return" in window_df.columns

    # proportion_split should divide by date order
    df2 = pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "label": [0,1,0,1],
        "transcript": ["a","b","c","d"],
    })
    train2, test2 = tm.proportion_split(df2, 0.5)
    # first two rows (Jan, Feb) should be in train
    assert len(train2) == 2 and all(pd.to_datetime(train2["date"]) <= pd.to_datetime("2020-02-01"))
    assert len(test2) == 2


def test_train_cli_with_defaults(capsys):
    """Ensure main.train can be invoked with default paths and handles missing
    test data gracefully.
    """
    import argparse
    from main import train

    # create a namespace mimicking argparse with defaults adjusted to point at
    # the sample files.  the code may attempt to auto-fetch transcripts if
    # ``earningscall`` is installed; either way we should end up with at least
    # one transcript row (the sample has three, and a fetch would yield more).
    ns = argparse.Namespace(
        transcripts=os.path.join(os.path.dirname(__file__), "..", "data", "sample_aapl_transcripts.csv"),
        prices=os.path.join(os.path.dirname(__file__), "..", "data", "sample_aapl_prices.csv"),
        train_until="2099-01-01",  # far future ensures all rows land in train
        days_forward=7,
        pct_threshold=0.02,
        days_before=5,
        days_after=5,
        split_frac=None,
        require_outlook=False,
        min_numeric=0,
        out_dir=os.path.join(os.path.dirname(__file__), "..", "models_test"),
        fetch_ticker=None,
        start_year=None,
        end_year=None,
    )
    # run train and capture stdout; should not raise and should mention transcripts
    train(ns)
    out, err = capsys.readouterr()
    assert "Auto-fetched" in out or "transcripts" in out
    assert "warning: no data after cutoff" in out

    # check that step files were created (even if empty)
    base = os.path.join(os.path.dirname(__file__), "..", "models_test")
    for fname in ["step1_transcripts.csv", "step2_window.csv", "step3_features.csv"]:
        assert os.path.exists(os.path.join(base, fname)), f"{fname} missing"

    # run again with an outlook requirement to force filtering
    ns.require_outlook = True
    train(ns)
    out2, err2 = capsys.readouterr()
    assert "Filtered" in out2

    # verify the automatic transcripts file (if created) does not contain any
    # future dates
    import pandas as pd
    path = os.path.join(os.path.dirname(__file__), "..", "models_test", "input_transcripts.csv")
    if os.path.exists(path):
        df2 = pd.read_csv(path)
        if not df2.empty:
            assert pd.to_datetime(df2["date"]).max() <= pd.to_datetime("today")


def test_fetch_transcripts_cli(capsys):
    """When --fetch-ticker is provided the pipeline should download data."""
    import argparse
    from main import train

    ns = argparse.Namespace(
        transcripts=None,
        prices=os.path.join(os.path.dirname(__file__), "..", "data", "sample_aapl_prices.csv"),
        train_until="2025-01-01",
        days_forward=7,
        pct_threshold=0.02,
        days_before=5,
        days_after=5,
        split_frac=None,
        require_outlook=False,
        min_numeric=0,
        out_dir=os.path.join(os.path.dirname(__file__), "..", "models_test"),
        fetch_ticker="AAPL",
        start_year=2021,
        end_year=2022,
    )
    # running this test requires earningscall; skip gracefully if unavailable
    try:
        train(ns)
    except ImportError:
        # earningscall not available, nothing to assert
        return
    out, err = capsys.readouterr()
    assert "Fetched" in out
    base = os.path.join(os.path.dirname(__file__), "..", "models_test")
    for fname in ["step1_transcripts.csv", "step2_window.csv", "step3_features.csv"]:
        assert os.path.exists(os.path.join(base, fname)), f"{fname} missing after fetch"


if __name__ == "__main__":
    test_earningscall_fetch()
    test_load_sample_csv()
    test_alignment_and_model_pipeline()
