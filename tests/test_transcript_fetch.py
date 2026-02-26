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
    # basic sanity checks
    assert not df.empty, "no transcripts returned"
    assert "transcript" in df.columns

if __name__ == "__main__":
    test_earningscall_fetch()
