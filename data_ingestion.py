import os
import re
from typing import Optional, List
import pandas as pd


def _try_import_yfinance():
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None


def _try_import_earningscall():
    try:
        import earningscall
        return earningscall
    except Exception:
        return None


class DataIngestion:
    """Load transcripts and stock price data, align them temporally, and create labels.

    Important: labeling uses future returns after the transcript date. To avoid lookahead
    bias, the training/testing split must be done by time; this module only aligns data
    and produces labels — the `TextModel` will perform temporal splits for training.

    Transcripts data may come with a ``company`` column (which is internally
    renamed to ``ticker``) and/or timezone-aware ``date`` values; both are
    handled transparently to match the format produced by the sample CSV file.
    """

    def __init__(self):
        pass

    def load_transcripts_csv(self, path: str) -> pd.DataFrame:
        """Load transcripts CSV with at least columns: `company` (or `ticker`),
        `date`, `transcript`.

        The ``date`` value may include a timezone offset (``-04:00`` in our sample
        CSV); the returned ``DataFrame`` normalizes all timestamps to naive
        ``datetime64`` values so that they can be merged cleanly with price data.

        ``company`` is renamed to ``ticker`` internally so that it matches the
        conventions used throughout the rest of the pipeline.  Any existing
        ``ticker`` column is left unchanged.
        """
        df = pd.read_csv(path)
        # standardize timestamp and drop tz information if present
        # some transcripts include mixed timezones ("-04:00", "-05:00");
        # parsing with ``utc=True`` avoids pandas falling back to object dtype.
        df["date"] = pd.to_datetime(df["date"], utc=True)
        # once converted to UTC we can discard the tz info
        df["date"] = df["date"].dt.tz_convert(None)

        # transcripts files often use "company" rather than "ticker"
        if "company" in df.columns and "ticker" not in df.columns:
            df = df.rename(columns={"company": "ticker"})
        return df

    def load_stock_csv(self, path: str) -> pd.DataFrame:
        """Load stock price CSV with `date` and `close` (and `ticker` optionally)."""
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_prices_yfinance(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily close prices using yfinance for a ticker between start and end dates.

        Returns DataFrame with columns `date`, `ticker`, `close`.
        This requires `yfinance` to be installed; if not installed, raises ImportError.
        """
        yf = _try_import_yfinance()
        if yf is None:
            raise ImportError("yfinance is required for fetch_prices_yfinance; install with `pip install yfinance`")
        data = yf.download(ticker, start=start, end=end, progress=False)
        data = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
        data["ticker"] = ticker
        data["date"] = pd.to_datetime(data["date"])
        return data[["date", "ticker", "close"]]

    def _has_future_outlook(self, text: str) -> bool:
        """Return True if the transcript contains language suggesting a forward outlook.

        The current heuristic looks for a small set of keywords that typically
        appear when management is discussing guidance, expectations, or plans.  It
        is intentionally lightweight so that the pipeline has no extra runtime
        dependencies.
        """
        if not isinstance(text, str):
            return False
        keywords = [
            r"\boutlook\b",
            r"\bguidance\b",
            r"\bforecast\b",
            r"\bexpect\b",
            r"\bplan\b",
            r"\bproject\b",
        ]
        pattern = re.compile("|".join(keywords), flags=re.IGNORECASE)
        return bool(pattern.search(text))

    def _count_numbers(self, text: str) -> int:
        """Count numeric mentions in the transcript (percentages, dollar figures, etc.).

        This is a very loose proxy for how much quantitative information is present.
        """
        if not isinstance(text, str):
            return 0
        # look for digits possibly followed by % or common units (m, b, k)
        return len(re.findall(r"\d+[\d\.]*\s*(%|billion|million|k|m)?", text, flags=re.IGNORECASE))

    def fetch_transcripts_earningscall(
        self,
        ticker: str,
        years: Optional[List[int]] = None,
        quarters: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Fetch transcripts via the *earningscall* Python library.

        This library provides a convenient API to pull earnings call text for
        a given ticker.  You can optionally filter by years and/or quarters.  The
        returned ``DataFrame`` has the same columns as ``load_transcripts_csv``
        and can be fed directly to :meth:`align_and_label`.

        Example::

            di = DataIngestion()
            df = di.fetch_transcripts_earningscall("AAPL", years=[2024])

        Requires ``earningscall`` to be installed; raises ``ImportError`` if not.
        """
        ec = _try_import_earningscall()
        if ec is None:
            raise ImportError(
                "earningscall is required for fetch_transcripts_earningscall; "
                "install with `pip install earningscall`"
            )
        company = ec.get_company(ticker)
        if company is None:
            raise ValueError(f"ticker {ticker} not found via earningscall")

        rows = []
        # ``today`` as a timezone-naive timestamp; event dates may have
        # tzinfo so we'll normalize both sides before comparing.
        today = pd.Timestamp.utcnow().tz_convert(None)
        for ev in company.events():
            if years and ev.year not in years:
                continue
            if quarters and ev.quarter not in quarters:
                continue
            tr = company.get_transcript(event=ev)
            if tr is None or not hasattr(tr, "text"):
                continue
            date = ev.conference_date if ev.conference_date is not None else pd.to_datetime(
                f"{ev.year}-{(ev.quarter - 1) * 3 + 1}-01"
            )
            # skip transcripts dated in the future; many corporate calendars
            # contain events scheduled well in advance which don't have an actual
            # transcript yet.
            if pd.to_datetime(date, utc=True).tz_convert(None) > today:
                continue
            rows.append({"company": ticker, "date": date, "transcript": tr.text})
        df = pd.DataFrame(rows)
        if not df.empty:
            # normalize timezone information exactly like load_transcripts_csv
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date"] = df["date"].dt.tz_convert(None)
            # align column name with rest of pipeline
            if "company" in df.columns and "ticker" not in df.columns:
                df = df.rename(columns={"company": "ticker"})
        return df

    def align_and_label(self, transcripts: pd.DataFrame, prices: pd.DataFrame, days_forward: int = 7, pct_threshold: float = 0.02) -> pd.DataFrame:
        """For each transcript, compute percent return over `days_forward` trading days and label.

        Labels: 1 = buy (future return >= pct_threshold), 0 = sell (<= -pct_threshold), 2 = hold otherwise.
        The function merges on company/ticker if present; otherwise it uses only dates.
        """
        t = transcripts.copy()
        p = prices.copy()

        # ensure transcript dates are naive datetimes (drop any tz info)
        t["date"] = pd.to_datetime(t["date"])
        try:
            if getattr(t["date"].dt, "tz", None) is not None:
                t["date"] = t["date"].dt.tz_convert(None)
        except Exception:
            pass

        # transcripts files sometimes provide a ``company`` column instead of
        # ``ticker``; the sample data uses "company" so normalize it early.
        if "company" in t.columns and "ticker" not in t.columns:
            t = t.rename(columns={"company": "ticker"})
        # likewise allow the prices file to use either name
        if "company" in p.columns and "ticker" not in p.columns:
            p = p.rename(columns={"company": "ticker"})

        # normalize column names for merging
        if "ticker" in t.columns and "ticker" in p.columns:
            merge_on = ["ticker"]
        else:
            merge_on = []
        p = p.sort_values([col for col in (merge_on + ["date"])])

        # prepare price lookup: for each date, find close price at date and at date+days_forward
        p_lookup = p.set_index([*merge_on, "date"])["close"].sort_index()

        def label_row(row):
            key = tuple(row[c] for c in merge_on) if merge_on else None
            start_date = row["date"]
            try:
                if merge_on:
                    start_price = p_lookup.loc[(*key, start_date)]
                else:
                    start_price = p_lookup.loc[(start_date,)] if isinstance(p_lookup.index, pd.MultiIndex) else p_lookup.loc[start_date]
            except Exception:
                return pd.Series({"future_return": None, "label": None})

            # find future date: nearest trading date at or after start_date + days_forward
            future_date = start_date + pd.Timedelta(days=days_forward)
            # find the next available trading close on or after future_date
            if merge_on:
                future_prices = p.loc[(p[merge_on[0]] == row[merge_on[0]]) & (p["date"] >= future_date)].sort_values("date")
            else:
                future_prices = p.loc[p["date"] >= future_date].sort_values("date")

            if future_prices.empty:
                return pd.Series({"future_return": None, "label": None})
            future_close = future_prices.iloc[0]["close"]
            fut_ret = (future_close - start_price) / start_price
            if fut_ret >= pct_threshold:
                lbl = 1
            elif fut_ret <= -pct_threshold:
                lbl = 0
            else:
                lbl = 2
            return pd.Series({"future_return": fut_ret, "label": lbl})

        labels = t.apply(label_row, axis=1)
        out = pd.concat([t.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)

        # annotate transcripts with a few simple text indicators that may be
        # useful for filtering or model features.  the naming and logic here are
        # deliberately conservative; you can expand the keyword list or adopt a
        # library such as spaCy/VADER for richer sentiment scores later.
        out["has_future_outlook"] = out["transcript"].fillna("").apply(self._has_future_outlook)
        out["numeric_count"] = out["transcript"].fillna("").apply(self._count_numbers)

        return out

    def align_with_window(
        self,
        transcripts: pd.DataFrame,
        prices: pd.DataFrame,
        days_before: int = 5,
        days_after: int = 5,
    ) -> pd.DataFrame:
        """Create a view of transcripts with a centred return window.

        For each transcript row we locate the nearest available trading close
        on the transcript date (before_close) and the nearest trading close on
        *or after* ``date + days_after`` (after_close).  The resulting ``window_return`` is
        computed as

            (after_close - before_close) / before_close

        Rows for which one of the bounding prices cannot be found are returned
        with ``NaN`` in the return columns.  This allows you to inspect the
        raw data before deciding how to handle missing values.
        """
        t = transcripts.copy()
        p = prices.copy()

        # normalize dates and tickers exactly like :meth:`align_and_label`
        t["date"] = pd.to_datetime(t["date"])
        try:
            if getattr(t["date"].dt, "tz", None) is not None:
                t["date"] = t["date"].dt.tz_convert(None)
        except Exception:
            pass

        if "company" in t.columns and "ticker" not in t.columns:
            t = t.rename(columns={"company": "ticker"})
        if "company" in p.columns and "ticker" not in p.columns:
            p = p.rename(columns={"company": "ticker"})

        if "ticker" in t.columns and "ticker" in p.columns:
            merge_on = ["ticker"]
        else:
            merge_on = []
        p = p.sort_values([col for col in (merge_on + ["date"])])
        p_lookup = p.set_index([*merge_on, "date"])["close"].sort_index()

        def window_row(row):
            key = tuple(row[c] for c in merge_on) if merge_on else None
            dt = row["date"]
            # helper to find close on or after a target date
            def find_close_after(target):
                if merge_on:
                    subset = p.loc[
                        (p[merge_on[0]] == row[merge_on[0]]) & (p["date"] >= target)
                    ].sort_values("date")
                else:
                    subset = p.loc[p["date"] >= target].sort_values("date")
                if subset.empty:
                    return None
                return float(subset.iloc[0]["close"])

            # helper to find close on or before a target date
            def find_close_before(target):
                if merge_on:
                    subset = p.loc[
                        (p[merge_on[0]] == row[merge_on[0]]) & (p["date"] <= target)
                    ].sort_values("date", ascending=False)
                else:
                    subset = p.loc[p["date"] <= target].sort_values("date", ascending=False)
                if subset.empty:
                    return None
                return float(subset.iloc[0]["close"])

            before_target = dt  # Use the transcript date itself - find most recent close on/before this date
            after_target = dt + pd.Timedelta(days=days_after)
            before_close = find_close_before(before_target)
            after_close = find_close_after(after_target)
            if before_close is None or after_close is None:
                return pd.Series({
                    "before_close": None,
                    "after_close": None,
                    "window_return": None,
                })
            win_ret = (after_close - before_close) / before_close
            return pd.Series({
                "before_close": before_close,
                "after_close": after_close,
                "window_return": win_ret,
            })

        windows = t.apply(window_row, axis=1)
        out = pd.concat([t.reset_index(drop=True), windows.reset_index(drop=True)], axis=1)

        # also add the text indicators to mirror align_and_label
        out["has_future_outlook"] = out["transcript"].fillna("").apply(self._has_future_outlook)
        out["numeric_count"] = out["transcript"].fillna("").apply(self._count_numbers)

        return out
