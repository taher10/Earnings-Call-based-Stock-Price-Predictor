import os
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



def _try_import_yfinance():
    try:
        import yfinance as yf
        return yf
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

    # scraping helper --------------------------------------------------
    def fetch_transcripts_seekingalpha(self, ticker: str) -> pd.DataFrame:
        """Scrape earnings call transcripts for *ticker* from SeekingAlpha.

        SeekingAlpha publishes each transcript as a separate HTML page rather
        than a bulk CSV, so this helper downloads the listing page, follows the
        links, and returns a dataframe with columns ``company``, ``date``, and
        ``transcript``.  You can then feed the returned ``DataFrame`` directly
        into :meth:`align_and_label` or write it out with
        ``df.to_csv(..., index=False)`` for later use.

        This method requires the ``requests`` and ``beautifulsoup4`` packages
        (added to ``requirements.txt``).  If they are not installed, an
        ``ImportError`` is raised with installation instructions.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "requests and beautifulsoup4 are required for fetching transcripts; "
                "install with `pip install requests beautifulsoup4`"
            ) from e

        base = "https://seekingalpha.com"
        url = f"{base}/symbol/{ticker}/earnings/transcripts"
        # the site blocks simple bots; use a realistic browser UA and a session
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": base,
        }
        session = requests.Session()
        resp = session.get(url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = []
        # the page uses `<a class="quoted" ...>` for each transcript link
        for link in soup.select("a.quoted"):
            href = link.get("href")
            if not href:
                continue
            span = link.find_previous("span")
            if span is None:
                continue
            date_str = span.get_text(strip=True)
            try:
                date = pd.to_datetime(date_str)
            except Exception:
                # skip rows with unparseable dates
                continue
            # fetch individual transcript page
            page = requests.get(base + href, headers=headers)
            page.raise_for_status()
            page_soup = BeautifulSoup(page.text, "html.parser")
            content = page_soup.select_one("div.article-content")
            if not content:
                continue
            text = content.get_text("\n", strip=True)
            rows.append({"company": ticker, "date": date, "transcript": text})
        return pd.DataFrame(rows)

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
            rows.append({"company": ticker, "date": date, "transcript": tr.text})
        return pd.DataFrame(rows)

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
        return out
