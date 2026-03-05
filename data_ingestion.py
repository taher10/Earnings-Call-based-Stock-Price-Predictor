import os
import re
from pathlib import Path
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

    def _transcript_cache_dir(self) -> Path:
        return Path(__file__).resolve().parent / "data" / "transcripts" / "cache"

    def _kaggle_transcript_roots(self) -> List[Path]:
        """Candidate roots for local Kaggle transcript text files."""
        base = Path(__file__).resolve().parent / "Kaggle Dataset" / "archive"
        return [
            base / "cleaned_ECTs_dataset",
            base / "NLP_Dataset" / "NLP_Dataset",
            base / "NLP_Dataset",
        ]

    def _ticker_aliases(self, ticker: str) -> List[str]:
        t = str(ticker).upper().strip()
        aliases = [t]
        manual = {
            "GOOG": ["GOOGL"],
            "GOOGL": ["GOOG"],
            "META": ["FB"],
            "FB": ["META"],
        }
        aliases.extend(manual.get(t, []))
        if "-" in t:
            aliases.append(t.replace("-", ""))
            aliases.append(t.replace("-", "."))
        if "." in t:
            aliases.append(t.replace(".", ""))
            aliases.append(t.replace(".", "-"))
        # preserve order + dedupe
        return list(dict.fromkeys(aliases))

    def _quarter_close_timestamp(self, year: int, quarter: int) -> pd.Timestamp:
        month = int(quarter) * 3
        dt = pd.Timestamp(year=int(year), month=month, day=1) + pd.offsets.MonthEnd(0)
        # Use market-close-ish timestamp, timezone naive (pipeline convention)
        return dt + pd.Timedelta(hours=21)

    def _fetch_transcripts_from_local_kaggle(
        self,
        ticker: str,
        years: Optional[List[int]] = None,
        quarters: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Load transcript files from local Kaggle archive for a ticker."""
        roots = [p for p in self._kaggle_transcript_roots() if p.exists()]
        if not roots:
            return pd.DataFrame(columns=["ticker", "year", "quarter", "date", "transcript", "content"])

        year_filter = set(int(y) for y in years) if years else None
        quarter_filter = set(int(q) for q in quarters) if quarters else None

        parsed = {}
        pattern = re.compile(r"(?P<year>\d{4})_Q(?P<quarter>[1-4])_(?P<sym>[a-zA-Z0-9\-\.]+)_processed\.txt$")

        for root in roots:
            for alias in self._ticker_aliases(ticker):
                file_glob = f"**/*_{alias.lower()}_processed.txt"
                for file_path in root.glob(file_glob):
                    name = file_path.name
                    match = pattern.match(name)
                    if not match:
                        continue
                    year = int(match.group("year"))
                    quarter = int(match.group("quarter"))
                    if year_filter and year not in year_filter:
                        continue
                    if quarter_filter and quarter not in quarter_filter:
                        continue

                    key = (year, quarter)
                    # Prefer files found earlier in root search order (cleaned first)
                    if key in parsed:
                        continue
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if not text:
                        continue

                    parsed[key] = {
                        "ticker": str(ticker).upper().strip(),
                        "year": year,
                        "quarter": quarter,
                        "date": self._quarter_close_timestamp(year, quarter),
                        "transcript": text,
                        "content": text,
                    }

        rows = [parsed[k] for k in sorted(parsed.keys())]
        return pd.DataFrame(rows, columns=["ticker", "year", "quarter", "date", "transcript", "content"])

    def _transcript_cache_path(self, ticker: str, year: int, quarter: int) -> Path:
        t = str(ticker).upper().strip()
        return self._transcript_cache_dir() / f"{t}_{int(year)}_Q{int(quarter)}.csv"

    def _load_cached_transcript(
        self,
        ticker: str,
        year: int,
        quarter: int,
    ) -> Optional[dict]:
        path = self._transcript_cache_path(ticker, year, quarter)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return None
        row = df.iloc[0]
        txt = row.get("transcript")
        if pd.isna(txt) or txt is None:
            txt = row.get("content")
        if pd.isna(txt) or txt is None:
            return None
        dt = row.get("date")
        if pd.isna(dt) or dt is None:
            return None
        return {
            "ticker": str(row.get("ticker", ticker)).upper(),
            "year": int(row.get("year", year)),
            "quarter": int(row.get("quarter", quarter)),
            "date": pd.to_datetime(dt, utc=True).tz_convert(None),
            "transcript": str(txt),
            "content": str(row.get("content", txt)),
        }

    def _save_cached_transcript(self, row: dict) -> None:
        cache_dir = self._transcript_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame([
            {
                "ticker": str(row["ticker"]).upper(),
                "year": int(row["year"]),
                "quarter": int(row["quarter"]),
                "date": pd.to_datetime(row["date"], utc=True).tz_convert(None),
                "transcript": str(row["transcript"]),
                "content": str(row.get("content", row["transcript"])),
            }
        ])
        out.to_csv(
            self._transcript_cache_path(out.iloc[0]["ticker"], out.iloc[0]["year"], out.iloc[0]["quarter"]),
            index=False,
        )

    def _normalize_yfinance_download(self, data: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
        """Return a normalized dataframe with ``date`` and scalar ``close`` columns.

        Recent ``yfinance`` versions can return MultiIndex columns even for a single
        ticker (for example ``("Close", "AAPL")`` and ``("Close", "MSFT")``), and
        selecting ``"Close"`` directly can produce a DataFrame instead of a Series.
        This helper flattens that output into one close series for the requested ticker.
        """
        df = data.copy()

        # flatten MultiIndex columns to simple strings where possible
        if isinstance(df.columns, pd.MultiIndex):
            if ticker is not None and ("Close", ticker) in df.columns:
                close_series = df[("Close", ticker)]
            elif ("Close", "") in df.columns:
                close_series = df[("Close", "")]
            else:
                close_cols = [c for c in df.columns if str(c[0]).lower() == "close"]
                if not close_cols:
                    raise ValueError("yfinance response did not include a Close column")
                close_series = df[close_cols[0]]
            out = pd.DataFrame({"close": close_series.values}, index=df.index)
        else:
            # normal single-level output
            close_col = None
            for c in df.columns:
                if str(c).lower() == "close":
                    close_col = c
                    break
            if close_col is None:
                raise ValueError("yfinance response did not include a Close column")
            out = pd.DataFrame({"close": pd.to_numeric(df[close_col], errors="coerce")}, index=df.index)

        out = out.reset_index()
        # yfinance may expose date index as Date/Datetime/index depending on source
        date_col = None
        for candidate in ("Date", "Datetime", "index"):
            if candidate in out.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = out.columns[0]
        out = out.rename(columns={date_col: "date"})
        out["date"] = pd.to_datetime(out["date"])
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["date", "close"])
        return out[["date", "close"]]

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
        data = self._normalize_yfinance_download(data, ticker=ticker)
        data["ticker"] = ticker
        return data[["date", "ticker", "close"]]

    def fetch_spy_prices(self, start: str, end: str) -> pd.DataFrame:
        """Fetch daily close prices for SPY (S&P 500 ETF) to use as market benchmark.
        
        Returns DataFrame with columns `date`, `close` (no ticker column).
        This requires `yfinance` to be installed; if not installed, raises ImportError.
        """
        yf = _try_import_yfinance()
        if yf is None:
            raise ImportError("yfinance is required for fetch_spy_prices; install with `pip install yfinance`")
        data = yf.download("SPY", start=start, end=end, progress=False)
        data = self._normalize_yfinance_download(data, ticker="SPY")
        return data[["date", "close"]]

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
        timeout: Optional[float] = None,
    ) -> pd.DataFrame:
        """Fetch transcripts from local Kaggle archive and hydrate cache files.

        This keeps the external interface unchanged for the pipeline while switching
        the data source from remote API pulls to local transcript text files.
        """
        _ = timeout  # kept for backward-compatible signature
        ticker = str(ticker).upper().strip()

        df = self._fetch_transcripts_from_local_kaggle(
            ticker=ticker,
            years=years,
            quarters=quarters,
        )

        if df.empty:
            print(f"warning: no local Kaggle transcripts found for {ticker}")
            return pd.DataFrame(columns=["ticker", "year", "quarter", "date", "transcript", "content"])

        # Ensure normalized date format and write each quarter to cache files.
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        for _, row in df.iterrows():
            self._save_cached_transcript(
                {
                    "ticker": row["ticker"],
                    "year": int(row["year"]),
                    "quarter": int(row["quarter"]),
                    "date": row["date"],
                    "transcript": row["transcript"],
                    "content": row.get("content", row["transcript"]),
                }
            )

        print(f"Local Kaggle transcripts: loaded {len(df)} row(s) for {ticker}")
        return df[["ticker", "year", "quarter", "date", "transcript", "content"]]

    def compute_volatility(self, prices: pd.DataFrame, as_of: pd.Timestamp, window: int = 30) -> Optional[float]:
        """Return historical volatility (std of daily pct change) up to `as_of`.

        ``window`` is the number of most recent trading days to include; if fewer
        days are available we use whatever exists.  The caller may pass an entire
        prices DataFrame once and call this repeatedly; internally it filters by
        date each time.
        """
        if prices.empty:
            return None
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df[df["date"] <= as_of]
        if df.empty:
            return None
        df["ret"] = df["close"].pct_change()
        recent = df["ret"].dropna().tail(window)
        if recent.empty:
            return 0.0
        return float(recent.std())

    def align_and_label(self, transcripts: pd.DataFrame, prices: pd.DataFrame, days_forward: int = 7, pct_threshold: float = 0.02, spy_prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """For each transcript, compute percent return over `days_forward` trading days and label.

        Labels: 1 = buy (future return >= pct_threshold), 0 = sell (<= -pct_threshold), 2 = hold otherwise.
        Rather than use an absolute percentage threshold we normalise the return by
        the stock's historical volatility at the time of the call.  The value
        assigned to ``pct_threshold`` is interpreted as *volatility multiples*
        (e.g. ``2.0`` means two standard deviations).

        If ``spy_prices`` is provided, labels are computed using residual return
        (stock return - SPY return) instead of absolute stock return. This predicts
        whether the stock beats the market rather than just if price goes up.

        The function merges on company/ticker if present; otherwise it uses only dates.
        """
        t = transcripts.copy()
        p = prices.copy()
        
        # Prepare SPY prices if provided
        spy_lookup = None
        if spy_prices is not None:
            s = spy_prices.copy()
            s["date"] = pd.to_datetime(s["date"])
            s = s.sort_values("date")
            spy_lookup = s.set_index("date")["close"].sort_index()

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
                return pd.Series({"future_return": None, "vol_adj_return": None, "residual_return": None, "label": None})
            future_close = future_prices.iloc[0]["close"]
            fut_ret = (future_close - start_price) / start_price
            
            # Compute residual return if SPY prices provided
            residual_ret = None
            if spy_lookup is not None:
                try:
                    # Get SPY price at start and future dates
                    spy_start = None
                    spy_future = None
                    
                    # Find SPY price on or before start_date
                    spy_at_start = spy_lookup[spy_lookup.index <= start_date]
                    if not spy_at_start.empty:
                        spy_start = spy_at_start.iloc[-1]
                    
                    # Find SPY price on or after future_date
                    spy_at_future = spy_lookup[spy_lookup.index >= future_date]
                    if not spy_at_future.empty:
                        spy_future = spy_at_future.iloc[0]
                    
                    if spy_start is not None and spy_future is not None:
                        spy_ret = (spy_future - spy_start) / spy_start
                        residual_ret = fut_ret - spy_ret
                except Exception:
                    pass
            
            # compute volatility on start_date (e.g. 30-day window)
            vol = self.compute_volatility(p, start_date)
            vol_adj = None
            if vol and vol > 0:
                vol_adj = fut_ret / vol

            # Use residual return for labeling if available, otherwise use absolute return
            return_to_classify = residual_ret if residual_ret is not None else fut_ret
            
            # classify based on volatility‑adjusted return or residual return
            if vol_adj is not None:
                if return_to_classify >= pct_threshold:
                    lbl = 1
                elif return_to_classify <= -pct_threshold:
                    lbl = 0
                else:
                    lbl = 2
            else:
                # fallback to raw threshold if volatility missing
                if return_to_classify >= pct_threshold:
                    lbl = 1
                elif return_to_classify <= -pct_threshold:
                    lbl = 0
                else:
                    lbl = 2
            return pd.Series({"future_return": fut_ret, "residual_return": residual_ret, "vol_adj_return": vol_adj, "label": lbl})

        labels = t.apply(label_row, axis=1)
        out = pd.concat([t.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
        
        # Winsorize residual returns to handle outliers (clip at 5th/95th percentiles)
        # This prevents black swan events from skewing feature weights
        if "residual_return" in out.columns:
            residual_vals = out["residual_return"].dropna()
            if len(residual_vals) > 0:
                lower_bound = residual_vals.quantile(0.05)
                upper_bound = residual_vals.quantile(0.95)
                out["residual_return_winsorized"] = out["residual_return"].clip(lower=lower_bound, upper=upper_bound)
            else:
                out["residual_return_winsorized"] = out["residual_return"]
        
        # Compute Z-score of residual returns for statistically significant labeling
        # This ensures BUY/SELL signals only trigger when outperformance is significant
        if "residual_return" in out.columns:
            residual_vals = out["residual_return"].dropna()
            if len(residual_vals) > 1:
                mean_residual = residual_vals.mean()
                std_residual = residual_vals.std()
                if std_residual > 0:
                    out["residual_zscore"] = (out["residual_return"] - mean_residual) / std_residual
                    # Re-label based on Z-score instead of raw residual return
                    # pct_threshold now interpreted as Z-score threshold (e.g., 2.0 = 2 standard deviations)
                    def _relabel_zscore(row):
                        if pd.notna(row.get("residual_zscore")):
                            zscore = row["residual_zscore"]
                            if zscore >= pct_threshold:
                                return 1  # BUY
                            elif zscore <= -pct_threshold:
                                return 0  # SELL
                            else:
                                return 2  # HOLD
                        else:
                            return row.get("label", 2)
                    out["label"] = out.apply(_relabel_zscore, axis=1)
                else:
                    out["residual_zscore"] = 0.0
            else:
                out["residual_zscore"] = None

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
