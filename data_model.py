from typing import Optional, Tuple
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class TextModel:
    """Trainable text model predicting buy/sell/hold from transcripts.

    Uses TF-IDF + LogisticRegression. Ensures temporal split for train/test to
    avoid lookahead bias: `train_until` provides the cutoff date (inclusive) for training.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def temporal_split(self, df: pd.DataFrame, train_until: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by a cutoff date (inclusive train, exclusive test).

        This is the original behaviour used by ``main.py`` and is necessary to
        avoid look‑ahead bias when the user supplies an absolute timestamp as
        ``--train-until``.  If you prefer an 80/20 pro rata split you can call
        :meth:`proportion_split` instead.
        """
        df2 = df.dropna(subset=["label", "transcript", "date"]).copy()
        df2["date"] = pd.to_datetime(df2["date"])
        cutoff = pd.to_datetime(train_until)
        train = df2[df2["date"] <= cutoff].copy()
        test = df2[df2["date"] > cutoff].copy()
        return train, test

    def proportion_split(self, df: pd.DataFrame, frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return a temporal split that places ``frac`` of the examples in train.

        The split is performed on the sorted ``date`` column in order to maintain
        the time‑series ordering; it is not a random shuffle.  ``frac`` must be
        between 0 and 1.  The caller can choose this when an explicit cutoff date
        isn’t known or when a simple 80/20 division is sufficient for prototyping.
        """
        if not 0 < frac < 1:
            raise ValueError("frac must be between 0 and 1")
        df2 = df.dropna(subset=["label", "transcript", "date"]).copy()
        df2["date"] = pd.to_datetime(df2["date"])
        df2 = df2.sort_values("date")
        cutoff = df2["date"].quantile(frac)
        train = df2[df2["date"] <= cutoff].copy()
        test = df2[df2["date"] > cutoff].copy()
        return train, test

    def fit(self, train_df: pd.DataFrame, text_col: str = "transcript", label_col: str = "label"):
        # similar to ``evaluate``, guard against a DataFrame being returned
        # when pandas finds multiple columns with the same name.  this can
        # happen earlier in the pipeline if the original and cleaned versions
        # of the transcript both end up as ``transcript`` in the dataframe.
        X_col = train_df[text_col]
        if isinstance(X_col, pd.DataFrame):
            X = X_col.iloc[:, 0].fillna("")
        else:
            X = X_col.fillna("")
        y = train_df[label_col]
        self.pipeline.fit(X, y)

    def evaluate(self, test_df: pd.DataFrame, text_col: str = "transcript", label_col: str = "label") -> str:
        """Return a scikit-learn classification report for the test set.

        If ``test_df`` is empty we simply return a helpful message rather than
        letting the vectorizer crash with a zero-sample error.  The caller can
        decide whether to treat it as fatal.
        """
        if test_df.empty:
            return "[no test data; evaluation skipped]"

        X_col = test_df[text_col]
        if isinstance(X_col, pd.DataFrame):
            X = X_col.iloc[:, 0].fillna("").tolist()
        else:
            X = X_col.fillna("").tolist()
        y = list(test_df[label_col])
        preds = self.pipeline.predict(X)
        return classification_report(y, preds, zero_division=0)

    def predict(self, text: str) -> int:
        pred = self.pipeline.predict([text])[0]
        return int(pred)

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        self.pipeline = joblib.load(path)
