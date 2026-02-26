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
        df2 = df.dropna(subset=["label", "transcript", "date"]).copy()
        df2["date"] = pd.to_datetime(df2["date"])
        cutoff = pd.to_datetime(train_until)
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
