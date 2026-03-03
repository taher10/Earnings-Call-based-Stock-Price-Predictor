from typing import Optional, Tuple, Dict, List
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


# Financial earnings-speak stop words to filter out
FINANCIAL_STOP_WORDS = [
    # Forward-looking statements
    "forward", "looking", "forward-looking", "statements", "statement",
    # Time references
    "quarter", "quarterly", "year", "annual", "annually",
    # Financial metrics (too common)
    "earnings", "revenue", "gross", "margin", "margins",
    "fy", "q1", "q2", "q3", "q4",
    # Conference/presentation artifacts
    "conference", "call", "participants", "operator", "question",
    # Filler words and non-substantive speech
    "uh", "um", "ah", "er", "basically", "really", "just", "like", "kind", "sort", "obviously",
    # Common operators/question phrasing
    "question is from", "next question", "operator", "thanks", "thank",
    # Generic verbs/pronouns
    "see", "saw", "seen", "say", "said", "says",
    "believe", "believes", "believed",
    "expect", "expects", "expected", "expectation",
    "business", "company", "we", "us", "our", "they", "their", "them",
    "would", "could", "should", "may", "might",
    "will", "is", "are", "be", "going", "etc", "guidance", "outlook",
    # Generic prepositions/articles
    "the", "a", "an", "and", "or", "of", "in", "to", "at", "by", "from", "as", "with",
]

# Financial lexicon: high-value terms that should get boosted weight
FINANCIAL_LEXICON = {
    # Profitability and growth indicators
    "margin": 1.5, "margins": 1.5, "revenue": 1.3, "growth": 1.4, "profitability": 1.6,
    "earnings": 1.2, "profit": 1.5, "ebitda": 1.4, "cash flow": 1.5, "fcf": 1.6,
    "guidance": 1.7, "outlook": 1.6, "forecast": 1.5,
    # Risk/Positive indicators
    "risk": 1.2, "opportunity": 1.3, "strength": 1.4, "momentum": 1.5, "accelerate": 1.4,
    "expand": 1.3, "improve": 1.3, "optimize": 1.2, "efficiency": 1.3,
    # Demand and market share
    "demand": 1.3, "market share": 1.5, "customer": 1.2, "adoption": 1.4, "penetration": 1.4,
    # Negative indicators
    "challenge": 0.8, "headwind": 0.7, "decline": 0.6, "weakness": 0.6, "pressure": 0.7,
    "competition": 0.8, "competitor": 0.8, "obsolescence": 0.5, "inventory": 0.7,
    # Macro/currency effects
    "currency": 0.9, "macro": 0.9, "macroeconomic": 0.9, "inflation": 0.8,
    # Product/segment terms
    "inventory": 0.7, "supply": 0.9, "supply chain": 0.8, "storage": 1.2,
}



class TextModel:
    """Trainable text model predicting buy/sell/hold from transcripts.

    Uses Financial TF-IDF (with custom stop words and bigrams) + LogisticRegression.
    Ensures temporal split for train/test to avoid lookahead bias.
    Supports feature importance extraction for weighted sentiment scoring.
    """

    def __init__(self, use_financial_stopwords: bool = True):
        stop_words = FINANCIAL_STOP_WORDS if use_financial_stopwords else "english"
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=30000,                    # allow more features
                ngram_range=(1, 3),  # unigrams, bigrams, trigrams
                stop_words=stop_words,
                min_df=1,
                max_df=0.9,                             # be a bit more aggressive filtering
            )),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga"))
        ])
        self.feature_importance = None  # Will store learned feature weights
        self.feature_names = None  # Will store TF-IDF feature names
        self.sentiment_scores = []  # Will store sentiment scores for percentile tuning
        self.threshold_buy = 0.2  # Default; will be tuned based on percentiles
        self.threshold_sell = -0.2  # Default; will be tuned based on percentiles
        
        # Sentiment history for velocity calculation (rolling window)
        self.sentiment_history = {}  # ticker -> [(date, score), ...]
        
        # TRAINING BASELINE: Store mean and std of training sentiment scores
        # Used to normalize test scores relative to training distribution
        self.training_sentiment_mean = 0.0
        self.training_sentiment_std = 1.0
        
        # Dynamic threshold tracking (rolling 8-quarter percentiles)
        self.rolling_percentiles = {}  # ticker -> (threshold_buy, threshold_sell)
        
        # Product introduction dates for temporal decay
        self.feature_dates = {
            "a18": "2024-09-01",
            "a17": "2023-09-01",
            "m4": "2024-05-01",
            "m3": "2023-05-01",
            "apple intelligence": "2024-09-01",
            "iphone 15": "2023-09-01",
            "iphone 14": "2022-09-01",
            "iphone 13": "2021-09-01",
            "iphone 12": "2020-10-01",
        }
        # Weights used when combining different sentiment signals
        self.mgmt_weight = 0.6      # default emphasis on management remarks
        self.qa_weight = 0.4        # weight on Q&A sentiment
        self.full_text_weight = 0.5 # combine section-based and whole-text scores equally
        # heuristic boosts based on simple text indicators
        self.numeric_weight = 0.005   # per numeric mention
        self.outlook_boost = 0.25     # fixed boost if forward-looking language present
        
        # Meta-model and scaling
        self.meta_model = None  # Will be initialized during fit if training succeeds
        self.meta_scaler = None  # Will be initialized during fit if meta_model is trained

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

    def _scores_to_sentiment(self, arr):
        """Convert raw classifier decision_function scores to normalized sentiment.
        
        Args:
            arr: Can be scalar, zero-dim array, or array-like from decision_function
            
        Returns:
            float: Normalized sentiment score (usually in range [-1, 1] or similar)
        """
        a = np.asarray(arr)
        if a.ndim == 0:
            # zero-dim array -> extract scalar
            return float(a)
        # now we have at least 1D
        a = a.ravel()
        if a.size >= 2:
            # binary case: return prob(positive_class) - prob(negative_class)
            return float(a[1] - a[0])
        elif a.size == 1:
            return float(a[0])
        else:
            return 0.0

    def fit(self, train_df: pd.DataFrame, text_col: str = "transcript", label_col: str = "label", return_col: str = "future_return"):
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
        
        # Ensure classifier knows about all three classes [0, 1, 2] even if training data
        # doesn't have all of them. This prevents IndexError during predict() on test data.
        clf = self.pipeline.named_steps.get("clf")
        if clf is not None and hasattr(clf, "classes_"):
            clf.classes_ = np.array([0, 1, 2])
        
        # Extract feature importance after fitting
        self._extract_feature_importance()
        
        # also train a simple meta-model that learns how to weight the various
        # sectional signals.  this converts the hard‑coded averaging logic into a
        # learned linear combination.  we will collect the features below.
        self.meta_model = None

        # COMPUTE TRAINING BASELINE: Calculate sentiment distribution on training data
        # This will be used to normalize test scores relative to training
        try:
            tfidf_vec = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]
            X_vec = tfidf_vec.transform(X)
            
            # Get decision function scores for training data
            training_scores = []
            for idx in range(len(X)):
                out = clf.decision_function(X_vec[idx:idx+1])
                # handle classifier with only one class: decision_function returns a scalar
                if np.isscalar(out):
                    scores = np.array([out])
                else:
                    scores = out[0]
                # Sentiment = prob(buy) - prob(sell) if we have at least two classes
                if hasattr(scores, '__len__') and len(scores) >= 2:
                    sentiment = scores[1] - scores[0]
                else:
                    sentiment = float(scores)
                training_scores.append(sentiment)
            
            if training_scores:
                # Store mean and std of training sentiment
                self.training_sentiment_mean = np.mean(training_scores)
                self.training_sentiment_std = np.std(training_scores)
                
                # Ensure std is not zero (avoid division by zero)
                if self.training_sentiment_std < 0.01:
                    self.training_sentiment_std = 0.1
                
                print(f"Training baseline: mean={self.training_sentiment_mean:.4f}, std={self.training_sentiment_std:.4f}")
            else:
                # no scores collected; use defaults
                self.training_sentiment_mean = 0.0
                self.training_sentiment_std = 1.0
        except Exception as e:
            print(f"Warning: Could not compute training baseline: {e}")
            self.training_sentiment_mean = 0.0
            self.training_sentiment_std = 1.0

        # ------------------------------------------------------------------
        # now build meta‑model features and fit a regression to the actual returns
        try:
            from sklearn.linear_model import Ridge
            import numpy as _np

            if return_col not in train_df.columns:
                return_col = "future_return"
            meta_rows = train_df.dropna(subset=[return_col, text_col, "date"]).copy()
            if not meta_rows.empty:
                X_meta = []
                y_meta = []
                for idx, row in meta_rows.iterrows():
                    feats = self._compute_raw_features(
                        row[text_col],
                        transcript_date=pd.Timestamp(row["date"]),
                        ticker=row.get("ticker", "")
                    )
                    X_meta.append([
                        feats["full_sentiment"],
                        feats["mgmt_sentiment"],
                        feats["qa_sentiment"],
                        feats["obfuscation_penalty"],
                        feats["numeric_count"],
                        float(feats.get("outlook_flag", 0)),
                    ])
                    y_meta.append(row[return_col])
                X_meta = _np.array(X_meta)
                y_meta = _np.array(y_meta)
                
                # Scale features to prevent one feature from drowning out others
                self.meta_scaler = StandardScaler()
                X_meta_scaled = self.meta_scaler.fit_transform(X_meta)
                
                self.meta_model = Ridge(alpha=1.0)
                self.meta_model.fit(X_meta_scaled, y_meta)
                print("Meta-model trained on", len(y_meta), "examples (with z-score scaling)")
        except Exception as e:
            print(f"Warning: could not train meta-model: {e}")
            self.meta_model = None
            self.meta_scaler = None

    def _extract_feature_importance(self):
        """Extract learned feature weights from the trained LogisticRegression model."""
        try:
            # Get feature names from TF-IDF vectorizer
            self.feature_names = self.pipeline.named_steps["tfidf"].get_feature_names_out()
            
            # Get coefficients from logistic regression
            coefficients = self.pipeline.named_steps["clf"].coef_
            
            # For multi-class (buy/sell/hold), get the average absolute importance
            # coefficients shape: (n_classes, n_features)
            importance = np.mean(np.abs(coefficients), axis=0)
            
            # Create a dictionary of feature -> importance
            self.feature_importance = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
            }
            
            # Boost importance of financial lexicon terms
            for feature_name in self.feature_importance:
                feature_lower = feature_name.lower()
                # Check if feature matches any financial lexicon term
                for lexicon_term, weight in FINANCIAL_LEXICON.items():
                    if lexicon_term in feature_lower:
                        # Apply financial weight multiplier (boost or dampen)
                        self.feature_importance[feature_name] *= weight
                        break  # Only apply first matching term per feature
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            self.feature_importance = None

    def _apply_temporal_decay(self, feature_name: str, current_date: pd.Timestamp, decay_rate: float = 0.05) -> float:
        """Apply exponential decay to feature importance based on age.
        
        Older product references decay exponentially, forcing the model to
        rely on more recent terminology for prediction.
        
        Args:
            feature_name: The word/phrase (e.g., "iphone 12")
            current_date: The transcript date
            decay_rate: Decay per month (0.05 = 5% per month)
            
        Returns:
            Decay multiplier (0 to 1)
        """
        # Find matching feature date
        feature_intro = None
        for key, date_str in self.feature_dates.items():
            if key in feature_name.lower():
                feature_intro = pd.to_datetime(date_str)
                break
        
        if feature_intro is None:
            return 1.0  # No decay for unknown features
        
        # Calculate months since introduction
        months_ago = (current_date - feature_intro).days / 30.0
        
        # Exponential decay: decay_multiplier = exp(-decay_rate * months_ago)
        decay_multiplier = np.exp(-decay_rate * max(0, months_ago))
        
        return decay_multiplier

    def _compute_raw_features(self, text: str, transcript_date: pd.Timestamp, ticker: str = "AAPL") -> dict:
        """Return the raw signals used by the meta‑model.

        The dictionary includes full-transcript sentiment, section sentiments,
        obfuscation penalty, numeric counts and outlook flag.  These are
        computed using the underlying TF-IDF/classifier and cleaning helpers.
        """
        # reuse scoring logic but stop before meta-model / decay / velocity
        from data_cleaning import DataCleaning
        dc = DataCleaning()

        # management / QA split and obfuscation
        mgmt_text, qa_text = dc.extract_qa_section(text)
        qa_hedging_density = dc.measure_hedging_density(qa_text) if qa_text else 0.0
        obfuscation_penalty = 1.0 - (qa_hedging_density * 0.5)

        tfidf_vec = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]

        def _score(txt):
            vec = tfidf_vec.transform([txt])
            raw = clf.decision_function(vec)
            # reuse conversion helper
            return self._scores_to_sentiment(raw)

        full_sentiment = _score(text)
        mgmt_sentiment = _score(mgmt_text)
        qa_sentiment = _score(qa_text) if qa_text else 0.0
        if qa_sentiment < 0:
            qa_sentiment *= 2.0

        numeric_count = None
        outlook_flag = 0
        try:
            di = DataIngestion()
            numeric_count = di._count_numbers(text)
            outlook_flag = 1 if di._has_future_outlook(text) else 0
        except Exception:
            numeric_count = 0
        result = {
            "full_sentiment": float(full_sentiment),
            "mgmt_sentiment": float(mgmt_sentiment),
            "qa_sentiment": float(qa_sentiment),
            "obfuscation_penalty": float(obfuscation_penalty),
            "numeric_count": float(numeric_count),
            "outlook_flag": float(outlook_flag),
        }
        return result

    def tune_thresholds_percentile(self, scores: List[float], percentile_buy: int = 85, percentile_sell: int = 15):
        """Tune classification thresholds based on percentiles of sentiment scores.
        
        Instead of fixed thresholds (0, -0), use percentile-based bucketing:
        - Top percentile_buy% = BUY
        - Bottom percentile_sell% = SELL
        - Middle = HOLD
        
        Args:
            scores: List of sentiment scores from training set
            percentile_buy: Percentile for BUY threshold (default 85 = top 15%)
            percentile_sell: Percentile for SELL threshold (default 15 = bottom 15%)
        """
        if not scores:
            return
        
        self.threshold_buy = np.percentile(scores, percentile_buy)
        self.threshold_sell = np.percentile(scores, percentile_sell)
        
        print(f"Percentile-based thresholds tuned:")
        print(f"  BUY (top 15%):  score >= {self.threshold_buy:.4f}")
        print(f"  SELL (bottom 15%): score <= {self.threshold_sell:.4f}")
        print(f"  HOLD (middle 70%): {self.threshold_sell:.4f} < score < {self.threshold_buy:.4f}")

    def calculate_sentiment_velocity(self, ticker: str, current_date: pd.Timestamp, current_score: float, lookback_quarters: int = 4) -> float:
        """Calculate Sentiment Velocity: change in sentiment from rolling average.
        
        Compares current quarter's score to the average of the last N quarters.
        A decline in sentiment—even if still positive—generates a bearish penalty.
        
        Args:
            ticker: Stock ticker symbol
            current_date: Date of current transcript
            current_score: Current sentiment score
            lookback_quarters: Number of past quarters to average (default 4)
            
        Returns:
            Velocity multiplier (0.7 to 1.0 scale; <1.0 = momentum penalty)
        """
        if ticker not in self.sentiment_history:
            return 1.0  # No history; return neutral multiplier
        
        history = self.sentiment_history[ticker]
        
        # Find scores from the last N quarters
        quarters_back = lookback_quarters * 91  # ~91 days per quarter
        cutoff_date = current_date - pd.Timedelta(days=quarters_back)
        
        recent_scores = [score for date, score in history if date >= cutoff_date and date < current_date]
        
        if not recent_scores:
            return 1.0  # Not enough history
        
        rolling_avg = np.mean(recent_scores)
        velocity = current_score - rolling_avg
        
        # Momentum penalty: if velocity is negative, reduce score
        # Penalty scales from 1.0 (neutral) to 0.7 (strong decline)
        if velocity < 0:
            momentum_multiplier = max(0.7, 1.0 + 0.1 * velocity)  # 10% penalty per point of negative velocity
        else:
            momentum_multiplier = min(1.1, 1.0 + 0.05 * velocity)  # 5% bonus per point of positive velocity
        
        return momentum_multiplier

    def calculate_velocity_zscore(self, ticker: str, current_date: pd.Timestamp, current_score: float) -> float:
        """Calculate Z-score of current sentiment relative to TRAINING sentiment distribution.
        
        Z-score = (current_score - training_mean) / training_std
        This normalizes test scores against the training data baseline, not other test samples.
        Only |Z| > 1.0 triggers strong BUY/SELL signals (more sensitive than 1.5)
        
        Args:
            ticker: Stock ticker symbol
            current_date: Current date
            current_score: Current sentiment score
            
        Returns:
            Z-score relative to training distribution (typically -3 to +3)
        """
        # Use training baseline to normalize: how many std devs from training mean?
        zscore = (current_score - self.training_sentiment_mean) / self.training_sentiment_std
        return zscore

    def detect_guidance_divergence(self, full_text: str, management_section: str, qa_section: str) -> dict:
        """Detect if guidance is lower than results (divergence warning).
        
        Uses the DataCleaning method to extract guidance vs results sections.
        
        Returns:
            {
                'has_divergence': bool,
                'results_sentiment': float or None,
                'guidance_sentiment': float or None,
                'divergence_magnitude': float or None
            }
        """
        try:
            from data_cleaning import DataCleaning
            cleaner = DataCleaning()
            results_text, guidance_text = cleaner.extract_guidance_and_results(full_text)
            
            if not guidance_text or len(guidance_text) < 10:
                return {
                    'has_divergence': False,
                    'results_sentiment': None,
                    'guidance_sentiment': None,
                    'divergence_magnitude': None
                }
            
            # Score results and guidance sections using TF-IDF weights
            if not hasattr(self, 'vectorizer') or self.vectorizer is None:
                return {
                    'has_divergence': False,
                    'results_sentiment': None,
                    'guidance_sentiment': None,
                    'divergence_magnitude': None
                }
            
            # Transform both sections
            results_vec = self.vectorizer.transform([results_text])
            guidance_vec = self.vectorizer.transform([guidance_text])
            
            # Predict sentiment using logistic regression
            if not hasattr(self, 'model') or self.model is None:
                return {
                    'has_divergence': False,
                    'results_sentiment': None,
                    'guidance_sentiment': None,
                    'divergence_magnitude': None
                }
            
            results_prob = self.model.predict_proba(results_vec)[0]  # [prob_sell, prob_hold, prob_buy]
            guidance_prob = self.model.predict_proba(guidance_vec)[0]
            
            # Score = P(buy) - P(sell)
            results_sentiment = results_prob[2] - results_prob[0]  # buy - sell
            guidance_sentiment = guidance_prob[2] - guidance_prob[0]
            
            divergence_magnitude = results_sentiment - guidance_sentiment  # Positive = guidance lower
            has_divergence = divergence_magnitude > 0.15  # Threshold: >0.15 is significant divergence
            
            return {
                'has_divergence': has_divergence,
                'results_sentiment': float(results_sentiment),
                'guidance_sentiment': float(guidance_sentiment),
                'divergence_magnitude': float(divergence_magnitude)
            }
        except Exception as e:
            print(f"Warning: Could not detect guidance divergence: {e}")
            return {
                'has_divergence': False,
                'results_sentiment': None,
                'guidance_sentiment': None,
                'divergence_magnitude': None
            }

    def update_sentiment_history(self, ticker: str, date: pd.Timestamp, score: float):
        """Add a sentiment score to the rolling history for velocity calculation."""
        if ticker not in self.sentiment_history:
            self.sentiment_history[ticker] = []
        
        self.sentiment_history[ticker].append((date, score))
        
        # Keep only last 8 quarters (for dynamic decile labeling)
        eight_quarters = 8 * 91  # ~728 days
        cutoff = date - pd.Timedelta(days=eight_quarters)
        self.sentiment_history[ticker] = [(d, s) for d, s in self.sentiment_history[ticker] if d >= cutoff]

    def tune_thresholds_dynamic_decile(self, ticker: str, lookback_quarters: int = 8) -> Tuple[float, float]:
        """Dynamic Decile Labeling: Use rolling window percentiles for regime-aware thresholds.
        
        Calculates buy/sell thresholds based only on the last N quarters,
        allowing the model to adapt to different market regimes.
        
        Args:
            ticker: Stock ticker symbol
            lookback_quarters: Number of recent quarters to use for threshold calculation
            
        Returns:
            (threshold_buy, threshold_sell) tuple
        """
        if ticker not in self.sentiment_history or not self.sentiment_history[ticker]:
            return self.threshold_buy, self.threshold_sell
        
        # Get scores from recent history
        history = self.sentiment_history[ticker]
        quarters_back = lookback_quarters * 91
        cutoff_date = history[-1][0] - pd.Timedelta(days=quarters_back)
        
        recent_scores = [score for date, score in history if date >= cutoff_date]
        
        if len(recent_scores) < 2:
            return self.threshold_buy, self.threshold_sell
        
        # Calculate percentiles on recent window only
        threshold_buy = np.percentile(recent_scores, 85)
        threshold_sell = np.percentile(recent_scores, 15)
        
        self.rolling_percentiles[ticker] = (threshold_buy, threshold_sell)
        
        print(f"Dynamic decile thresholds ({ticker}, last {lookback_quarters}Q):")
        print(f"  BUY (top 15%):  score >= {threshold_buy:.4f}")
        print(f"  SELL (bottom 15%): score <= {threshold_sell:.4f}")
        
        return threshold_buy, threshold_sell

    def get_top_features(self, label: int, n: int = 100) -> List[Tuple[str, float]]:
        """Get top N features (words/phrases) for a specific class.
        
        label: 0=sell, 1=buy, 2=hold
        Returns: List of (feature_name, coefficient_value) tuples
        """
        try:
            coefficients = self.pipeline.named_steps["clf"].coef_[label]
            feature_names = self.pipeline.named_steps["tfidf"].get_feature_names_out()
            
            # Get indices of top features
            top_indices = np.argsort(coefficients)[-n:][::-1]
            
            return [(feature_names[i], float(coefficients[i])) for i in top_indices]
        except Exception as e:
            print(f"Could not get top features: {e}")
            return []

    def score_transcript(self, text: str) -> float:
        """Score a single transcript using learned feature importance.
        
        Returns a continuous score (not a hard label) based on weighted keywords.
        Positive score = bullish, negative = bearish, near 0 = neutral.
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before scoring")
        
        # Tokenize and weight the text
        tokens = text.lower().split()
        score = 0.0
        
        for token in tokens:
            if token in self.feature_importance:
                score += self.feature_importance[token]
        
        return score / max(len(tokens), 1)  # Normalize by text length

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

    # ------------------------------------------------------------------
    def information_coefficient(self, returns: List[float], scores: List[float]) -> float:
        """Compute Spearman rank correlation between two sequences.

        The information coefficient (IC) is simply the rank correlation
        between the signal (scores) and the future returns.  Values closer to
        1 or -1 indicate a strong monotonic relationship; 0 indicates no
        predictive power.  A hedge fund will often look for an IC of 0.05+
        consistently across time.
        """
        try:
            from scipy.stats import spearmanr
            ic = spearmanr(returns, scores).correlation
            return float(ic)
        except Exception:
            return 0.0

    def bootstrap_ic(self, df: pd.DataFrame, n_splits: int = 100, test_size: float = 0.2, text_col: str = "transcript", return_col: str = "future_return") -> Tuple[float, float]:
        """Estimate IC mean/std via repeated shuffle splits (bootstrap-like).

        Returns a tuple ``(mean_ic, std_ic)`` computed over ``n_splits``
        randomly selected train/test splits of the provided DataFrame.  This
        provides an idea of how stable the signal is and guards against
        overfitting on a small sample.
        """
        from sklearn.model_selection import ShuffleSplit
        import numpy as _np

        ics = []
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        for train_idx, test_idx in splitter.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            # fit on train only
            try:
                self.fit(train, text_col=text_col, label_col="label", return_col=return_col)
            except Exception:
                continue
            scores = []
            returns = []
            for _, row in test.iterrows():
                if pd.isna(row.get(return_col)):
                    continue
                _, score, _ = self.score_and_label(
                    row[text_col],
                    transcript_date=pd.to_datetime(row.get("date")),
                    ticker=row.get("ticker", "")
                )
                scores.append(score)
                returns.append(row.get(return_col))
            if len(scores) > 1:
                ic = self.information_coefficient(returns, scores)
                ics.append(ic)
        if not ics:
            return 0.0, 0.0
        return float(_np.mean(ics)), float(_np.std(ics))

    def predict(self, text: str) -> int:
        pred = self.pipeline.predict([text])[0]
        return int(pred)

    def score_and_label(self, text: str, transcript_date: Optional[pd.Timestamp] = None, ticker: str = "AAPL", threshold_buy: Optional[float] = None, threshold_sell: Optional[float] = None) -> Tuple[dict, float, int]:
        """Score a transcript and assign a label based on learned feature weights.
        
        Includes:
        - Temporal decay: older product terms lose importance
        - Q&A stress metric: 2x weight for negative sentiment in analyst Q&A section
        - Sentiment velocity with ASYMMETRIC WEIGHTING: -15% penalty for decline, +5% for rise
        - Management obfuscation: penalizes high hedging language density
        - Z-score normalization: only |Z| > 1.5 triggers strong BUY/SELL signals
        - Guidance divergence detection: warns if outlook sentiment < results sentiment
        
        Args:
            text: The transcript text to score
            transcript_date: Date of the earnings call (for temporal decay)
            ticker: Stock ticker (for sentiment history tracking)
            threshold_buy: Score must exceed this to assign label 1 (buy); uses self.threshold_buy if None
            threshold_sell: Score must be below this to assign label 0 (sell); uses self.threshold_sell if None
            
        Returns:
            (diagnostics_dict, score, label) where:
                - diagnostics_dict contains velocity_zscore, divergence info
                - score is the final sentiment score
                - label is 0=sell, 1=buy, 2=hold
        """
        if threshold_buy is None:
            threshold_buy = self.threshold_buy
        if threshold_sell is None:
            threshold_sell = self.threshold_sell
        
        # Default to recent date if not provided
        if transcript_date is None:
            transcript_date = pd.Timestamp.now()
        
        diagnostics = {
            'velocity_zscore': 0.0,
            'guidance_divergence': False,
            'results_sentiment': None,
            'guidance_sentiment': None,
            'divergence_magnitude': None,
        }
        
        try:
            from data_cleaning import DataCleaning
            dc = DataCleaning()
            
            # Extract management and Q&A sections
            mgmt_text, qa_text = dc.extract_qa_section(text)
            
            # Measure management obfuscation in Q&A section
            qa_hedging_density = dc.measure_hedging_density(qa_text) if qa_text else 0.0
            obfuscation_penalty = 1.0 - (qa_hedging_density * 0.5)  # 50% penalty for max hedging

            
            # Vectorize both sections separately
            tfidf_vec = self.pipeline.named_steps["tfidf"]
            mgmt_vec = tfidf_vec.transform([mgmt_text])
            qa_vec = tfidf_vec.transform([qa_text]) if qa_text else None
            
            # Get coefficients for the logistic regression model
            clf = self.pipeline.named_steps["clf"]

            full_vec = tfidf_vec.transform([text])
            raw_full = clf.decision_function(full_vec)
            full_sentiment = self._scores_to_sentiment(raw_full)

            # Management section score
            raw_mgmt = clf.decision_function(mgmt_vec)
            mgmt_sentiment = self._scores_to_sentiment(raw_mgmt)
            
            # Q&A section score (double the weight for negative sentiment, aka "stress")
            qa_sentiment = 0.0
            if qa_vec is not None and qa_text.strip():
                raw_qa = clf.decision_function(qa_vec)
                qa_sentiment = self._scores_to_sentiment(raw_qa)
                
                # If negative sentiment detected in Q&A, double the penalty (stress metric)
                if qa_sentiment < 0:
                    qa_sentiment *= 2.0
            
            # Combine section scores using configurable weights
            section_sentiment = (
                self.mgmt_weight * mgmt_sentiment
                + self.qa_weight * qa_sentiment
            )

            # average with the full-text score so that the model's learned
            # weights are always factored in
            combined_sentiment = (
                section_sentiment * self.full_text_weight
                + full_sentiment * (1 - self.full_text_weight)
            )

            # add heuristic boosts from numeric mentions / outlook language
            num_count = 0
            outlook_flag = 0
            try:
                # reuse ingestion heuristics
                from data_ingestion import DataIngestion
                helper = DataIngestion()
                num_count = helper._count_numbers(text)
                outlook_flag = 1 if helper._has_future_outlook(text) else 0
            except Exception:
                pass
            combined_sentiment += self.numeric_weight * num_count + self.outlook_boost * outlook_flag

            # Apply temporal decay to all features (modulate the final score)
            # Features from older products get less influence
            decay_multiplier = self._apply_temporal_decay("general", transcript_date)
            
            # Apply obfuscation penalty (high hedging = bearish)
            obfuscated_sentiment = combined_sentiment * obfuscation_penalty

            # if a meta-model has been trained, use it to compute the base score
            if self.meta_model is not None:
                try:
                    feat_vec = np.array([[
                        full_sentiment,
                        mgmt_sentiment,
                        qa_sentiment,
                        obfuscation_penalty,
                        float(num_count),
                        float(outlook_flag),
                    ]])
                    # Apply the same scaling that was used during training
                    if self.meta_scaler is not None:
                        feat_vec_scaled = self.meta_scaler.transform(feat_vec)
                    else:
                        feat_vec_scaled = feat_vec
                    meta_pred = self.meta_model.predict(feat_vec_scaled)[0]
                    # meta_pred represents a return prediction; apply same decay/obfuscation
                    obfuscated_sentiment = float(meta_pred) * obfuscation_penalty
                except Exception as e:
                    print(f"meta-model prediction failed: {e}")
                    # leave obfuscated_sentiment as previously computed
                    pass
            
            # Calculate Z-score relative to TRAINING sentiment distribution
            # This tells us: how extreme is this score compared to training?
            velocity_zscore = self.calculate_velocity_zscore(ticker, transcript_date, obfuscated_sentiment)
            diagnostics['velocity_zscore'] = float(velocity_zscore)
            
            # Apply sentiment velocity with ASYMMETRIC WEIGHTING (momentum multiplier)
            velocity_multiplier = self.calculate_sentiment_velocity(ticker, transcript_date, obfuscated_sentiment)
            
            # Final score combines decay, obfuscation, and velocity
            final_sentiment = obfuscated_sentiment * decay_multiplier * velocity_multiplier
            
            # Detect guidance vs. results divergence
            divergence_info = self.detect_guidance_divergence(text, mgmt_text, qa_text)
            diagnostics['guidance_divergence'] = divergence_info['has_divergence']
            diagnostics['results_sentiment'] = divergence_info['results_sentiment']
            diagnostics['guidance_sentiment'] = divergence_info['guidance_sentiment']
            diagnostics['divergence_magnitude'] = divergence_info['divergence_magnitude']
            
            # Update sentiment history for future velocity calculations
            self.update_sentiment_history(ticker, transcript_date, final_sentiment)
            
            # ASSIGN LABEL: Use threshold-based classification WITHOUT Z-score filtering
            # The thresholds (buy/sell) were tuned on training data, so use them directly
            # Z-score is provided for diagnostics only
            if final_sentiment >= threshold_buy:
                label = 1  # BUY
            elif final_sentiment <= threshold_sell:
                label = 0  # SELL
            else:
                label = 2  # HOLD

            # ensure final_sentiment is a plain Python float (avoid 0-d arrays)
            try:
                final_sentiment = float(final_sentiment)
            except Exception:
                try:
                    # try converting via numpy
                    final_sentiment = float(np.asarray(final_sentiment).item())
                except Exception:
                    final_sentiment = 0.0

            return diagnostics, final_sentiment, int(label)
            
        except Exception as e:
            print(f"Error in score_and_label: {e}")
            # Fallback to simple decision function approach
            try:
                tfidf_vec = self.pipeline.named_steps["tfidf"]
                text_vec = tfidf_vec.transform([text])
                clf = self.pipeline.named_steps["clf"]
                scores = clf.decision_function(text_vec)[0]
                sentiment = scores[1] - scores[0] if len(scores) >= 2 else scores[0]
                
                if sentiment >= threshold_buy:
                    label = 1
                elif sentiment <= threshold_sell:
                    label = 0
                else:
                    label = 2
                    
                return diagnostics, float(sentiment), int(label)
            except:
                return diagnostics, 0.0, 2  # Default to HOLD

    def save(self, path: str):
        """Save pipeline, meta-model, and scaler to disk."""
        # Save main pipeline
        joblib.dump(self.pipeline, path)
        # Save meta-model and scaler if they exist
        if self.meta_model is not None:
            base_path = path.replace('.pkl', '')
            joblib.dump(self.meta_model, f"{base_path}_meta_model.pkl")
        if self.meta_scaler is not None:
            base_path = path.replace('.pkl', '')
            joblib.dump(self.meta_scaler, f"{base_path}_meta_scaler.pkl")

    def load(self, path: str):
        """Load pipeline, meta-model, and scaler from disk."""
        self.pipeline = joblib.load(path)
        # Load meta-model and scaler if they exist
        base_path = path.replace('.pkl', '')
        try:
            self.meta_model = joblib.load(f"{base_path}_meta_model.pkl")
        except Exception:
            self.meta_model = None
        try:
            self.meta_scaler = joblib.load(f"{base_path}_meta_scaler.pkl")
        except Exception:
            self.meta_scaler = None
