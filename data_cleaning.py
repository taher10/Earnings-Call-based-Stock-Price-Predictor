import re
from typing import Optional
import nltk
from nltk.stem import WordNetLemmatizer


class DataCleaning:
    """Simple cleaning utilities for earnings transcripts.

    - removes timestamps and common speaker labels
    - lowercases, removes non-alphanumeric (but keeps spaces), lemmatizes
    """

    def __init__(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/wordnet")
        except Exception:
            nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # remove timestamps like [00:01:23] or (00:01:23)
        text = re.sub(r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?", " ", text)
        # remove speaker labels like 'Operator:' or 'John Doe:' at line starts
        text = re.sub(r"^\s*[A-Za-z0-9 \-\.]{1,60}:", " ", text, flags=re.MULTILINE)
        # remove non-letter chars except basic punctuation preserved for later tokenization
        text = re.sub(r"[^\w\s\.,'-]", " ", text)
        text = text.lower()
        # simple lemmatization using regex tokenization to avoid heavy NLTK downloads
        tokens = re.findall(r"\w+", text)
        lem = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(lem)
