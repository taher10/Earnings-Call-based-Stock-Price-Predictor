import re
from typing import Optional, Tuple
import nltk
from nltk.stem import WordNetLemmatizer


# Hedging language (Management Obfuscation indicators)
HEDGING_WORDS = [
    "potentially", "approximately", "appears", "appears to be",
    "seems", "might", "may", "could", "possibly", "arguably",
    "largely", "mainly", "partly", "somewhat", "relatively",
    "sort of", "kind of", "rather", "fairly", "pretty",
    "believe", "think", "expect", "anticipate", "suggest",
    "likely", "unlikely", "uncertain", "uncertain", "unclear",
    "subject to", "dependent on", "contingent", "if conditions",
    "to some extent", "in some cases", "at times", "some",
    "estimates", "estimated", "projected", "guidance", "target",
]

# Apple products and components for entity masking
PRODUCT_NAMES = {
    # iPhones
    "iphone 15 pro max": "[PRODUCT_GEN]",
    "iphone 15 pro": "[PRODUCT_GEN]",
    "iphone 15": "[PRODUCT_GEN]",
    "iphone 14 pro max": "[PRODUCT_GEN]",
    "iphone 14 pro": "[PRODUCT_GEN]",
    "iphone 14": "[PRODUCT_GEN]",
    "iphone 13": "[PRODUCT_GEN]",
    "iphone 12": "[PRODUCT_GEN]",
    "iphone 11": "[PRODUCT_GEN]",
    # Chips/Processors
    "a18": "[CHIP_GEN]",
    "a18 pro": "[CHIP_GEN]",
    "a17": "[CHIP_GEN]",
    "a16": "[CHIP_GEN]",
    "a15": "[CHIP_GEN]",
    "m4": "[CHIP_GEN]",
    "m3": "[CHIP_GEN]",
    "m2": "[CHIP_GEN]",
    "m1": "[CHIP_GEN]",
    "a17 pro": "[CHIP_GEN]",
    # Services/Features
    "apple intelligence": "[FEATURE_GEN]",
    "apple pay": "[FEATURE_GEN]",
    "icloud": "[FEATURE_GEN]",
}



# keep a module‑level flag so we only download once per session; the NLTK
# check is relatively expensive and was causing repeated log spam during
# batch scoring.
_nltk_ready = False

class DataCleaning:
    """Simple cleaning utilities for earnings transcripts.

    - removes timestamps and common speaker labels
    - masks entity names (products, chips) to prevent overfitting
    - extracts Q&A section for stress metrics
    - lowercases, removes non-alphanumeric (but keeps spaces), lemmatizes
    """

    def __init__(self):
        global _nltk_ready
        if not _nltk_ready:
            try:
                nltk.data.find("tokenizers/punkt")
            except Exception:
                nltk.download("punkt")
            try:
                nltk.data.find("corpora/wordnet")
            except Exception:
                nltk.download("wordnet")
            _nltk_ready = True
        self.lemmatizer = WordNetLemmatizer()

    def mask_entities(self, text: str) -> str:
        """Replace product/chip/service names with generic placeholders.
        
        This forces the model to learn the context around products rather
        than overfitting to specific product names.
        """
        if not isinstance(text, str):
            return ""
        
        result = text.lower()
        # Sort by length (longest first) to avoid partial replacements
        for entity, placeholder in sorted(PRODUCT_NAMES.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(entity) + r'\b', placeholder, result, flags=re.IGNORECASE)
        
        return result

    def extract_qa_section(self, text: str) -> Tuple[str, str]:
        """Split transcript into management section and Q&A section.
        
        Q&A typically starts with "Questions and Answers", "Q&A", "Operator:", or
        the first analyst question pattern like "Name with Company".
        
        Returns: (management_text, qa_text)
        """
        if not isinstance(text, str):
            return text, ""
        
        # Look for common Q&A section markers
        qa_markers = [
            r'questions?\s+and?\s+answers?',
            r'^\s*operator',
            r'^\s*[a-z\s]+,\s+[a-z\s]+\s+research',
            r'^\s*[a-z\s]+\s+with\s+[a-z\s]+',
        ]
        
        for pattern in qa_markers:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                split_pos = match.start()
                return text[:split_pos], text[split_pos:]
        
        # If no marker found, return full text as management
        return text, ""

    def measure_hedging_density(self, text: str) -> float:
        """Quantify 'Management Obfuscation' by calculating hedging language density.
        
        Count occurrences of hedging words/phrases and return density as a percentage.
        Higher density indicates more cautious/obfuscating language.
        
        Returns:
            Hedging density (0.0 to 1.0 scale)
        """
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        
        text_lower = text.lower()
        hedging_count = 0
        
        for hedge in HEDGING_WORDS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(hedge) + r'\b'
            matches = re.findall(pattern, text_lower)
            hedging_count += len(matches)
        
        # Normalize by word count
        word_count = len(text_lower.split())
        if word_count == 0:
            return 0.0
        
        density = min(hedging_count / word_count, 1.0)  # Cap at 1.0
        return density

    def measure_linguistic_complexity(self, text: str) -> float:
        """Measure linguistic complexity (entropy) via average sentence length.
        
        Hedge funds use this as a proxy for "Management Defensiveness"—if a CEO
        starts using very long, complex sentences to answer a simple question,
        it's often a bearish signal. Long rambling answers suggest evasion.
        
        Returns:
            Average sentence length in words (0.0 if text empty)
        """
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        
        # Split into sentences using common punctuation
        # This is a simple heuristic; more sophisticated NLP could use spaCy
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Measure average sentence length (in words)
        total_words = 0
        for sentence in sentences:
            words = sentence.split()
            total_words += len(words)
        
        avg_sentence_length = total_words / len(sentences)
        return avg_sentence_length

    def extract_guidance_and_results(self, text: str) -> Tuple[str, str]:
        """Extract 'Guidance/Outlook' section and 'Results' section for divergence detection.
        
        Guidance/Outlook typically appears near the end of prepared remarks:
        'going forward', 'guidance', 'outlook', 'we expect', 'we believe'
        
        Results typically in early sections: 'revenue', 'earnings', 'grew', 'increased'
        
        Returns:
            (results_text, guidance_text)
        """
        if not isinstance(text, str) or len(text) == 0:
            return text, ""
        
        text_lower = text.lower()
        
        # Find guidance markers (typically appear after results discussion)
        guidance_markers = [
            r'(?:fiscal\s+year\s+)?guidance',
            r'(?:fiscal\s+)?outlook',
            r'going\s+forward',
            r'we\s+(?:expect|believe|anticipate)',
            r'(?:for\s+the\s+(?:next|following))',
        ]
        
        guidance_start = None
        for pattern in guidance_markers:
            match = re.search(pattern, text_lower)
            if match:
                guidance_start = match.start()
                break
        
        if guidance_start is None:
            # No guidance section found; return full text as results, empty guidance
            return text, ""
        
        # Split at guidance start
        results_text = text[:guidance_start]
        guidance_text = text[guidance_start:]
        
        return results_text, guidance_text

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
        cleaned = " ".join(lem)
        # collapse any remaining runs of whitespace (including newlines)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
