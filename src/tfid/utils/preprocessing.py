"""Text preprocessing pipeline for TF-IDF classification."""

import re
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# stopwords and lemmatizer 

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalize and clean raw text for TF-IDF vectorization.

    Pipeline:
        1. Lowercase
        2. Expand contractions (e.g., "don't" -> "do not")
        3. Remove punctuation
        4. Remove digits
        5. Remove non-alphanumeric characters
        6. Tokenize
        7. Remove stopwords
        8. Lemmatize

    Args:
        text: Raw input text string.

    Returns:
        A cleaned, lemmatized string with stopwords removed.
    """
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    return " ".join(tokens)