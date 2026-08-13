"""NLTK-based preprocessing: lowercase, tokenize, strip punctuation,
remove stopwords, apply synonym/alias normalization, then stem.

Used by the TF-IDF vectorizer as the tokenizer. The SYNONYMS map converts
common variants (e.g. "pw" -> "password", "card" -> "payment", "broken"
-> "defective") so that matching is more robust to phrasing differences.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()

# Normalise common alternative words/abbreviations to a canonical token.
SYNONYMS = {
    "pw": "password",
    "pswd": "password",
    "passcode": "password",
    "passwort": "password",
    "pasword": "password",
    "acct": "account",
    "accounts": "account",
    "ship": "shipping",
    "deliver": "shipping",
    "delivery": "shipping",
    "card": "payment",
    "pay": "payment",
    "login": "login",
    "log": "login",
    "signin": "login",
    "sign": "login",
    "reset": "reset",
    "track": "track",
    "tracking": "track",
    "refund": "refund",
    "return": "return",
    "returns": "return",
    "warranty": "warranty",
    "warrant": "warranty",
    "cancel": "cancel",
    "contact": "contact",
    "support": "support",
    "broken": "defective",
    "defect": "defective",
    "defective": "defective",
    "discount": "discount",
    "coupon": "discount",
    "promo": "discount",
    "promocode": "discount",
    "newsletter": "newsletter",
    "subscribe": "newsletter",
    "subscription": "newsletter",
    "delete": "delete",
    "remove": "delete",
    "removal": "delete",
}


def preprocess(text):
    """Return a list of cleaned, normalized, stemmed tokens for the text."""
    if not text:
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        tok = re.sub(r"[^a-z0-9]", "", tok)  # keep only alphanumeric characters
        if not tok:
            continue
        tok = SYNONYMS.get(tok, tok)  # alias normalization
        if tok in _STOPWORDS:
            continue
        cleaned.append(_STEMMER.stem(tok))
    return cleaned


def preprocess_str(text):
    """Same as preprocess but joins tokens into a single space-separated string."""
    return " ".join(preprocess(text))
