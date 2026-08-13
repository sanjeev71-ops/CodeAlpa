"""FAQ chatbot core: TF-IDF + cosine similarity with a confidence score.

Robustness: the match score combines two signals that both rely on shared
vocabulary, so an unrelated FAQ can never score well:
  * TF-IDF cosine similarity (weighted lexical match)
  * token-overlap Jaccard on the FAQ question (rescues cases where TF-IDF
    dilution lowered the cosine below threshold even though the words match)

Synonym/alias normalization happens in preprocess.py, so variants like
"pw", "card", or "broken" still hit the right FAQ.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import FAQS
from preprocess import preprocess, preprocess_str

# Confidence below this is treated as "no good match" (fallback answer).
THRESHOLD = 0.15


def _build_corpus(faqs):
    # Index the whole FAQ entry (question + answer) so answer-only terms
    # (e.g. "Mastercard") still help retrieve the entry.
    return [preprocess_str(f["question"] + " " + f["answer"]) for f in faqs]


def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class FAQBot:
    def __init__(self, faqs=FAQS, threshold=THRESHOLD):
        self.faqs = faqs
        self.threshold = threshold
        self.corpus = _build_corpus(faqs)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        # Preprocessed question tokens, used for the Jaccard signal.
        self.q_tokens = [set(preprocess(f["question"])) for f in faqs]

    def _scores(self, query):
        q = (query or "").strip()
        if not q:
            return np.zeros(len(self.faqs)), np.zeros(len(self.faqs))
        q_vec = self.vectorizer.transform([q])
        cos = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        q_set = set(preprocess(q))
        jac = np.array([_jaccard(q_set, qt) for qt in self.q_tokens])
        # Combined: keep the stronger of the two signals.
        combined = np.maximum(cos, jac)
        return combined, cos

    def respond(self, query):
        combined, cos = self._scores(query)
        if len(combined) == 0 or combined.max() < self.threshold:
            return self._fallback(combined)

        best_idx = int(np.argmax(combined))
        return {
            "answer": self.faqs[best_idx]["answer"],
            "confidence": float(combined[best_idx]),
            "matched_question": self.faqs[best_idx]["question"],
            "index": best_idx,
            "alternatives": self._top_alternatives(combined, best_idx),
        }

    def _top_alternatives(self, combined, exclude_idx, top_n=3):
        order = np.argsort(-combined)
        alts = []
        for idx in order:
            if idx == exclude_idx:
                continue
            if len(alts) >= top_n:
                break
            alts.append(
                {
                    "question": self.faqs[idx]["question"],
                    "confidence": float(combined[idx]),
                }
            )
        return alts

    def _fallback(self, combined):
        # Even when below threshold, surface the closest few as suggestions.
        alts = self._top_alternatives(combined, exclude_idx=-1)
        best_idx = int(np.argmax(combined)) if len(combined) else -1
        return {
            "answer": (
                "Sorry, I couldn't find a matching answer. "
                "Please try rephrasing your question or contact support."
            ),
            "confidence": float(combined.max()) if len(combined) else 0.0,
            "matched_question": None,
            "index": best_idx,
            "alternatives": alts,
        }


if __name__ == "__main__":
    bot = FAQBot()
    samples = [
        "How do I reset my password?",
        "where is my order",
        "can i get a refund for a broken item",
        "do you take mastercard",
        "what is the meaning of life",
        "",
        "i forgot my pasword and need to log back in",
        "is there a coupon for students",
    ]
    for s in samples:
        r = bot.respond(s)
        print(f"\nQ: {s!r}")
        print(f"  A: {r['answer']}")
        print(f"  confidence={r['confidence'] * 100:.1f}%  match={r['matched_question']}")
        for a in r["alternatives"]:
            print(f"    alt: {a['confidence'] * 100:.1f}%  {a['question']}")
